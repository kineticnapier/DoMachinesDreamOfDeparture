from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dmdod.training.timing_motor_probe import (
    _aggregate_readout_stats,
    _check_student_unchanged,
    collect_student_rollout,
    error_distribution,
    evaluate_time_readouts,
    motor_statistics,
    pair_command_events,
    predict_readout,
    save_frame_data,
    train_readout,
)


def test_command_to_physical_key_edges_and_unmatched_latency():
    edges = pair_command_events(
        [(0.0, (0.9, 0.0)), (0.1, (-0.9, 0.0))],
        [(0.040, "left", "down"), (0.136, "left", "up"),
         (0.15, "right", "down")],
        ("left", "right"),
    )
    down = next(e for e in edges if e["kind"] == "keydown" and e["key"] == "left")
    up = next(e for e in edges if e["kind"] == "keyup")
    unknown = next(e for e in edges if e["kind"] == "keydown" and e["key"] == "right")
    assert down["latency_ms"] == pytest.approx(40.0)
    assert up["latency_ms"] == pytest.approx(36.0)
    assert unknown["latency_ms"] is None


def test_pair_commands_rejects_unknown_keys_and_wrong_width():
    with pytest.raises(ValueError, match="width"):
        pair_command_events([(0, (1.0,))], [], ("left", "right"))
    with pytest.raises(ValueError, match="invalid physical"):
        pair_command_events([], [(0.1, "invalid", "down")], ("left",))


def test_error_distribution_abs_quantiles_and_empty():
    empty = error_distribution([])
    assert empty == {"n": 0, "mae_ms": None, "p50_ms": None, "p90_ms": None, "p99_ms": None}
    stats = error_distribution([-10.0, 0.0, 20.0, -30.0])
    assert stats["n"] == 4
    assert stats["mae_ms"] == pytest.approx(15.0)


def test_train_readout_does_not_mutate_inputs_and_predicts_train_labels():
    torch.manual_seed(14)
    xs = torch.linspace(-1, 1, 400).unsqueeze(1)
    features = torch.cat((xs, xs.square()), dim=1)
    truth = 0.4 + 0.2 * xs[:, 0]
    copied = features.clone()
    trained = train_readout(
        features, truth, device=torch.device("cpu"), steps=300, seed=4
    )
    pred = predict_readout(trained, features, torch.device("cpu"))
    assert torch.equal(features, copied)
    # A genuinely linear target should converge, not merely pass a loose MAE.\n    assert (pred - truth).abs().mean() < 0.01
    assert all(not parameter.requires_grad or parameter.grad is not None
               for parameter in trained.model.parameters())


def test_nonfinite_labels_are_excluded_from_readout_training():
    x = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    y = torch.tensor([0.1, 0.2, float("nan"), 0.4, 0.5, 0.6])
    trained = train_readout(x, y, device=torch.device("cpu"), steps=2)
    pred = predict_readout(trained, x, torch.device("cpu"))
    assert torch.isfinite(pred).all()


def test_no_student_checkpoint_mutation_accepts_identical_state_and_rejects_change():
    model = torch.nn.Linear(2, 1)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    _check_student_unchanged(model, state)
    with torch.no_grad():
        model.bias.add_(0.5)
    with pytest.raises(RuntimeError, match="Student checkpoint state changed"):
        _check_student_unchanged(model, state)


def _tiny_segment():
    from dmdod.adofai_chart import parse_adofai_text
    from dmdod.adofai_timing import compile_adofai
    from dmdod.adofai_playable import build_playable_segment
    chart = parse_adofai_text(
        '{"angleData":[0,90,0,0],"settings":{"bpm":120,'
        '"pitch":100,"countdownTicks":0,"separateCountdownTime":false},'
        '"actions":[]}'
    )
    compiled = compile_adofai(chart)
    return build_playable_segment(compiled, start_s=0.0, end_s=compiled.duration_s)


class TinyStudent(torch.nn.Module):
    key_count = 8
    input_dim = 263

    def __init__(self):
        super().__init__()
        self.extra = torch.nn.Parameter(torch.tensor([0.02]))

    def initial_state(self, device):
        return torch.zeros(6, device=device)

    def forward_step(self, x, state):
        assert x.shape == (263,)
        return torch.zeros(8, device=x.device) + self.extra * 0.0, None, None, state + 1.0


def _rollout():
    named = SimpleNamespace(chart_name="test_chart", segment=_tiny_segment())
    prepared = SimpleNamespace(
        device=torch.device("cpu"), key_count=8, input_dim=263,
        control_dt_s=0.01, physics_dt_s=0.001,
    )
    model = TinyStudent()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rollout = collect_student_rollout(model, prepared, named, role="validation", index=1)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    return rollout


def test_rollout_observations_rnn_actions_and_privileged_labels_separate():
    row = _rollout()
    assert row.observations.shape[1] == 263
    assert row.hidden_states.shape[1] == 6
    assert row.actions.shape[1] == 8
    assert row.target_indices[0] == 0
    assert row.remaining_s[0] == pytest.approx(
        _tiny_segment().targets[0].episode_time_s
    )
    assert row.frame_times_s[0] == 0.0
    assert row.trace.termination.reason in ("Time limit", "All targets resolved")
    assert row.trace.termination.keydowns == row.stats.hits + row.stats.too_early_presses
    assert torch.all(row.hidden_states[1:, 0] > row.hidden_states[:-1, 0])


def test_frame_jsonl_retains_full_visible_and_hidden_without_leakage(tmp_path):
    rollout = _rollout()
    path = tmp_path / "frames.jsonl.gz"
    save_frame_data(rollout, path)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        first = json.loads(next(f))
    assert len(first["visible_263d"]) == 263
    assert len(first["student_rnn_state_after_step"]) == 6
    assert len(first["continuous_action"]) == 8
    assert first["next_target_index"] == 0
    assert first["target_remaining_s"] == pytest.approx(0.5)


def test_motor_metrics_preserve_episode_score_and_zero_keydown_case():
    rollout = _rollout()
    stats = motor_statistics(rollout)
    assert stats["hits"] == rollout.stats.hits
    assert stats["too_early"] == rollout.stats.too_early_presses
    assert stats["keydowns"] == rollout.trace.termination.keydowns
    assert stats["press_command_to_down"]["n"] == 0
    assert stats["keydown_target_error"]["n"] == 0


def test_prehit_metric_is_separate_from_all_frames():
    rollout = _rollout()
    states = rollout.hidden_states
    obs = rollout.observations
    label = rollout.remaining_s
    hidden = train_readout(states, label, device=torch.device("cpu"), steps=4)
    visible = train_readout(obs, label, device=torch.device("cpu"), steps=4)
    metrics, hp, vp = evaluate_time_readouts(
        rollout, hidden, visible, torch.device("cpu")
    )
    assert hp.shape == vp.shape == label.shape
    assert metrics["valid_frames"] > 0
    assert metrics["prehit_frames"] > 0
    assert metrics["hidden"]["prehit_0_150ms"]["n"] == metrics["prehit_frames"]
    assert metrics["visible_only"]["all"]["n"] == metrics["valid_frames"]


def test_validation_metric_aggregation_is_frame_weighted():
    def r(n, mae):
        return {"timing": {"hidden": {
            "all": {"n": n, "mae_ms": mae},
            "prehit_0_150ms": {"n": 0, "mae_ms": None}
        }}}
    aggregate = _aggregate_readout_stats([r(10, 20.0), r(30, 40.0)], "hidden")
    assert aggregate["all"]["mae_ms"] == pytest.approx(35.0)
    assert aggregate["prehit_0_150ms"]["mae_ms"] is None


def test_no_student_checkpoint_mutation_with_sparse_connectome_coo_and_csr():
    # torch.equal() on SparseCPU tensors is unsupported in PyTorch.
    class SparseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("coo", torch.sparse_coo_tensor(
                torch.tensor([[0, 1], [1, 0]]),
                torch.tensor([1.5, -2.0]),
                (2, 2),
            ).coalesce())
            self.register_buffer("csr", self.coo.to_sparse_csr())
            self.register_buffer("dense", torch.tensor([1.0, 2.0]))

    model = SparseModel()
    snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    _check_student_unchanged(model, snapshot)

    with torch.no_grad():
        model.csr.values()[0].add_(0.5)
    with pytest.raises(RuntimeError, match="Student checkpoint state changed"):
        _check_student_unchanged(model, snapshot)
    model.csr = snapshot["csr"].clone()
    _check_student_unchanged(model, snapshot)

    with torch.no_grad():
        # COO values() is mutable when the tensor has been coalesced.
        model.coo._values()[0].add_(0.5)
    with pytest.raises(RuntimeError, match="Student checkpoint state changed"):
        _check_student_unchanged(model, snapshot)
    model.coo = snapshot["coo"].clone()
    _check_student_unchanged(model, snapshot)


def test_sparse_state_comparison_detects_layout_and_index_changes():
    from dmdod.training.timing_motor_probe import _unchanged_tensor

    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [0, 1]]), torch.tensor([3.0, 4.0]), (2, 2)
    ).coalesce()
    assert _unchanged_tensor(sparse, sparse.clone())
    assert not _unchanged_tensor(sparse, sparse.to_dense())
    reordered = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [1, 0]]), torch.tensor([3.0, 4.0]), (2, 2)
    ).coalesce()
    assert not _unchanged_tensor(sparse, reordered)
    assert not _unchanged_tensor(sparse, torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [0, 1]]), torch.tensor([3.0, 5.0]), (2, 2)
    ).coalesce())
