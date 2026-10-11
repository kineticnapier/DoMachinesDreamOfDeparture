from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dmdod.adofai_chart import parse_adofai_text
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import compile_adofai
from dmdod.training.beta_training_ablation import (
    ARMS, _assert_beta_reference, preflight_outputs,
)
from dmdod.training.config import load_training_config
from dmdod.training.dagger_continuation import (
    collect_n_key_intervention_dagger_sequence_with_continuation,
)


class _AlwaysPressPolicy(torch.nn.Module):
    key_count = 4

    def initial_state(self, device):
        return torch.zeros(1, device=device)

    def forward_step(self, x, state):
        mean = torch.full((4,), 6.0, dtype=x.dtype, device=x.device)
        return mean, None, None, state


def _segment():
    chart = parse_adofai_text(
        '{"angleData":[0,90,180,270,0,90,180,270,0,90,180,270,0,90,180,270],'
        '"settings":{"bpm":120,"pitch":100,"countdownTicks":0,'
        '"separateCountdownTime":false},"actions":[]}'
    )
    compiled = compile_adofai(chart)
    return build_playable_segment(compiled, start_s=0.0, end_s=compiled.duration_s)


def _collect(limit):
    return collect_n_key_intervention_dagger_sequence_with_continuation(
        _AlwaysPressPolicy(), _segment(), lead_s=0.044,
        beta=0.0, seed=2401, press_threshold=0.25,
        release_threshold=-0.45,
        control_dt_s=0.010, physics_dt_s=0.001,
        device=torch.device("cpu"),
        action_mode="continuous", continue_after_failure=True,
        max_post_failure_s=limit,
    )


def test_post_failure_cap_retains_exact_first_terminal_and_prefix():
    full = _collect(None)
    capped = _collect(1.0)
    assert full.stats.overloaded
    assert capped.stats.overloaded
    assert full.stats.hits == capped.stats.hits
    assert full.stats.too_early_presses == capped.stats.too_early_presses
    assert full.physical_keydowns == capped.physical_keydowns
    assert full.terminal_frame_count == capped.terminal_frame_count
    assert full.terminal_control_time_s == capped.terminal_control_time_s
    assert capped.sequence.frames == capped.terminal_frame_count + 100
    assert full.sequence.frames >= capped.sequence.frames
    assert torch.equal(
        capped.sequence.observations,
        full.sequence.observations[:capped.sequence.frames],
    )
    assert torch.equal(
        capped.sequence.teacher_actions,
        full.sequence.teacher_actions[:capped.sequence.frames],
    )
    assert torch.equal(
        capped.executed_actions,
        full.executed_actions[:capped.sequence.frames],
    )


def test_zero_post_failure_cap_disables_continuation():
    capped = _collect(0.0)
    assert capped.stats.overloaded
    assert capped.sequence.frames == capped.terminal_frame_count


@pytest.mark.parametrize("limit", [-1, float("nan"), float("inf")])
def test_invalid_post_failure_cap_rejected(limit):
    with pytest.raises(ValueError, match="max_post_failure_s"):
        _collect(limit)


def test_arm_definition_and_beta1_reproducibility_guard():
    assert [x.name for x in ARMS] == [
        "beta1_full", "beta05_full", "beta0_full", "beta0_post1s",
    ]
    assert [x.beta for x in ARMS] == [1.0, 0.5, 0.0, 0.0]
    assert ARMS[-1].max_post_failure_s == 1.0
    expected = (9, 1546, 250, 1796)
    actual = dict(zip(("safe", "hits", "early", "keydowns"), expected))
    _assert_beta_reference(actual, expected)
    with pytest.raises(RuntimeError, match="reference candidate mismatch"):
        _assert_beta_reference({**actual, "hits": 1545}, expected)


def test_preflight_never_overwrites_existing_outputs(tmp_path):
    report = tmp_path / "beta.json"
    source = tmp_path / "source.pt"
    paths = preflight_outputs(report, save_candidates=True, source_checkpoint=source)
    assert len(paths) == 4
    assert all(not path.exists() for path in paths.values())
    report.write_text("{}")
    with pytest.raises(FileExistsError):
        preflight_outputs(report, save_candidates=False, source_checkpoint=source)
    report.unlink()
    next(iter(paths.values())).write_text("occupied")
    with pytest.raises(FileExistsError):
        preflight_outputs(report, save_candidates=True, source_checkpoint=source)


def test_ablation_config_has_fixed_20_validation_anchors_and_no_alpha_sweep():
    cfg = load_training_config(
        "configs/training/human_visible_beta_training_ablation.toml"
    )
    assert cfg.data.anchor_limit == 20
    assert cfg.data.validation_limit == 20
    assert cfg.human_visible.updates_per_epoch == 2
    assert not cfg.human_visible.alpha_sweep_enabled
    assert "UNUSED_beta_training_ablation" in cfg.run.output


def test_four_arms_restore_initial_model_and_optimizer_without_promotion(monkeypatch, tmp_path):
    import dmdod.training.beta_training_ablation as ablation
    from dmdod.training.human_visible_curriculum import CurriculumEpochMetrics

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(3.0))

        def prepare_recurrent_runtime(self):
            pass

    model = DummyModel()
    config = SimpleNamespace(
        human_visible=SimpleNamespace(
            seed=2401, burn_in_steps=0, supervised_steps=1,
            anchors_per_batch=1, grad_clip=1.0,
        ),
    )
    prepared = SimpleNamespace(
        model=model,
        parent={"human_visible_best_validation_summary": {"safe": 12, "hits": 1786}},
        source_checkpoint=tmp_path / "selected-best.pt",
        validation=[object()] * 20,
        anchors=[object()] * 20,
        control_dt_s=0.01,
        physics_dt_s=0.001,
        device=torch.device("cpu"),
    )
    starts = []
    schedules = []
    eval_calls = []

    class DummyOptimizer:
        def __init__(self):
            self.marker = -1

    def checked_load(optimizer, parent, model):
        optimizer.marker = 0

    def collect_variant(prepared, model, arm, *, epoch, seed):
        starts.append((arm.name, float(model.weight.detach().item())))
        return [SimpleNamespace(frames=1)], ()

    def train_epoch(model, windows, plan, *, optimizer, **kwargs):
        assert optimizer.marker == 0
        assert float(model.weight.detach().item()) == 3.0
        optimizer.marker = 2
        schedules.append(plan)
        with torch.no_grad():
            model.weight.add_(1.0)
        return CurriculumEpochMetrics(2, 16, 0.1, 0.09, 1.0, 1.0, {})

    def eval_role(model, *args, **kwargs):
        eval_calls.append(float(model.weight.detach().item()))
        return [object()]

    def summary(results):
        if len(eval_calls) == 1:
            return {"safe": 12, "anchors": 20, "hits": 1786, "targets": 4682,
                    "early": 278, "keydowns": 2064, "x": 49.62}
        return {"safe": 9, "anchors": 20, "hits": 1546, "targets": 4682,
                "early": 250, "keydowns": 1796, "x": 49.62}

    monkeypatch.setattr(ablation, "_validate_experiment", lambda *a, **kw: 3)
    monkeypatch.setattr(ablation, "_collect_expert_trajectories",
                        lambda p: [SimpleNamespace(frames=4)])
    monkeypatch.setattr(ablation, "_collect_variant", collect_variant)
    monkeypatch.setattr(ablation, "format_beta_summary", lambda rows: ())
    monkeypatch.setattr(ablation, "build_human_visible_optimizer",
                        lambda *a: DummyOptimizer())
    monkeypatch.setattr(ablation, "_load_checked_optimizer", checked_load)
    monkeypatch.setattr(ablation, "build_replay_windows", lambda *a, **kw: [0])
    monkeypatch.setattr(ablation, "build_balanced_epoch_plan",
                        lambda *a, **kw: (("a",), ("b",)))
    monkeypatch.setattr(ablation, "train_human_visible_epoch", train_epoch)
    monkeypatch.setattr(ablation, "evaluate_role_continuous", eval_role)
    monkeypatch.setattr(ablation, "validation_summary", summary)
    monkeypatch.setattr(ablation, "_check_validation_summary",
                        lambda actual, expected: None)

    report_path = tmp_path / "ablation.json"
    result = ablation.run_beta_training_ablation(
        prepared, config, report_path=report_path, save_candidates=False,
    )
    assert [n for n, _ in starts] == [a.name for a in ARMS]
    assert all(w == 3.0 for _, w in starts)
    assert len(schedules) == 4 and all(p == schedules[0] for p in schedules)
    assert eval_calls == [3.0, 4.0, 4.0, 4.0, 4.0]
    assert report_path.is_file()
    assert result["baseline_validation"]["safe"] == 12
    assert len(result["arms"]) == 4
    assert all(a["training"]["updates"] == 2 for a in result["arms"])
    assert all(not a["selected_for_checkpoint"] for a in result["arms"])
    assert not list(tmp_path.glob("*.pt"))
