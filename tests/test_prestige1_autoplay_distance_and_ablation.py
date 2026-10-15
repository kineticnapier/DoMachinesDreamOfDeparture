from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dmdod.training import prestige1 as p
from dmdod.training import prestige1_autoplay_diagnostics as diag
from dmdod.training import prestige1_autoplay_near_ablation as ablation
from dmdod.training import prestige1_autoplay_readout as t


def synthetic_seq(*, length=6, anchor=1):
    observed = torch.zeros(length, 263)
    observed[:, 32] = torch.arange(length, dtype=torch.float32) / 100.
    # 0..50, 50..100, 100..200, >200 (including exact boundary cases).
    times = torch.tensor([0.0, 0.05, 0.099, 0.10, 0.20, 0.41])[:length]
    near = torch.tensor([True, True, True, True, True, False])[:length]
    mask = torch.ones(length, dtype=torch.bool)
    return t.TimingSequence(
        anchor=anchor, chart_sha256=f"{anchor:064x}",
        visible=observed, remaining_s=times, near=near, mask=mask,
        event_indices=tuple(range(length)),
    )


def test_distance_bins_exact_partition_and_boundaries():
    source = synthetic_seq()
    bins = diag._bin_masks(source)
    assert list(bins) == list(diag.BINS)
    assert [int(bins[name].sum()) for name in diag.BINS] == [1, 2, 2, 1]
    assert sum(int(mask.sum()) for mask in bins.values()) == 6
    audit = diag.distance_metrics(None, [source], {
        "time_s": .07, "near_probability": .5})
    assert audit["aggregate"]["0-50ms"]["mae_ms"] == pytest.approx(70., abs=1e-4)
    assert audit["aggregate"]["over-200ms"]["mae_ms"] == pytest.approx(340., abs=1e-4)


def test_masked_frames_are_never_scored():
    source = synthetic_seq()
    mask = source.mask.clone()
    mask[5] = False
    masked = t.TimingSequence(
        source.anchor, source.chart_sha256, source.visible, source.remaining_s,
        source.near, mask, source.event_indices,
    )
    bins = diag._bin_masks(masked)
    assert int(bins[diag.BINS[-1]].sum()) == 0
    metrics = diag.distance_metrics(None, [masked], {
        "time_s": .05, "near_probability": .5})["aggregate"]
    assert metrics[diag.BINS[-1]]["frames"] == 0
    assert metrics[diag.BINS[-1]]["mae_ms"] is None


def test_frozen_stage_t_replay_is_no_training_contract():
    assert diag.FORMAT != t.FORMAT
    assert ablation.FORMAT != t.FORMAT
    assert ablation.REPORT != t.REPORT
    assert ablation.CHECKPOINT not in (t.MLP_CHECKPOINT, t.GRU_CHECKPOINT)
    assert p.STAGE_A_REPORT == "stage_a_v2.json"


def test_far_truth_has_zero_gradient_on_time_regression_but_not_classification():
    source = synthetic_seq()
    raw = torch.tensor([
        [0.1, -0.2], [0.2, 0.1], [0.3, -0.3],
        [0.4, -0.4], [0.5, 0.2], [0.6, 0.3],
    ], requires_grad=True)
    loss = ablation.near_only_timing_loss(raw, source, 0, 6)
    loss.backward()
    assert raw.grad is not None
    assert raw.grad[5, 0].item() == 0.
    assert raw.grad[5, 1].item() != 0.
    assert raw.grad[0, 0].item() != 0.


def test_original_stage_t_loss_still_uses_far_regression():
    source = synthetic_seq()
    near = torch.zeros(6, 2, requires_grad=True)
    old = t.timing_loss(near, source, 0, 6)
    old.backward()
    assert near.grad[5, 0].item() != 0.  # single changed variable


def test_classification_still_learns_from_far_only_chunks():
    full = synthetic_seq()
    mask = torch.zeros_like(full.mask)
    mask[-1] = True
    far_only = t.TimingSequence(
        full.anchor, full.chart_sha256, full.visible, full.remaining_s,
        full.near, mask, full.event_indices)
    raw = torch.zeros(6, 2, requires_grad=True)
    loss = ablation.near_only_timing_loss(raw, far_only, 0, 6)
    assert torch.isfinite(loss)
    loss.backward()
    assert raw.grad[-1, 0].item() == 0.
    assert raw.grad[-1, 1].item() != 0.


def test_all_masked_chunk_has_zero_loss():
    source = synthetic_seq()
    mask = torch.zeros_like(source.mask)
    seq = t.TimingSequence(
        source.anchor, source.chart_sha256, source.visible, source.remaining_s,
        source.near, mask, source.event_indices)
    raw = torch.randn(6, 2, requires_grad=True)
    loss = ablation.near_only_timing_loss(raw, seq, 0, 6)
    assert loss.item() == 0.
    loss.backward()
    assert torch.all(raw.grad == 0.)


def test_fixed_training_schedule_same_seed_reproducible():
    source = synthetic_seq()
    torch.manual_seed(2401)
    a = t.TimingReadout(12, recurrent=True)
    torch.manual_seed(2401)
    b = t.TimingReadout(12, recurrent=True)
    log_a, chunks_a = ablation.train_near_only(
        a, [source], updates=3, frames=2, lr=3e-4, seed=2401)
    log_b, chunks_b = ablation.train_near_only(
        b, [source], updates=3, frames=2, lr=3e-4, seed=2401)
    assert chunks_a == chunks_b == 3
    assert log_a == pytest.approx(log_b, abs=1e-6)
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key])


def test_near_only_loss_cannot_unlock_stage_a_gate():
    examples = lambda mae: {
        "aggregate": {
            "near_mae_ms": mae,
            "positive_anchors": 5,
            "balanced_accuracy": .7,
            "false_positive_rate": .01,
        }
    }
    result = t.exploratory_gate({
        "constant": examples(40), "mlp": examples(50), "gru": examples(20),
    })
    assert result["observability_evidence_passed"] is True
    assert result["original_stage_a_gate_overridden"] is False
    assert result["motor_student_training_permitted"] is False
    assert result["automatically_adopted"] is False


def test_existing_output_is_never_overwritten(tmp_path):
    output = tmp_path / "distance.json"
    output.write_text("immutable")
    with pytest.raises(FileExistsError):
        diag.prepare(SimpleNamespace(), output_paths=(output,))
    assert output.read_text() == "immutable"


def test_report_distance_rollup_checks_counts_and_error():
    metrics = {
        "0-50ms": {"frames":1, "mae_ms":20.},
        "50-100ms": {"frames":2, "mae_ms":30.},
        "100-200ms": {"frames":3, "mae_ms":40.},
        "over-200ms": {"frames":4, "mae_ms":90.},
    }
    scores = {
        split: {
            name: {"aggregate": metrics}
            for name in ("constant", "mlp", "gru")
        }
        for split in ("train_fit", "train_heldout")
    }
    expected = {
        split: {
            name: {"aggregate": {
                "near_positive_frames": 6,
                "frames_observable": 10,
                "near_mae_ms": 200 / 6,
            }}
            for name in ("constant", "mlp", "gru")
        }
        for split in ("train_fit", "train_heldout")
    }
    fake_source = SimpleNamespace(
        fit=[], heldout=[], stage_report=expected,
    )
    diag._verify_distance_rollup(scores, fake_source)
    scores["train_heldout"]["gru"]["aggregate"]["100-200ms"]["mae_ms"] = 80.
    with pytest.raises(RuntimeError, match="Near MAE"):
        diag._verify_distance_rollup(scores, fake_source)
