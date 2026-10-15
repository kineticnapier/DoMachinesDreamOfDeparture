from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dmdod.training import prestige1 as frozen
from dmdod.training import prestige1_autoplay_readout as timing


def seq(anchor: int = 1, *, key: str = "a", length: int = 6):
    visible = torch.zeros(length, frozen.INPUT_DIM, dtype=torch.float32)
    visible[:, 31] = torch.linspace(0, 1, length)
    # Exact fraction-of-10ms values; next event remains >200ms on some frames.
    remaining = torch.tensor(
        [0.7012345, 0.3612345, 0.1912345, 0.0812345, 0.0062345, 0.],
        dtype=torch.float32,
    )[:length]
    near = remaining <= .2
    mask = torch.tensor([True] * max(length-1, 0) + [False])
    return timing.TimingSequence(
        anchor=anchor, chart_sha256=key * 64, visible=visible,
        remaining_s=remaining, near=near, mask=mask,
        event_indices=tuple(range(length)),
    )


def test_autoplay_truth_is_not_clipped_to_200ms():
    source = seq()
    assert source.remaining_s[0].item() > .7
    assert source.mask[0]
    assert not source.near[0]
    model = timing.TimingReadout(12, recurrent=False)
    raw = model(source.visible)
    predicted_s, near_probability = timing.decode(raw)
    assert predicted_s.shape == (6,)
    assert near_probability.shape == (6,)
    assert bool((predicted_s > 0.).all())
    loss = timing.timing_loss(raw, source, 0, 6)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_only_visible_features_can_reach_model():
    source = seq()
    model = timing.TimingReadout(12, recurrent=True)
    # The model accepts [T,263], never extra oracle timestamp/target data.
    with pytest.raises(ValueError, match="263"):
        model(torch.zeros(6, 264))
    first = model(source.visible)
    changed_labels = timing.TimingSequence(
        source.anchor, source.chart_sha256, source.visible,
        torch.ones_like(source.remaining_s) * .9,
        torch.zeros_like(source.near),
        source.mask, tuple([123] * 6),
    )
    assert torch.equal(first, model(changed_labels.visible))
    assert first.shape == (6, 2)


@pytest.mark.parametrize("recurrent", [False, True])
def test_forward_predictions_are_causal_prefix_invariant(recurrent):
    torch.manual_seed(88)
    model = timing.TimingReadout(12, recurrent=recurrent).eval()
    visible = torch.randn(8, 263)
    with torch.no_grad():
        earlier = model(visible[:5])
        full = model(visible)
    assert torch.allclose(earlier, full[:5], atol=1e-6, rtol=1e-6)


def test_constant_and_models_share_identical_eval_frames():
    source = seq()
    reference = {"time_s": .1, "near_probability": .4}
    baseline = timing.evaluate(None, [source], reference)
    result = timing.evaluate(timing.TimingReadout(12, recurrent=False),
                             [source], reference)
    for score in (baseline, result):
        metrics = score["aggregate"]
        assert metrics["frames_observable"] == 5
        assert metrics["frames_masked"] == 1
        assert metrics["near_positive_frames"] == 3
        assert metrics["near_negative_frames"] == 2
        assert metrics["positive_anchors"] == 1
        assert metrics["near_p95_abs_ms"] >= metrics["near_p50_abs_ms"]
    assert baseline["aggregate"]["near_mae_ms"] > 0


def test_fit_labels_only_determine_constant_and_metrics():
    source = seq()
    reference = timing.constant_reference([source])
    assert reference["fit_near_positive_count"] == 3
    assert reference["fit_valid_frames"] == 5
    assert 0. < reference["time_s"] < .2
    assert reference["near_probability"] == pytest.approx(.6)
    chunks = timing._chunks([source], 2)
    assert len(chunks) == 3
    assert all(bool(s.mask[a:b].any()) for s, a, b in chunks)
    model = timing.TimingReadout(8, recurrent=True)
    curve, count = timing.train_one(
        model, [source], updates=2, frames=2, lr=.0003, seed=2401)
    assert count == 3 and len(curve) == 3  # at least one full chunk pass
    assert all(torch.isfinite(torch.tensor(curve)))


def test_existing_stage_a_gate_cannot_be_unlocked():
    fake_metrics = lambda mae, bal=.8, fpr=.01: {
        "aggregate": {
            "near_mae_ms": mae, "positive_anchors": 5,
            "balanced_accuracy": bal, "false_positive_rate": fpr,
        }
    }
    gate = timing.exploratory_gate({
        "constant": fake_metrics(80),
        "mlp": fake_metrics(65),
        "gru": fake_metrics(30),
    })
    assert gate["observability_evidence_passed"]
    assert gate["motor_student_training_permitted"] is False
    assert gate["original_stage_a_gate_overridden"] is False
    assert frozen.STAGE_A_REPORT == "stage_a_v2.json"
    assert timing.REPORT != frozen.STAGE_A_REPORT
    assert timing.MLP_CHECKPOINT != "single_v2.pt"


def test_gate_fails_when_gru_does_not_improve_near_mae():
    example = lambda mae: {
        "aggregate": {
            "near_mae_ms": mae, "positive_anchors": 5,
            "balanced_accuracy": .6, "false_positive_rate": .01,
        }
    }
    gate = timing.exploratory_gate({
        "constant": example(80), "mlp": example(50), "gru": example(60),
    })
    assert not gate["observability_evidence_passed"]
    assert "gru_not_20pct_better_than_constant_and_mlp" in gate["reasons"]


@pytest.mark.parametrize("tamper", [
    lambda s: s.visible.__setitem__((0, 0), float("nan")),
    lambda s: s.remaining_s.__setitem__((0,), -1.),
])
def test_nonfinite_or_negative_training_inputs_fail_closed(tamper):
    source = seq()
    tamper(source)
    with pytest.raises(ValueError, match="Invalid causal Timing"):
        timing.TimingSequence(
            source.anchor, source.chart_sha256, source.visible,
            source.remaining_s, source.near, source.mask,
            source.event_indices,
        )


def test_chart_disjoint_split_remains_frozen_train_only():
    sequences = [
        timing.TimingSequence(
            anchor=i, chart_sha256=f"{(i-1)//2:064x}",
            visible=seq().visible, remaining_s=seq().remaining_s,
            near=seq().near, mask=seq().mask, event_indices=seq().event_indices)
        for i in range(1, 21)
    ]
    manifest = {
        "train": [{"anchor":i, "sha256": sequences[i-1].chart_sha256}
                  for i in range(1, 21)]
    }
    fit, heldout, metadata = frozen.split_stage_a_train(
        sequences, manifest, seed=2401,
    )
    assert set(s.chart_sha256 for s in fit).isdisjoint(
        s.chart_sha256 for s in heldout
    )
    assert len(fit) + len(heldout) == 20
    assert metadata["validation_used_for_selection"] is False
