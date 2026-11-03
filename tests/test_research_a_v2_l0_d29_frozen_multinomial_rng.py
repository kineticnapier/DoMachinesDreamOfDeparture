"""D29 preregistered frozen RNG replay tests (no learning, Motor, or inference)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import a002v2_l0_d29_frozen_multinomial_rng as d29


def synthetic_steps(probabilities=None, count=480):
    p = probabilities or [0.35, 0.25, 0.25, 0.15]
    rng = torch.Generator(device="cpu").manual_seed(d29.SEED)
    steps = []
    for i in range(count):
        action = int(torch.multinomial(torch.tensor(p, dtype=torch.float32),
                                      1, generator=rng).item())
        steps.append({"step": i, "sampled_action": action,
                      "action": action, "action_probabilities": list(p)})
    return steps


def test_replay_matches_all_archived_draws():
    r = d29.replay_sequence(synthetic_steps())
    assert r["matched_steps"] == 480
    assert len(r["pivot_state_sha256"]) == 64


def test_identical_input_has_identical_rng_state():
    steps = synthetic_steps()
    assert torch.equal(d29.replay_sequence(steps)["pivot_state"],
                       d29.replay_sequence(copy.deepcopy(steps))["pivot_state"])


def test_pivot_original_draw_is_reproducible():
    steps = synthetic_steps()
    state = d29.replay_sequence(steps)["pivot_state"]
    p = d29.tensor_probs(steps[d29.STEP])
    assert d29.draw_at_state(p, state) == steps[d29.STEP]["sampled_action"]


def test_bad_archived_draw_is_rejected():
    steps = synthetic_steps()
    steps[12]["sampled_action"] = (steps[12]["sampled_action"] + 1) % 4
    with pytest.raises(ValueError, match="sample mismatch at step 12"):
        d29.replay_sequence(steps)


def test_invalid_probability_is_rejected():
    steps = synthetic_steps()
    steps[0]["action_probabilities"] = [0.5, 0.4, -0.1, 0.2]
    with pytest.raises(ValueError):
        d29.replay_sequence(steps)


def test_invalid_step_number_is_rejected():
    steps = synthetic_steps()
    steps[12]["step"] = 11
    with pytest.raises(ValueError, match="step index mismatch"):
        d29.replay_sequence(steps)


def test_missing_pivot_is_rejected():
    with pytest.raises(ValueError, match="missing pivot"):
        d29.replay_sequence(synthetic_steps(count=413))


def test_frozen_cross_draws_are_deterministic():
    steps = synthetic_steps()
    state = d29.replay_sequence(steps)["pivot_state"]
    p = torch.tensor([0.2, 0.3, 0.1, 0.4], dtype=torch.float32)
    assert d29.draw_at_state(p, state) == d29.draw_at_state(p, state)


def test_full_synthetic_analyze_without_physics(monkeypatch):
    monkeypatch.setattr(d29.d28, "analyze", lambda _source, _traces: {})
    before = synthetic_steps()
    after = copy.deepcopy(before)
    traces = {"before": {"steps": before}, "after_to_none": {"steps": after}}
    r = d29.analyze({}, traces)
    assert r["classification"] == "valid_frozen_multinomial_replay"
    assert r["pivot_rng_state_exact"] is True
    assert r["replay"]["before"]["matched_steps"] == 480
    assert r["pivot_cross_draws"]["before"]["before"] == before[413]["sampled_action"]
    assert r["new_physical_keydowns"] == r["new_policy_inferences"] == 0
