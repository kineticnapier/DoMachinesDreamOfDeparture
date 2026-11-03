"""D28 archived policy/reward audit checks (no training or Motor)."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import a002v2_l0_d28_step413_policy_reward as d28


def make_step(i, action=0, *, p=None):
    return {
        "step": i, "time_s": i / 100,
        "observation": [0.0] * 8,
        "action_probabilities": p or [0.35, 0.25, 0.25, 0.15],
        "action": action, "sampled_action": action,
        "intervention_applied": False,
        "pre_motor": {"left_position_m": 0.0},
        "pre_diagnostics": {"left_fatigue": 0.0},
        "reward": 0.0,
    }


def event(t, i, hit=False, reward=0.0):
    return {
        "source_step": i,
        "event": {
            "time_s": t, "key": "left", "hit": hit,
            "closest_target_ordinal": 12,
            "offset_ms": 12.0, "immediate_reward": reward,
        },
    }


def make_traces():
    b = {
        "steps": [make_step(i) for i in range(480)],
        "physical_keydowns": [event(3.718, 371, True, 0.9775),
                              event(4.019, 401)],
        "row": {"hits": 1, "extra_keydowns": 1, "physical_keydowns": 2},
    }
    a = copy.deepcopy(b)
    a["steps"][413] = make_step(413, 2, p=[0.30, 0.26, 0.28, 0.16])
    a["physical_keydowns"].append(event(4.327, 432))
    a["row"] = {"hits": 1, "extra_keydowns": 2, "physical_keydowns": 3}
    return {
        "before": b, "after_to_none": a,
        "before_negative": copy.deepcopy(b),
        "after_negative": copy.deepcopy(a),
        "after": copy.deepcopy(a),
        "before_to_left": copy.deepcopy(b),
    }


def run_fixture(monkeypatch, traces=None):
    monkeypatch.setattr(d28.d23, "validate_sources", lambda _s, _t: None)
    return d28.analyze({}, traces or make_traces())


def test_probability_accepts_normalized():
    assert d28.probabilities(make_step(413)) == [.35, .25, .25, .15]


@pytest.mark.parametrize("bad", [
    [0.3, 0.3, 0.3, 0.3],
    [0.5, -0.1, 0.4, 0.2],
    [math.nan, 0.1, 0.4, 0.5],
    [1.0, 0.0, 0.0],
])
def test_probability_rejects_bad(bad):
    with pytest.raises(ValueError):
        d28.probabilities(make_step(413, p=bad))


def test_snapshot_rejects_overridden_step():
    s = make_step(413)
    s["intervention_applied"] = True
    with pytest.raises(ValueError):
        d28.snapshot(s)


def test_event_extraction_and_absence():
    t = make_traces()
    assert d28.event_at_extra(t["before"]) is None
    assert d28.event_at_extra(t["after_to_none"])["source_step"] == 432


def test_event_rejects_hit():
    t = make_traces()["after_to_none"]
    t["physical_keydowns"][-1]["event"]["hit"] = True
    with pytest.raises(ValueError):
        d28.event_at_extra(t)


def test_full_audit_with_matching_observation(monkeypatch):
    result = run_fixture(monkeypatch)
    assert result["classification"] == "valid_step413_archived_policy_reward_audit"
    assert result["step413_comparison"]["observation_exact"] is True
    assert result["step413_comparison"]["probability_l1"] == pytest.approx(.10)
    assert result["extra_at_4_327"]["after_to_none"]["immediate_reward"] == 0
    assert result["new_motor_replays"] == result["new_policy_inferences"] == 0


def test_observation_difference_is_reported_not_rejected(monkeypatch):
    t = make_traces()
    t["after_to_none"]["steps"][413]["observation"][2] = .125
    result = run_fixture(monkeypatch, t)
    assert not result["step413_comparison"]["observation_exact"]
    assert result["step413_comparison"]["observation_max_abs_difference"] == .125


def test_negative_control_mismatch_rejected(monkeypatch):
    t = make_traces()
    t["before_negative"]["steps"][10]["reward"] = 1
    with pytest.raises(ValueError):
        run_fixture(monkeypatch, t)


def test_physical_prefix_difference_rejected(monkeypatch):
    t = make_traces()
    t["after_to_none"]["steps"][300]["action"] = 2
    with pytest.raises(ValueError):
        run_fixture(monkeypatch, t)


def test_wrong_step413_source_rejected(monkeypatch):
    t = make_traces()
    t["after_to_none"]["steps"][413]["action"] = 0
    t["after_to_none"]["steps"][413]["sampled_action"] = 0
    with pytest.raises(ValueError):
        run_fixture(monkeypatch, t)
