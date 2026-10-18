from __future__ import annotations

import ast
import inspect
import json

import pytest

from dmdod.adofai.chart import parse_adofai_text
from scripts import audit_bool_second_event_counterfactual_v01 as audit


def synthetic_chart(*, action=True):
    raw = {
        "angleData": [0, 90, 180, 270, 0, 90, 180],
        "settings": {"bpm": 120.0, "countdownTicks": 3, "pitch": 100.0},
        "actions": ([{
            "floor": 1,
            "eventType": "SetSpeed",
            "speedType": "Bpm",
            "beatsPerMinute": 480.0,
        }] if action else []),
    }
    return parse_adofai_text(json.dumps(raw))


def test_intervention_preserves_marker_and_edits_only_floor1_numeric():
    chart = synthetic_chart()
    other, meta = audit.intervene_floor1_setspeed(chart)
    assert len(other.actions) == len(chart.actions)
    assert other.actions[0].floor == 1
    assert other.actions[0].event_type == "SetSpeed"
    assert other.actions[0].beats_per_minute == pytest.approx(60.0)
    assert chart.actions[0].beats_per_minute == pytest.approx(480.0)
    assert meta["modified_field"] == "beatsPerMinute"
    assert chart.angles == other.angles
    assert chart.countdown_ticks == other.countdown_ticks


def test_missing_floor1_setspeed_is_not_claimed_proven():
    with pytest.raises(ValueError, match="NO_FLOOR1_SETSPEED"):
        audit.intervene_floor1_setspeed(synthetic_chart(action=False))


def test_synthetic_predeadline_hud278_indistinguishable():
    original, alternate, intervention = audit.compile_pair(synthetic_chart())
    report = audit.compare_pair(original, alternate, intervention)
    assert report["first_due_unchanged"]
    assert report["first_three_floors_visible_topology_equal"]
    assert report["second_due_gap_ms_posthoc"] > 44.0
    assert report["indistinguishable_prefix"]
    assert report["first_distinguishable_frame_in_prefix"] is None
    assert report["status"] == "INDISTINGUISHABLE_BEFORE_44MS_DEADLINE"


def test_capture_function_does_not_read_target_truth_or_setspeed_values():
    tree = ast.parse(inspect.getsource(audit.capture_causal_prefix))
    attributes = {
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    assert attributes.isdisjoint({
        "targets", "episode_time_s", "target_time_s", "actions", "bpm_multiplier",
        "beats_per_minute", "_used", "_missed", "judgement_trace",
    })
