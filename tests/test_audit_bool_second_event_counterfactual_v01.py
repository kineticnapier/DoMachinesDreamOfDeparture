from __future__ import annotations

import ast
import inspect
import json

import pytest

from dmdod.adofai.chart import parse_adofai_text
from scripts import audit_bool_second_event_counterfactual_v01 as audit


def synthetic_chart(*, action=True, second_angle=150.0):
    raw = {
        "angleData": [0, second_angle, 180, 270, 0, 90, 180],
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
    # Floor 1 is a 30-degree rotation, i.e. 20.833ms at 480 BPM.
    # A 44ms preparation deadline is strictly BEFORE the first event,
    # so the new BPM cannot become visible on the current floor yet.
    assert (report["deadline_44ms_s_posthoc"]
            < report["first_due_original_s_posthoc"])
    assert report["first_three_floors_visible_topology_equal"]
    assert report["second_due_gap_ms_posthoc"] > 44.0
    assert report["indistinguishable_prefix"]
    assert report["first_distinguishable_frame_in_prefix"] is None
    assert report["status"] == "INDISTINGUISHABLE_BEFORE_44MS_DEADLINE"


def test_wide_arc_negative_control_becomes_distinguishable_before_deadline():
    # 90-degree Floor 1 takes 62.5ms at 480 BPM. The 44ms deadline
    # is AFTER the first target, so the current-floor Mod BPM reveals
    # the changed SetSpeed value. The audit must NOT claim aliasing.
    original, alternate, intervention = audit.compile_pair(
        synthetic_chart(second_angle=90.0)
    )
    report = audit.compare_pair(original, alternate, intervention)
    assert (report["deadline_44ms_s_posthoc"]
            > report["first_due_original_s_posthoc"])
    assert report["second_due_gap_ms_posthoc"] > 44.0
    assert not report["indistinguishable_prefix"]
    assert report["first_distinguishable_frame_in_prefix"] is not None
    assert report["status"] == "NOT_PROVEN"


def test_short_arc_is_aligned_with_bool_second_event_spacing():
    chart = synthetic_chart()
    original, alternate, _ = audit.compile_pair(chart)
    gap = (original.targets[1].episode_time_s
           - original.targets[0].episode_time_s)
    assert gap * 1000.0 == pytest.approx(20.833333, abs=.01)
    assert (alternate.targets[1].episode_time_s
            - original.targets[1].episode_time_s) > .044



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
