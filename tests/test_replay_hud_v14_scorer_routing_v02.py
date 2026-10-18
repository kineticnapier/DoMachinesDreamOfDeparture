from __future__ import annotations

from copy import deepcopy

from scripts import replay_hud_v14_scorer_routing_v02 as forensic
from scripts.run_hud_v14_fingering_motor_integration_v02 import synthetic_segment


def prepared_shifted_down():
    segment = synthetic_segment(480.)
    first = float(segment.targets[0].episode_time_s)
    second = float(segment.targets[1].episode_time_s)
    assert abs((second - first) - .0625) < 1e-7
    # Physical/logical KeyDown exactly at target 1 due, but target 0 is
    # unresolved and gets the KeyDown within its normal 65ms counted radius.
    arm = {
        "frames": int((second + .12) / .010) + 1,
        "logical_switch_trace": [{
            "time_s": second, "key": "left_1", "edge": "down"
        }],
        "target_outcomes_posthoc": [
            {"ordinal": i, "due_s": float(t.episode_time_s),
             "hit": i == 0, "miss": i == 1}
            for i, t in enumerate(segment.targets)
        ],
        "stats": {"hits": 1, "misses": 1, "too_early": 0},
    }
    return segment, arm


def test_perfectly_timed_keydown_can_be_consumed_by_previous_unresolved_target():
    segment, arm = prepared_shifted_down()
    out = forensic.replay_scorer(segment, arm)
    assert out["score_replay_exact_match"] is True
    routed = out["logical_down_target_routing"]
    assert len(routed) == 1
    assert routed[0]["consumed_target_ordinal"] == 0
    assert routed[0]["hit"] is True
    missed = out["confirmed_missed_target_routing"]
    assert len(missed) == 1
    assert missed[0]["ordinal"] == 1
    assert missed[0]["mechanism"] == (
        "nearby_keydown_consumed_by_earlier_unresolved_target"
    )
    assert missed[0]["nearby_logical_keydowns"][0]["actually_consumed_target"] == 0
    assert abs(missed[0]["nearby_logical_keydowns"][0]["error_to_missed_due_ms"]) < .001


def test_scorer_replay_mismatch_refuses_target_attribution():
    segment, arm = prepared_shifted_down()
    incompatible = deepcopy(arm)
    incompatible["target_outcomes_posthoc"][1]["miss"] = False
    out = forensic.replay_scorer(segment, incompatible)
    assert out["score_replay_exact_match"] is False
    assert "confirmed_missed_target_routing" not in out
    assert "mismatch" in out


def test_no_motor_or_fingering_rerun_in_forensic_import_surface():
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(forensic))
    imports = [
        x.module or "" for x in ast.walk(tree)
        if isinstance(x, ast.ImportFrom)
    ]
    assert not any("timing_fingering_motor_policy" in x or
                   "hud_v14_forecast_source" in x or
                   "contact_16k" in x for x in imports)
