"""D25 tests: fixed suffix provenance and independent 1ms Motor replays."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import a002v2_l0_d25_frozen_suffix as d25


def mock_traces():
    before = [0] * d25.CONTROLS
    after = [0] * d25.CONTROLS
    after[d25.STEP] = 2
    before_closed = list(before)
    before_closed[d25.STEP] = 2
    before_closed[d25.STEP + 1] = 2
    after_closed = list(after)
    after_closed[d25.STEP] = 0
    after_closed[d25.STEP + 2] = 2
    plans = {
        "before": before, "after": after,
        "before_to_left": before_closed, "after_to_none": after_closed,
        "before_negative": before, "after_negative": after,
    }
    return {name: {"steps": [{"action": a} for a in actions]}
            for name, actions in plans.items()}


class TestD25(unittest.TestCase):
    def test_preregistered_limits(self):
        self.assertEqual((d25.STEP, d25.CONTROLS, d25.MAX_TICKS),
                         (362, 480, 38400))
        self.assertEqual((d25.DT, d25.TICKS_PER_CONTROL, d25.TIME_TOL),
                         (0.001, 10, 1e-8))
        self.assertEqual(d25.SOURCE_RUN, 41515190995)

    def test_fixed_suffix_changes_exactly_one_step(self):
        traces = mock_traces()
        original = deepcopy(traces)
        plans = d25.build_plans(traces)
        for base, fixed in (("before", "before_fixed_suffix_left"),
                            ("after", "after_fixed_suffix_none")):
            differences = [i for i, (a, b) in
                           enumerate(zip(plans[base], plans[fixed])) if a != b]
            self.assertEqual(differences, [362])
        self.assertEqual(traces, original)

    def test_different_suffix_does_not_leak_into_fixed(self):
        plans = d25.build_plans(mock_traces())
        self.assertNotEqual(plans["before_to_left"][363],
                            plans["before_fixed_suffix_left"][363])
        self.assertNotEqual(plans["after_to_none"][364],
                            plans["after_fixed_suffix_none"][364])

    def test_wrong_step362_rejected(self):
        traces = mock_traces()
        traces["before"]["steps"][362]["action"] = 2
        with self.assertRaisesRegex(ValueError, "step362"):
            d25.build_plans(traces)

    def test_archived_prefix_change_rejected(self):
        traces = mock_traces()
        traces["before_to_left"]["steps"][100]["action"] = 2
        with self.assertRaisesRegex(ValueError, "prefix"):
            d25.build_plans(traces)

    def test_wrong_length_rejected(self):
        traces = mock_traces()
        traces["after"]["steps"].pop()
        with self.assertRaisesRegex(ValueError, "malformed"):
            d25.build_plans(traces)

    def test_bad_action_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid action"):
            d25.replay_plan([0, 2, 4], name="bad")

    def test_oversized_replay_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid plan length"):
            d25.replay_plan([0] * 481, name="oversized")

    def test_motor_replay_deterministic_and_unscored(self):
        actions = [2] * 40 + [0] * 40
        original = list(actions)
        first, window, states = d25.replay_plan(actions, name="test")
        second, _, states2 = d25.replay_plan(actions, name="test-repeat")
        self.assertEqual(first, second)
        self.assertEqual(states, states2)
        self.assertEqual(actions, original)
        self.assertEqual(first["ticks"], 800)
        self.assertEqual(len(states), 800)
        self.assertEqual(window, [])
        self.assertGreaterEqual(first["keydown_count"], 1)
        self.assertTrue(all(e["key"] == "left" for e in first["keydowns"]))

    def test_archived_event_mismatch_rejected(self):
        data, _, _ = d25.replay_plan([2] * 40 + [0] * 40,
                                     name="test-events")
        original = [{"key": e["key"], "time_s": e["time_s"]}
                    for e in data["keydowns"]]
        trace = {"row": {"physical_trace": deepcopy(original)}}
        # check_archived_replay also checks the D22 first event time; a
        # deliberately wrong event must fail before that assertion.
        trace["row"]["physical_trace"][0]["time_s"] += 0.01
        with self.assertRaisesRegex(ValueError, "archived KeyDown mismatch"):
            d25.check_archived_replay("before", data, trace)

    def test_compare_pair_counts_only_post_step_changes(self):
        plans = d25.build_plans(mock_traces())
        series = {name: {"first_keydown_s": 3.7, "keydown_count": 1,
                         "keydowns": [{"time_s": 3.7, "key": "left"}]}
                  for name in plans}
        comparison = d25.compare_pair(plans, series, "before",
                                       "before_to_left",
                                       "before_fixed_suffix_left")
        self.assertEqual(comparison["suffix_action_differences"], 1)
        self.assertEqual(comparison["first_suffix_action_difference_step"], 363)
        self.assertEqual(comparison["effective_differences_before_first_keydown"], 1)
        self.assertEqual(comparison["fixed_minus_closed_ms"], 0)


if __name__ == "__main__":
    unittest.main()
