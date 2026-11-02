"""D26: verify preregistered archived-command suffix swap boundaries."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import a002v2_l0_d26_post362_suffix_swap as d26


def mock_traces():
    before = [0] * d26.CONTROLS
    after = [0] * d26.CONTROLS
    before[100] = 1  # Different index but same physical left command in A0.
    after[100] = 0
    after[362] = 2
    after[380] = 2
    before_to_left = list(before)
    before_to_left[362] = 2
    after_to_none = list(after)
    after_to_none[362] = 0
    return {
        name: {"steps": [{"action": action} for action in actions]}
        for name, actions in {
            "before": before, "after": after,
            "before_to_left": before_to_left,
            "after_to_none": after_to_none,
            "before_negative": before,
            "after_negative": after,
        }.items()
    }


class TestD26(unittest.TestCase):
    def test_fixed_protocol(self):
        self.assertEqual((d26.STEP, d26.CONTROLS, d26.MAX_TICKS),
                         (362, 480, 28800))
        self.assertEqual((d26.SOURCE_RUN, d26.SOURCE_ARTIFACT),
                         (41515190995, 13175252296))

    def test_suffix_only_swap(self):
        source = mock_traces()
        original = deepcopy(source)
        plans, prefix, suffix = d26.build_plans(source)
        self.assertEqual(source, original)
        self.assertEqual(prefix["effective_left_differences"], 0)
        self.assertEqual(prefix["action_differences"], 1)
        self.assertEqual(suffix["effective_left_differences"], 1)
        self.assertEqual(suffix["first_effective_difference_step"], 380)
        self.assertEqual(plans["before_with_after_suffix"][:363],
                         plans["before"][:363])
        self.assertEqual(plans["before_with_after_suffix"][363:],
                         plans["after_to_none"][363:])
        self.assertEqual(plans["after_with_before_suffix"][:363],
                         plans["after_to_none"][:363])
        self.assertEqual(plans["after_with_before_suffix"][363:],
                         plans["before"][363:])

    def test_prefix_physical_difference_rejected(self):
        source = mock_traces()
        source["after"]["steps"][100]["action"] = 2
        source["after_to_none"]["steps"][100]["action"] = 2
        with self.assertRaisesRegex(ValueError, "effective left-command prefix"):
            d26.build_plans(source)

    def test_step362_mismatch_rejected(self):
        source = mock_traces()
        source["after_to_none"]["steps"][362]["action"] = 2
        with self.assertRaisesRegex(ValueError, "step362"):
            d26.build_plans(source)

    def test_short_stream_rejected(self):
        source = mock_traces()
        source["before"]["steps"].pop()
        with self.assertRaisesRegex(ValueError, "malformed"):
            d26.build_plans(source)

    def test_invalid_action_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid action"):
            d26.left_command(4)
        with self.assertRaisesRegex(ValueError, "invalid action"):
            d26.left_command(True)

    def test_event_matching_checks_time_and_key(self):
        a = {"keydowns": [{"key": "left", "time_s": 3.718}]}
        b = {"keydowns": [{"key": "left", "time_s": 3.718 + 1e-9}]}
        self.assertTrue(d26.event_match(a, b))
        b["keydowns"][0]["time_s"] += 0.001
        self.assertFalse(d26.event_match(a, b))
        b["keydowns"][0]["time_s"] = 3.718
        b["keydowns"][0]["key"] = "right"
        self.assertFalse(d26.event_match(a, b))
        b["keydowns"].append({"key": "left", "time_s": 4.0})
        self.assertFalse(d26.event_match(a, b))

    def test_empty_suffix_difference_is_valid(self):
        actions = [0] * d26.CONTROLS
        diff = d26.action_differences(actions, actions, 363, 480)
        self.assertEqual(diff["effective_left_differences"], 0)
        self.assertIsNone(diff["first_effective_difference_step"])

    def test_motor_replay_deterministic_and_unscored(self):
        import a002v2_l0_d25_frozen_suffix as d25
        plan = [2] * 40 + [0] * 40
        first, _, states = d25.replay_plan(plan, name="d26-unit")
        second, _, states_again = d25.replay_plan(plan, name="d26-repeat")
        self.assertEqual(first, second)
        self.assertEqual(states, states_again)
        self.assertEqual(first["ticks"], 800)
        self.assertGreaterEqual(first["keydown_count"], 1)


if __name__ == "__main__":
    unittest.main()
