"""D27: verify preregistered single-step command intervention."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import a002v2_l0_d27_step413_single_command as d27


def mock_traces():
    before = [0] * d27.CONTROLS
    after = [0] * d27.CONTROLS
    before[100] = 1  # Different action index, same A0 left command.
    after[362] = 2
    after[d27.STEP] = 2
    after[d27.LATER_STEP] = 2
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


class TestD27(unittest.TestCase):
    def test_fixed_preregistration(self):
        self.assertEqual((d27.STEP, d27.LATER_STEP, d27.CONTROLS),
                         (413, 453, 480))
        self.assertEqual(d27.MAX_TICKS, 28800)
        self.assertEqual((d27.SOURCE_RUN, d27.SOURCE_ARTIFACT),
                         (41515190995, 13175252296))

    def test_one_step_both_directions_and_source_immutable(self):
        source = mock_traces()
        frozen = deepcopy(source)
        plans, prefix, suffix = d27.build_plans(source)
        self.assertEqual(source, frozen)
        self.assertEqual(prefix["effective_left_differences"], 0)
        self.assertEqual(suffix["effective_difference_steps"], [413, 453])
        for base, altered, donor in (
            ("before", "before_step413_from_after", "after_to_none"),
            ("after_to_none", "after_step413_from_before", "before"),
        ):
            self.assertEqual([i for i, (a, b) in enumerate(
                zip(plans[base], plans[altered])) if a != b], [413])
            self.assertEqual(plans[altered][413], plans[donor][413])
            self.assertEqual(plans[altered][453], plans[base][453])
        self.assertEqual(plans["before"], plans["before_negative_d27"])
        self.assertEqual(plans["after_to_none"], plans["after_negative_d27"])

    def test_missing_step413_effective_difference_rejected(self):
        source = mock_traces()
        source["after"]["steps"][413]["action"] = 0
        source["after_to_none"]["steps"][413]["action"] = 0
        with self.assertRaisesRegex(ValueError, "two-step effective"):
            d27.build_plans(source)

    def test_extra_effective_difference_rejected(self):
        source = mock_traces()
        source["after"]["steps"][430]["action"] = 2
        source["after_to_none"]["steps"][430]["action"] = 2
        with self.assertRaisesRegex(ValueError, "two-step effective"):
            d27.build_plans(source)

    def test_prefix_difference_rejected(self):
        source = mock_traces()
        source["after"]["steps"][300]["action"] = 2
        source["after_to_none"]["steps"][300]["action"] = 2
        with self.assertRaisesRegex(ValueError, "effective left-command prefix"):
            d27.build_plans(source)

    def test_wrong_step362_rejected(self):
        source = mock_traces()
        source["after_to_none"]["steps"][362]["action"] = 2
        with self.assertRaisesRegex(ValueError, "step362"):
            d27.build_plans(source)

    def test_malformed_action_rejected(self):
        source = mock_traces()
        source["after_to_none"]["steps"][413]["action"] = True
        with self.assertRaisesRegex(ValueError, "malformed"):
            d27.build_plans(source)

    def test_extra_at_432_checks_time_step_and_key(self):
        e = {"key": "left", "time_s": 4.327, "source_step": 432}
        self.assertTrue(d27.extra_at_432({"keydowns": [e]}))
        self.assertFalse(d27.extra_at_432({"keydowns": [
            {**e, "source_step": 433}]}))
        self.assertFalse(d27.extra_at_432({"keydowns": [
            {**e, "time_s": 4.328}]}))
        self.assertFalse(d27.extra_at_432({"keydowns": [
            {**e, "key": "right"}]}))

    def test_first_state_divergence_boundary(self):
        states = [{"time_s": (i + 1) / 1000, "x": 0}
                  for i in range(d27.CONTROLS * d27.SUBSTEPS)]
        changed = deepcopy(states)
        changed[4130]["x"] = 1
        result = d27.first_state_divergence(states, changed)
        self.assertEqual(result["tick_index"], 4130)
        self.assertTrue(result["after_step413_start"])
        changed[4129]["x"] = 1
        self.assertFalse(d27.first_state_divergence(
            states, changed)["after_step413_start"])

    def test_identical_states_report_none(self):
        states = [{"time_s": (i + 1) / 1000}
                  for i in range(d27.CONTROLS * d27.SUBSTEPS)]
        result = d27.first_state_divergence(states, deepcopy(states))
        self.assertIsNone(result["tick_index"])
        self.assertTrue(result["after_step413_start"])

    def test_motor_replay_deterministic(self):
        import a002v2_l0_d25_frozen_suffix as d25
        plan = [2] * 40 + [0] * 40
        a, _, state_a = d25.replay_plan(plan, name="d27-unit")
        b, _, state_b = d25.replay_plan(plan, name="d27-repeat")
        self.assertEqual(a, b)
        self.assertEqual(state_a, state_b)
        self.assertEqual(a["ticks"], 800)


if __name__ == "__main__":
    unittest.main()
