"""D21 single-case policy command / physical KeyDown diagnostic contracts."""
from pathlib import Path
import copy
import math
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import torch
import a002v2_l0_low_speed_ppo as l0
import a002v2_l0_d21_command_latency as d21


class TestD21(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        cls.model = l0.ActorCritic()

    def test_fixed_case(self):
        self.assertEqual((d21.CASE, d21.REPETITION, d21.SEED),
                         (4, 3, l0.eval_seed(4, 3)))
        self.assertEqual((d21.INTERVAL_MS, d21.START_S), (240, 1.075))

    def test_repeatability_full_step(self):
        a = d21.trace_one(self.model)
        b = d21.trace_one(self.model)
        self.assertEqual(a, b)
        self.assertEqual(a["row"]["steps"], len(a["steps"]))
        self.assertEqual(sum(a["row"]["controls"]), len(a["steps"]))

    def test_observation_probabilities_and_time(self):
        result = d21.trace_one(self.model)
        for i, step in enumerate(result["steps"]):
            self.assertEqual(step["step"], i)
            self.assertEqual(len(step["observation"]), 8)
            self.assertEqual(len(step["action_probabilities"]), 4)
            self.assertAlmostEqual(sum(step["action_probabilities"]), 1.0,
                                   places=5)
            self.assertTrue(all(math.isfinite(x) for x in step["observation"]))
            self.assertLess(step["time_s"], step["end_time_s"])
            if i:
                self.assertAlmostEqual(step["time_s"],
                                       result["steps"][i - 1]["end_time_s"],
                                       places=9)

    def test_physical_events_are_mapped_to_steps(self):
        result = d21.trace_one(self.model)
        emitted = [index for step in result["steps"]
                   for index in step["emitted_keydown_indices"]]
        self.assertEqual(emitted, list(range(len(result["physical_keydowns"]))))
        for i, event in enumerate(result["physical_keydowns"]):
            self.assertEqual(event["event_index"], i)
            step = result["steps"][event["source_step"]]
            self.assertIn(i, step["emitted_keydown_indices"])
            self.assertLessEqual(step["time_s"], event["event"]["time_s"])
            self.assertLessEqual(event["event"]["time_s"], step["end_time_s"] + 1e-8)
            self.assertEqual(event["event"], result["row"]["physical_trace"][i])
            if event["command_to_keydown_ms"] is not None:
                self.assertGreaterEqual(event["command_to_keydown_ms"], -1e-6)

    def test_first_difference_no_change(self):
        result = d21.trace_one(self.model)
        comparison = d21.compare_traces(result, copy.deepcopy(result))
        self.assertIsNone(comparison["first_action_difference"])
        self.assertIsNone(comparison["first_observation_difference"])
        self.assertIsNone(comparison["first_physical_difference"])
        self.assertEqual((comparison["hit_delta"], comparison["extra_delta"],
                          comparison["keydown_delta"]), (0, 0, 0))

    def test_first_action_difference_detected(self):
        a = d21.trace_one(self.model)
        b = copy.deepcopy(a)
        b["steps"][3]["action"] = (b["steps"][3]["action"] + 1) % 4
        comparison = d21.compare_traces(a, b)
        self.assertEqual(comparison["first_action_difference"]["step"], 3)
        self.assertIsNone(comparison["first_physical_difference"])

    def test_first_observation_difference_detected(self):
        a = d21.trace_one(self.model)
        b = copy.deepcopy(a)
        b["steps"][5]["observation"][0] += 0.1
        comparison = d21.compare_traces(a, b)
        self.assertEqual(comparison["first_observation_difference"]["step"], 5)
        self.assertIsNone(comparison["first_action_difference"])

    def test_reject_unequal_control_steps(self):
        a = d21.trace_one(self.model)
        b = copy.deepcopy(a)
        b["steps"].pop()
        with self.assertRaisesRegex(ValueError, "control step counts"):
            d21.compare_traces(a, b)


if __name__ == "__main__":
    unittest.main()
