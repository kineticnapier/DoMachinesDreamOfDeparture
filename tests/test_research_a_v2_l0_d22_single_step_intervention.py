"""D22 single-step intervention and physical KeyDown accounting contracts."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import a002v2_l0_d21_command_latency as d21
import a002v2_l0_d22_single_step_intervention as d22


class ConstantNone(torch.nn.Module):
    def forward(self, observation):
        return torch.tensor([100.0, -100.0, -100.0, -100.0]), torch.tensor(0.0)


class TestD22(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = ConstantNone()

    def test_preregistered_single_factor(self):
        self.assertEqual(d22.INTERVENTION_STEP, 362)
        self.assertEqual((d22.CASE, d22.REPETITION, d22.SEED),
                         (4, 3, 320403))
        self.assertEqual((d22.INTERVAL_MS, d22.START_S), (240, 1.075))

    def test_baseline_identical_to_d21(self):
        d22_baseline = d22.trace_one(self.model)
        d21_baseline = d21.trace_one(self.model)
        self.assertEqual(d22.strip_extra(d22_baseline), d21_baseline)
        self.assertTrue(all(s["sampled_action"] == s["action"] for s
                            in d22_baseline["steps"]))
        self.assertEqual(d22.summarize(d22_baseline)["intervention_count"], 0)

    def test_one_step_override_and_rng_accounting(self):
        baseline = d22.trace_one(self.model)
        intervention = d22.trace_one(self.model, intervention_action=2,
                                     expected_sampled_action=0)
        self.assertEqual([s["step"] for s in intervention["steps"]
                          if s["intervention_applied"]], [362])
        self.assertEqual(intervention["steps"][362]["sampled_action"], 0)
        self.assertEqual(intervention["steps"][362]["action"], 2)
        self.assertEqual(intervention["steps"][362]["observation"],
                         baseline["steps"][362]["observation"])
        self.assertEqual(intervention["steps"][:362],
                         baseline["steps"][:362])
        self.assertEqual(sum(intervention["row"]["sampled_controls"]),
                         len(intervention["steps"]))
        self.assertEqual(sum(intervention["row"]["controls"]),
                         len(intervention["steps"]))
        self.assertTrue(all(s["action"] == s["sampled_action"] for i, s in
                            enumerate(intervention["steps"]) if i != 362))

    def test_repeatable_intervention(self):
        a = d22.trace_one(self.model, intervention_action=2,
                          expected_sampled_action=0)
        b = d22.trace_one(self.model, intervention_action=2,
                          expected_sampled_action=0)
        self.assertEqual(a, b)

    def test_reject_wrong_original_action(self):
        with self.assertRaisesRegex(ValueError, "unexpected sampled action"):
            d22.trace_one(self.model, intervention_action=2,
                          expected_sampled_action=1)

    def test_reject_noop_override(self):
        with self.assertRaisesRegex(ValueError, "invalid/no-op"):
            d22.trace_one(self.model, intervention_action=0,
                          expected_sampled_action=0)

    def test_physical_accounting(self):
        baseline = d22.trace_one(self.model)
        intervention = d22.trace_one(self.model, intervention_action=2,
                                     expected_sampled_action=0)
        d22.check_physical(baseline)
        d22.check_physical(intervention)
        for trace in (baseline, intervention):
            summary = d22.summarize(trace)
            self.assertEqual(summary["physical_keydowns"],
                             summary["hits"] + summary["extra_keydowns"])
            self.assertEqual(summary["intervention_count"],
                             0 if trace is baseline else 1)


if __name__ == "__main__":
    unittest.main()
