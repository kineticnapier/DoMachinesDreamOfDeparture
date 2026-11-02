"""D15: contracts for first PPO backward gradient norm clipping."""
from pathlib import Path
import math
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import a002v2_l0_d15_first_ppo_clipping as d15
import a002v2_l0_d13_frozen_gradient_stage as d13
import a002_exploratory_ppo as ppo


class TestD15(unittest.TestCase):
    def test_inactive(self):
        self.assertEqual(d15.classify(0.75, 0), "clipping_inactive")

    def test_threshold_exactly_inactive(self):
        self.assertEqual(d15.classify(1.0, 0), "clipping_inactive")

    def test_active(self):
        self.assertEqual(d15.classify(1.25, 5), "clipping_active")

    def test_active_without_gradient_change_rejected(self):
        with self.assertRaisesRegex(ValueError, "did not change"):
            d15.classify(1.5, 0)

    def test_inactive_with_gradient_change_rejected(self):
        with self.assertRaisesRegex(ValueError, "inactive clipping"):
            d15.classify(0.5, 2)

    def test_nonfinite_rejected(self):
        for value in (float("nan"), float("inf"), -0.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                d15.classify(value, 0)

    def test_first_backward_does_not_update_weights(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        model = ppo.ActorCritic()
        saved = {name: param.detach().clone()
                 for name, param in model.named_parameters()}
        batch = []
        for i in range(8):
            batch.append({
                "observations": torch.zeros((2, 8), dtype=torch.float32),
                "actions": torch.tensor([i % 4, (i + 1) % 4], dtype=torch.long),
                "old_log_probs": torch.tensor([-1.2, -1.4], dtype=torch.float32),
                "returns": torch.tensor([float(i) / 10, -float(i) / 20]),
                "old_values": torch.tensor([0.0, 0.0]),
            })
        before, after, norm = d15.original_first_backward(model, batch)
        self.assertTrue(math.isfinite(norm))
        self.assertGreaterEqual(norm, 0)
        self.assertEqual(set(before), set(after))
        summary = d13.compare_parameters(before, after)
        self.assertEqual(summary["total_elements"], 1509)
        self.assertEqual(d15.classify(norm, summary["different_elements"]),
                         "clipping_active" if norm > 1 else "clipping_inactive")
        for name, param in model.named_parameters():
            self.assertTrue(torch.equal(param, saved[name]))

    def test_gradient_map_malformed_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            d13.compare_parameters({"w": [0]}, {"w": [0, 1]})


if __name__ == "__main__":
    unittest.main()
