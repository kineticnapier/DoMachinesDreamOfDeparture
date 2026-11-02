"""D16 contracts: selected-action PPO signal, no training or optimizer update."""
from pathlib import Path
import math
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import a002v2_l0_d16_first_batch_action_signal as d16
import a002_exploratory_ppo as ppo


class TestD16(unittest.TestCase):
    @staticmethod
    def fixture():
        actions = torch.tensor([0, 1, 1, 3], dtype=torch.long)
        advantages = torch.tensor([1.0, -2.0, 3.0, -1.0])
        ratio = torch.tensor([1.0, 1.5, 0.5, 1.0])
        raw = ratio * advantages
        clipped = ratio.clamp(0.8, 1.2) * advantages
        objective = torch.min(raw, clipped)
        return actions, advantages, ratio, objective, raw, clipped

    def test_all_four_action_categories(self):
        result = d16.summarize(*self.fixture())
        self.assertEqual(result["steps"], 4)
        self.assertEqual([result["by_action"][name]["count"]
                          for name in d16.ACTION_NAMES], [1, 2, 0, 1])
        self.assertIsNone(result["by_action"]["left"]["advantage_mean"])
        self.assertEqual(result["by_action"]["right"]["positive_advantage"], 1)
        self.assertEqual(result["by_action"]["right"]["negative_advantage"], 1)

    def test_objective_is_original_clipped_min(self):
        result = d16.summarize(*self.fixture())
        self.assertEqual(result["by_action"]["right"]["clipped_side_selected"], 0)
        self.assertAlmostEqual(result["objective_sum"], -1.5)
        self.assertAlmostEqual(result["advantage_sum"], 1.0)

    def test_reject_invalid_action(self):
        data = list(self.fixture())
        data[0] = torch.tensor([0, 1, 4, 3])
        with self.assertRaisesRegex(ValueError, "invalid action"):
            d16.summarize(*data)

    def test_reject_nonfinite(self):
        data = list(self.fixture())
        data[1] = data[1].clone()
        data[1][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            d16.summarize(*data)

    def test_reject_objective_mismatch(self):
        data = list(self.fixture())
        data[3] = data[3].clone()
        data[3][0] += 0.5
        with self.assertRaisesRegex(ValueError, "minimum"):
            d16.summarize(*data)

    def test_reject_wrong_shape(self):
        data = list(self.fixture())
        data[2] = data[2][:2]
        with self.assertRaisesRegex(ValueError, "shape"):
            d16.summarize(*data)

    def test_forward_does_not_modify_model(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)
        model = ppo.ActorCritic()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        batch = []
        for i in range(8):
            batch.append({
                "observations": torch.full((2, 8), i / 100.0),
                "actions": torch.tensor([i % 4, (i + 1) % 4], dtype=torch.long),
                "old_log_probs": torch.tensor([-1.2, -1.4]),
                "returns": torch.tensor([float(i) / 10, -float(i) / 20]),
                "old_values": torch.tensor([0.0, 0.0]),
            })
        result = d16.frozen_forward(model, batch)
        self.assertEqual(result["steps"], 16)
        self.assertTrue(math.isfinite(result["policy_loss"]))
        for name, param in model.state_dict().items():
            self.assertTrue(torch.equal(param, before[name]))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_action_labels_match_original_l0(self):
        self.assertEqual(tuple(d16.ACTION_NAMES),
                         ("none", "right", "left", "both"))


if __name__ == "__main__":
    unittest.main()
