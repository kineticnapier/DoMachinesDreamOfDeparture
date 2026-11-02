"""D17 contracts: original first PPO gradient split, no optimizer or Motor."""
from pathlib import Path
import math
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import a002_exploratory_ppo as ppo
import a002v2_l0_d17_action_gradient as d17

class TestD17(unittest.TestCase):
    @staticmethod
    def fixture():
        torch.manual_seed(0)
        torch.set_num_threads(1)
        model = ppo.ActorCritic()
        batch = []
        for i in range(8):
            batch.append({
                "observations": torch.full((2, 8), i / 100.0),
                "actions": torch.tensor([i % 4, (i + 1) % 4], dtype=torch.long),
                "old_log_probs": torch.tensor([-1.2, -1.4]),
                "returns": torch.tensor([float(i) / 10, -float(i) / 20]),
                "old_values": torch.tensor([0.0, 0.0]),
            })
        return model, batch

    def test_all_action_groups_cover_batch(self):
        model, batch = self.fixture()
        result = d17.decompose(model, batch)
        self.assertEqual(result["steps"], 16)
        self.assertEqual(result["counts"], [4, 4, 4, 4])
        self.assertEqual(result["autograd_grad_calls"], 7)
        self.assertEqual(list(result["by_action"]), list(d17.ACTION_NAMES))

    def test_both_gradient_reconstructions(self):
        model, batch = self.fixture()
        result = d17.decompose(model, batch)
        self.assertLessEqual(result["policy_reconstruction_max_abs_error"], 1e-6)
        self.assertLessEqual(result["full_reconstruction_max_abs_error"], 1e-6)
        self.assertGreater(result["full_ppo_gradient_l2"], 0)

    def test_no_weight_mutation_or_grad_side_effect(self):
        model, batch = self.fixture()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        d17.decompose(model, batch)
        for name, param in model.state_dict().items():
            self.assertTrue(torch.equal(param, before[name]))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_actor_bias_vector_and_finite_metrics(self):
        model, batch = self.fixture()
        result = d17.decompose(model, batch)
        for row in result["by_action"].values():
            self.assertEqual(len(row["actor_bias_gradient"]), 4)
            self.assertTrue(all(math.isfinite(v) for v in row["actor_bias_gradient"]))
            self.assertTrue(math.isfinite(row["l2_norm"]))
            self.assertTrue(math.isfinite(row["dot_with_full_ppo_gradient"]))

    def test_reject_invalid_action(self):
        model, batch = self.fixture()
        batch[0]["actions"][0] = 4
        with self.assertRaisesRegex(ValueError, "invalid action"):
            d17.decompose(model, batch)

    def test_reject_nonfinite_input(self):
        model, batch = self.fixture()
        batch[0]["observations"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite input"):
            d17.decompose(model, batch)

    def test_reject_wrong_shape(self):
        model, batch = self.fixture()
        batch[0]["actions"] = batch[0]["actions"][:1]
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            d17.decompose(model, batch)

    def test_gradient_sums_detect_perturbation(self):
        a = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
        b = [torch.tensor([4.0, 5.0]), torch.tensor([6.0])]
        total = d17.sum_grads([a, b])
        self.assertEqual([t.tolist() for t in total], [[5.0, 7.0], [9.0]])
        self.assertEqual(d17.max_difference(total, total), 0.0)
        total[0][0] += 1e-3
        self.assertGreater(d17.max_difference(total, d17.sum_grads([a, b])), 1e-6)

    def test_zero_gradient_cosine_is_none(self):
        row = d17.gradient_metrics([torch.zeros(2)], [torch.ones(2)])
        self.assertIsNone(row["cosine_with_full_ppo_gradient"])
        self.assertEqual(row["l2_norm"], 0)

    def test_action_names_match_original(self):
        self.assertEqual(tuple(d17.ACTION_NAMES), ("none", "right", "left", "both"))

if __name__ == "__main__":
    unittest.main()
