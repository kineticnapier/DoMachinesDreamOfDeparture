"""D18 contracts: frozen policy probabilities, no training or Motor execution."""
from pathlib import Path
import math
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import a002_exploratory_ppo as ppo
import a002v2_l0_d12_frozen_ppo_cpu as d12
import a002v2_l0_d18_frozen_policy_probability as d18


class TestD18(unittest.TestCase):
    @staticmethod
    def fixture():
        torch.manual_seed(0)
        torch.set_num_threads(1)
        model = ppo.ActorCritic()
        obs = torch.linspace(-1, 1, d18.N * 8, dtype=torch.float32).reshape(d18.N, 8)
        actions = torch.arange(d18.N, dtype=torch.long) % 4
        return model, obs, actions

    def test_restored_weight_bits_exact_and_initial_unchanged(self):
        model, _, _ = self.fixture()
        before = d18.weights_bits(model)
        bits = {k: list(v) for k, v in before.items()}
        bits["actor.bias"][0] = d12.tensor_bits(
            torch.tensor([float(model.actor.bias[0].item()) + 0.125]))[0]
        restored = d18.restore_updated(model, bits)
        self.assertEqual(d18.weights_bits(restored), bits)
        self.assertEqual(d18.weights_bits(model), before)

    def test_reject_missing_weight_name(self):
        model, _, _ = self.fixture()
        bits = d18.weights_bits(model)
        bits.pop("actor.bias")
        with self.assertRaisesRegex(ValueError, "parameter names"):
            d18.restore_updated(model, bits)

    def test_reject_wrong_weight_length(self):
        model, _, _ = self.fixture()
        bits = d18.weights_bits(model)
        bits["actor.bias"] = bits["actor.bias"][:-1]
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            d18.restore_updated(model, bits)

    def test_reject_nonfinite_restored_weights(self):
        model, _, _ = self.fixture()
        bits = d18.weights_bits(model)
        bits["actor.bias"][0] = 0x7f800000
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            d18.restore_updated(model, bits)

    def test_policy_forward_is_deterministic_and_read_only(self):
        model, obs, _ = self.fixture()
        bits = d18.weights_bits(model)
        p = d18.policy_probabilities(model, obs)
        self.assertTrue(torch.equal(p, d18.policy_probabilities(model, obs)))
        self.assertEqual(p.shape, (d18.N, 4))
        self.assertLessEqual(float((p.sum(dim=1) - 1).abs().max()), 1e-6)
        self.assertEqual(d18.weights_bits(model), bits)
        self.assertTrue(all(param.grad is None for param in model.parameters()))

    def test_identical_probabilities_are_zero_change(self):
        model, obs, actions = self.fixture()
        p = d18.policy_probabilities(model, obs)
        row = d18.compare_probabilities(p, p, actions)
        self.assertEqual(row["argmax_changed_rows"], 0)
        self.assertEqual(row["l1_max"], 0)
        self.assertEqual(row["kl_before_after_max"], 0)
        for name in d18.ACTION_NAMES:
            self.assertEqual(row["by_action"][name]["unchanged_rows"], d18.N)

    def test_changed_probabilities_are_measured(self):
        _, _, actions = self.fixture()
        b = torch.tensor([[0.5, 0.2, 0.2, 0.1]]).repeat(d18.N, 1)
        a = torch.tensor([[0.1, 0.2, 0.6, 0.1]]).repeat(d18.N, 1)
        row = d18.compare_probabilities(b, a, actions)
        self.assertEqual(row["argmax_changed_rows"], d18.N)
        self.assertGreater(row["kl_before_after_mean"], 0)
        self.assertGreater(row["l1_mean"], 0)
        self.assertLess(row["by_action"]["none"]["mean_delta"], 0)
        self.assertGreater(row["by_action"]["left"]["mean_delta"], 0)
        self.assertEqual(row["by_action"]["right"]["unchanged_rows"], d18.N)

    def test_reject_wrong_shapes_and_invalid_actions(self):
        model, obs, actions = self.fixture()
        p = d18.policy_probabilities(model, obs)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            d18.compare_probabilities(p[:1], p, actions)
        actions[0] = 4
        with self.assertRaisesRegex(ValueError, "invalid selected action"):
            d18.compare_probabilities(p, p, actions)

    def test_reject_bad_observations(self):
        model, obs, _ = self.fixture()
        obs[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite observation"):
            d18.policy_probabilities(model, obs)


if __name__ == "__main__":
    unittest.main()
