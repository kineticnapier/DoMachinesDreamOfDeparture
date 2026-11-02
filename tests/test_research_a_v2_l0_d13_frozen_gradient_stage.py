"""D13: archived, read-only gradient-stage comparison contracts."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import a002v2_l0_d13_frozen_gradient_stage as d13


def example(first_changed=False, second_changed=False):
    zeros = [0] * 1509
    n = {"schema": "research-a-v2-l0-d12-update-v1",
         "condition": "native", "initial_state_sha256": "a" * 64,
         "batch_sha256": "b" * 64,
         "runtime": {"python": "3.11", "torch": "2.5.1+cpu",
                     "platform": "Linux", "cpu_model": "CPU",
                     "torch_threads": 1, "cpu_capability": "AVX2"},
         "episodes": 8, "targets": 128, "hits": 0, "extras": 9,
         "physical_keydowns": 9, "ppo_updates": 1, "adam_steps": 2,
         "new_training_episodes": 0, "motor_steps": 0,
         "new_physical_keydowns": 0,
         "actor_and_critic_gradients": [{"weight": zeros.copy()},
                                        {"weight": zeros.copy()}],
         "losses": {"loss": 1., "policy_loss": 0., "value_loss": 0.,
                    "entropy": 0.}}
    d = deepcopy(n)
    d["condition"] = "default"
    d["runtime"]["cpu_capability"] = "NO AVX"
    if first_changed:
        d["actor_and_critic_gradients"][0]["weight"][0] = 1
    if second_changed:
        d["actor_and_critic_gradients"][1]["weight"][5] = 1
    c = {"schema": "research-a-v2-l0-d12-capture-v1",
         "initial_state_sha256": "a" * 64, "batch_sha256": "b" * 64,
         "episodes": 8, "targets": 128, "hits": 0, "extras": 9,
         "physical_keydowns": 9}
    stages = [d13.compare_parameters(x, y)
              for x, y in zip(n["actor_and_critic_gradients"],
                              d["actor_and_critic_gradients"])]
    previous = {"schema": "research-a-v2-l0-d12-comparison-v1",
                "classification": "ppo_update_difference_observed",
                "initial_state_sha256": "a" * 64, "batch_sha256": "b" * 64,
                "same_environment": True, "distinct_dispatch": True,
                "weights": {"different_elements": 10},
                "adam": {"exp_avg": {"different_elements": 572},
                         "exp_avg_sq": {"different_elements": 739}},
                "targets": 128, "hits": 0, "extras": 9,
                "physical_keydowns": 9,
                "loss_delta_default_minus_native":
                    {key: 0 for key in n["losses"]},
                "gradients_by_epoch": stages}
    return {"capture": c, "native": n, "default": d,
            "comparison": previous}


class TestD13(unittest.TestCase):
    def test_first_step_difference(self):
        result = d13.analyze(example(first_changed=True, second_changed=True))
        self.assertEqual(result["classification"],
                         "gradient_difference_before_first_adam_step")
        self.assertEqual(result["gradient_before_adam_step_1"]["different_elements"], 1)
        self.assertEqual(result["gradient_before_adam_step_2"]["different_elements"], 1)
        self.assertEqual(result["new_ppo_updates"], 0)
        self.assertEqual(result["physical_keydowns"], 9)

    def test_second_step_only(self):
        result = d13.analyze(example(second_changed=True))
        self.assertEqual(result["classification"],
                         "gradient_difference_only_before_second_adam_step")
        self.assertEqual(result["gradient_before_adam_step_1"]["different_elements"], 0)

    def test_no_difference_is_valid_negative_result(self):
        result = d13.analyze(example())
        self.assertEqual(result["classification"], "no_gradient_difference_in_two_steps")

    def test_d12_identity_mismatch_rejected(self):
        source = example()
        source["default"]["batch_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "identity"):
            d13.analyze(source)

    def test_d12_runtime_mismatch_rejected(self):
        source = example()
        source["default"]["runtime"]["cpu_model"] = "OTHER"
        with self.assertRaisesRegex(ValueError, "runtime"):
            d13.analyze(source)

    def test_dispatch_must_differ(self):
        source = example()
        source["default"]["runtime"]["cpu_capability"] = "AVX2"
        with self.assertRaisesRegex(ValueError, "dispatch"):
            d13.analyze(source)

    def test_missing_gradient_rejected(self):
        source = example()
        source["native"]["actor_and_critic_gradients"] = []
        with self.assertRaisesRegex(ValueError, "gradient count"):
            d13.analyze(source)

    def test_archived_gradient_count_must_match(self):
        source = example(first_changed=True)
        source["comparison"]["gradients_by_epoch"][0]["different_elements"] = 0
        with self.assertRaisesRegex(ValueError, "archived gradient"):
            d13.analyze(source)

    def test_physical_event_accounting_must_match(self):
        source = example()
        source["capture"]["hits"] = 1
        with self.assertRaisesRegex(ValueError, "physical"):
            d13.analyze(source)

    def test_nonfinite_float32_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            d13.decode_bits(0x7fc00000)

    def test_out_of_range_bits_rejected(self):
        with self.assertRaisesRegex(ValueError, "bit representation"):
            d13.decode_bits(1 << 40)

    def test_tensor_shape_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            d13.compare_parameters({"weight": [0]}, {"weight": [0, 1]})

    def test_d12_loss_difference_rejected(self):
        source = example()
        source["default"]["losses"]["loss"] = 2.
        with self.assertRaisesRegex(ValueError, "loss delta"):
            d13.analyze(source)


if __name__ == "__main__":
    unittest.main()
