"""D14: fixed float32 ULP classification and archived input invariants."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import a002v2_l0_d13_frozen_gradient_stage as d13
import a002v2_l0_d14_gradient_ulp_audit as d14
from test_research_a_v2_l0_d13_frozen_gradient_stage import example


def fixture(step1_bits=1, step2_bits=2):
    source = example()
    for index in range(292):
        source["default"]["actor_and_critic_gradients"][0]["weight"][index] = step1_bits
    for index in range(407):
        source["default"]["actor_and_critic_gradients"][1]["weight"][index] = step2_bits
    source["comparison"]["gradients_by_epoch"] = [
        d13.compare_parameters(n, d) for n, d in zip(
            source["native"]["actor_and_critic_gradients"],
            source["default"]["actor_and_critic_gradients"])]
    return source


class TestD14(unittest.TestCase):
    def test_ulp_monotonic_positive(self):
        self.assertEqual(d14.ulp_distance(0, 1), 1)
        self.assertEqual(d14.ulp_distance(0, 2), 2)
        self.assertEqual(d14.ulp_distance(1, 4), 3)

    def test_ulp_monotonic_negative(self):
        self.assertEqual(d14.ulp_distance(-1082130432, -1082130431), 1)
        self.assertEqual(d14.ulp_distance(-1082130431, -1082130432), 1)

    def test_signed_zero_difference_is_one_ulp(self):
        self.assertEqual(d14.ulp_distance(-2147483648, 0), 1)
        stage = d14.analyze_stage({"w": [-2147483648]}, {"w": [0]})
        self.assertEqual(stage["signed_zero_only_differences"], 1)
        self.assertEqual(stage["zero_to_nonzero_transitions"], 0)

    def test_multi_ulp_classification(self):
        result = d14.analyze(fixture())
        self.assertEqual(result["classification"], "multi_ulp_difference_observed")
        self.assertEqual(result["before_adam_step_1"]["different_elements"], 292)
        self.assertEqual(result["before_adam_step_1"]["ulp_histogram"]["1"], 292)
        self.assertEqual(result["before_adam_step_2"]["different_elements"], 407)
        self.assertEqual(result["before_adam_step_2"]["ulp_histogram"]["2"], 407)
        self.assertEqual(result["new_ppo_updates"], 0)

    def test_all_one_ulp_classification(self):
        result = d14.analyze(fixture(step2_bits=1))
        self.assertEqual(result["classification"], "all_differences_within_one_ulp")
        self.assertEqual(result["before_adam_step_2"]["more_than_one_ulp"], 0)

    def test_sign_reversal(self):
        stage = d14.analyze_stage({"w": [-1082130432]}, {"w": [1065353216]})
        self.assertEqual(stage["sign_reversals"], 1)
        self.assertGreater(stage["max_ulp_distance"], 1)

    def test_zero_to_nonzero(self):
        stage = d14.analyze_stage({"w": [0]}, {"w": [1]})
        self.assertEqual(stage["zero_to_nonzero_transitions"], 1)
        self.assertEqual(stage["signed_zero_only_differences"], 0)

    def test_histogram_boundaries(self):
        self.assertEqual([d14.bucket(i) for i in (1, 2, 3, 4, 5, 16, 17)],
                         ["1", "2", "3-4", "3-4", "5-16", "5-16", "17+"])

    def test_nonfinite_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            d14.ulp_distance(0, 0x7f800000)

    def test_bits_out_of_range_rejected(self):
        with self.assertRaisesRegex(ValueError, "bit representation"):
            d14.ulp_distance(0, 1 << 33)

    def test_malformed_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            d14.analyze_stage({"w": [0]}, {"w": [0, 1]})

    def test_missing_archived_gradient_difference_rejected(self):
        source = fixture()
        source["default"]["actor_and_critic_gradients"][0]["weight"] = [0] * 1509
        source["comparison"]["gradients_by_epoch"][0] = d13.compare_parameters(
            source["native"]["actor_and_critic_gradients"][0],
            source["default"]["actor_and_critic_gradients"][0])
        with self.assertRaisesRegex(ValueError, "classification mismatch"):
            d14.analyze(source)

    def test_d13_counts_cannot_be_reinterpreted(self):
        source = fixture()
        source["default"]["actor_and_critic_gradients"][0]["weight"][291] = 0
        source["comparison"]["gradients_by_epoch"][0] = d13.compare_parameters(
            source["native"]["actor_and_critic_gradients"][0],
            source["default"]["actor_and_critic_gradients"][0])
        with self.assertRaisesRegex(ValueError, "counts changed"):
            d14.analyze(source)

    def test_batch_identity_mismatch_rejected(self):
        source = fixture()
        source["default"]["batch_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "identity"):
            d14.analyze(source)

    def test_physical_keydown_accounting_rejected(self):
        source = fixture()
        source["capture"]["physical_keydowns"] = 10
        with self.assertRaisesRegex(ValueError, "physical"):
            d14.analyze(source)


if __name__ == "__main__":
    unittest.main()
