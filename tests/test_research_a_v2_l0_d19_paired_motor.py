"""D19: preregistered physical Motor pairing and accounting contracts."""
from pathlib import Path
import copy
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import a002v2_l0_low_speed_ppo as l0
import a002v2_l0_d19_paired_motor as d19


def fixture(mode="argmax", hits=0, extras=1):
    rows = []
    for case in range(9):
        for rep in (range(4) if mode == "sampled" else (0,)):
            trace = [{"time_s": 1.0, "key": "right", "hit": False,
                      "closest_target_ordinal": 0, "offset_ms": 10.0,
                      "immediate_reward": -0.05} for _ in range(extras)]
            trace += [{"time_s": 1.0, "key": "right", "hit": True,
                       "closest_target_ordinal": 0, "offset_ms": 0.0,
                       "immediate_reward": 1.0} for _ in range(hits)]
            rows.append({
                "case_index": case, "repetition": rep, "arm": "a0",
                "mode": mode, "interval_ms": l0.EVAL_INTERVALS_MS[case // 3],
                "start_s": l0.EVAL_STARTS_S[case % 3],
                "sampling_seed": l0.eval_seed(case, rep) if mode == "sampled" else None,
                "targets": 16, "hits": hits, "extra_keydowns": extras,
                "physical_keydowns": hits + extras,
                "physical_left": 0, "physical_right": hits + extras,
                "physical_trace": trace, "steps": 10, "controls": [10, 0, 0, 0],
            })
    return {
        "episodes": len(rows), "targets": 16 * len(rows),
        "hits": hits * len(rows), "extra_keydowns": extras * len(rows),
        "physical_left": 0, "physical_right": (hits + extras) * len(rows),
        "rows": rows,
    }


class TestD19(unittest.TestCase):
    def test_valid_argmax_grid(self):
        row = d19.validate_set(fixture(), "argmax")
        self.assertEqual(row["physical_keydowns"], 9)
        self.assertEqual(row["hits"], 0)

    def test_valid_sampled_grid(self):
        row = d19.validate_set(fixture("sampled"), "sampled")
        self.assertEqual(row["physical_keydowns"], 36)

    def test_reject_seed_mismatch(self):
        row = fixture("sampled")
        row["rows"][0]["sampling_seed"] += 1
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            d19.validate_set(row, "sampled")

    def test_reject_wrong_grid(self):
        row = fixture()
        row["rows"][0]["interval_ms"] = 260
        with self.assertRaisesRegex(ValueError, "interval/start mismatch"):
            d19.validate_set(row, "argmax")

    def test_reject_trace_mismatch(self):
        row = fixture()
        row["rows"][0]["physical_trace"] = []
        with self.assertRaisesRegex(ValueError, "trace count"):
            d19.validate_set(row, "argmax")

    def test_reject_keydown_mismatch(self):
        row = fixture()
        row["rows"][0]["physical_keydowns"] = 2
        with self.assertRaisesRegex(ValueError, "accounting mismatch"):
            d19.validate_set(row, "argmax")

    def test_compare_paired_hits_and_extras(self):
        before = fixture("sampled", hits=0, extras=1)
        after = fixture("sampled", hits=1, extras=0)
        result = d19.compare_sets(before, after, "sampled")
        self.assertEqual(result["metrics"]["hits"]["delta"], 36)
        self.assertEqual(result["metrics"]["extra_keydowns"]["delta"], -36)
        self.assertEqual(result["metrics"]["physical_keydowns"]["delta"], 0)
        self.assertEqual(result["physical_traces_equal_pairs"], 0)

    def test_identical_negative_control(self):
        before = fixture()
        control = copy.deepcopy(before)
        self.assertEqual(before, control)
        result = d19.compare_sets(before, control, "argmax")
        self.assertEqual(result["physical_traces_equal_pairs"], 9)
        self.assertEqual(result["metrics"]["hits"]["delta"], 0)

    def test_reject_paired_order_mismatch(self):
        before, after = fixture(), fixture()
        after["rows"][0], after["rows"][1] = after["rows"][1], after["rows"][0]
        with self.assertRaisesRegex(ValueError, "grid/order mismatch"):
            d19.compare_sets(before, after, "argmax")


if __name__ == "__main__":
    unittest.main()
