"""D20: archived physical trace comparison contracts (no Motor or training)."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import a002v2_l0_d20_first_keydown_divergence as d20


def event(time_s=1.0, hit=False):
    return {"time_s": time_s, "key": "right", "hit": hit,
            "closest_target_ordinal": 0, "offset_ms": 10.0,
            "immediate_reward": 1.0 if hit else -0.05}


def make_row(mode, case, rep, trace):
    hits = sum(e["hit"] for e in trace)
    extras = len(trace) - hits
    return {
        "case_index": case, "repetition": rep, "arm": "a0",
        "mode": mode, "interval_ms": (210, 240, 250)[case // 3],
        "start_s": (0.835, 1.075, 1.290)[case % 3],
        "sampling_seed": 320000 + 100 * case + rep if mode == "sampled" else None,
        "targets": 16, "hits": hits, "extra_keydowns": extras,
        "physical_keydowns": len(trace), "physical_left": 0,
        "physical_right": len(trace), "physical_trace": trace,
        "steps": 10, "controls": [10, 0, 0, 0],
    }


def make_set(mode, traces):
    rows = []
    for case in range(9):
        for rep in (range(4) if mode == "sampled" else (0,)):
            rows.append(make_row(mode, case, rep, copy.deepcopy(traces.get((case, rep), []))))
    return {
        "episodes": len(rows), "targets": len(rows) * 16,
        "hits": sum(r["hits"] for r in rows),
        "extra_keydowns": sum(r["extra_keydowns"] for r in rows),
        "physical_left": 0, "physical_right": sum(r["physical_right"] for r in rows),
        "rows": rows,
    }


def make_data(sampled_before=None, sampled_after=None):
    sets = {}
    for mode in ("argmax", "sampled"):
        before = make_set(mode, sampled_before or {} if mode == "sampled" else {})
        after = make_set(mode, sampled_after or {} if mode == "sampled" else {})
        paired = []
        for br, ar in zip(before["rows"], after["rows"]):
            paired.append({
                "case_index": br["case_index"], "repetition": br["repetition"],
                "seed": br["sampling_seed"],
                "hit_delta": ar["hits"] - br["hits"],
                "extra_delta": ar["extra_keydowns"] - br["extra_keydowns"],
                "physical_keydown_delta": ar["physical_keydowns"] - br["physical_keydowns"],
                "physical_trace_equal": br["physical_trace"] == ar["physical_trace"],
            })
        metrics = {}
        for name in ("hits", "extra_keydowns"):
            metrics[name] = {"before": before[name], "after": after[name],
                             "delta": after[name] - before[name]}
        metrics["physical_keydowns"] = {
            "before": before["physical_right"], "after": after["physical_right"],
            "delta": after["physical_right"] - before["physical_right"],
        }
        sets[mode] = {
            "before": before, "after": after, "negative_control_exact": True,
            "comparison": {
                "paired_rows": paired, "metrics": metrics,
                "physical_traces_equal_pairs":
                    sum(p["physical_trace_equal"] for p in paired),
            },
        }
    b = sets["sampled"]["before"]
    a = sets["sampled"]["after"]
    return {
        "schema": d20.D19_SCHEMA, "classification": "valid_paired_motor_evaluation",
        "arm": "a0", "training_seed": 0, "new_motor_episodes": 135,
        "new_training_episodes": 0, "new_ppo_updates": 0, "new_adam_steps": 0,
        "sampled_hit_increase_without_extra_increase":
            a["hits"] > b["hits"] and a["extra_keydowns"] <= b["extra_keydowns"],
        "sets": sets,
    }


class TestD20(unittest.TestCase):
    def test_identical_empty_traces(self):
        result = d20.analyze(make_data(), require_archived=False)
        self.assertEqual(result["sets"]["argmax"]["identical_pairs"], 9)
        self.assertEqual(result["sets"]["sampled"]["identical_pairs"], 36)
        self.assertIsNone(result["sets"]["sampled"]["earliest_divergence"])
        self.assertEqual(result["new_motor_episodes"], 0)

    def test_extra_event_added(self):
        data = make_data(sampled_after={(0, 0): [event()]})
        result = d20.analyze(data, require_archived=False)
        sample = result["sets"]["sampled"]
        self.assertEqual(sample["divergent_pairs"], 1)
        first = sample["earliest_divergence"]["first_divergence"]
        self.assertEqual(first["event_index"], 0)
        self.assertIsNone(first["before"])
        self.assertEqual(first["after"]["time_s"], 1.0)

    def test_hit_replaced_by_extra_in_same_pair(self):
        data = make_data(sampled_before={(2, 1): [event(hit=True)]},
                         sampled_after={(2, 1): [event(hit=False)]})
        result = d20.analyze(data, require_archived=False)
        self.assertEqual(result["sets"]["sampled"]["hit_decrease_pairs"], 1)
        self.assertEqual(result["sets"]["sampled"]["extra_increase_pairs"], 1)
        self.assertEqual(result["sets"]["sampled"]
                         ["hit_decrease_and_extra_increase_pairs"], 1)

    def test_first_difference_index_and_missing_event(self):
        first = d20.first_difference([event(1.0), event(2.0)], [event(1.0)])
        self.assertEqual(first["event_index"], 1)
        self.assertIsNone(first["after"])
        self.assertEqual(first["first_divergence_time_s"], 2.0)

    def test_reject_wrong_seed(self):
        data = make_data()
        data["sets"]["sampled"]["after"]["rows"][0]["sampling_seed"] += 1
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            d20.analyze(data, require_archived=False)

    def test_reject_nonfinite_event_time(self):
        data = make_data(sampled_before={(0, 0): [event(float("nan"))]})
        with self.assertRaisesRegex(ValueError, "invalid event time"):
            d20.analyze(data, require_archived=False)

    def test_reject_negative_control(self):
        data = make_data()
        data["sets"]["sampled"]["negative_control_exact"] = False
        with self.assertRaisesRegex(ValueError, "negative control"):
            d20.analyze(data, require_archived=False)

    def test_reject_saved_pair_inconsistency(self):
        data = make_data(sampled_after={(0, 0): [event()]})
        data["sets"]["sampled"]["comparison"]["paired_rows"][0]["extra_delta"] = 0
        with self.assertRaisesRegex(ValueError, "saved paired comparison"):
            d20.analyze(data, require_archived=False)

    def test_reject_wrong_archived_totals(self):
        with self.assertRaisesRegex(ValueError, "archived D19 totals"):
            d20.analyze(make_data(), require_archived=True)

    def test_reject_trace_accounting(self):
        data = make_data(sampled_after={(0, 0): [event()]})
        data["sets"]["sampled"]["after"]["rows"][0]["hits"] = 1
        with self.assertRaisesRegex(ValueError, "trace Hit/Extra mismatch"):
            d20.analyze(data, require_archived=False)


if __name__ == "__main__":
    unittest.main()
