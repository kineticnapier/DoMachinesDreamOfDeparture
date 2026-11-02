#!/usr/bin/env python3
"""D20: describe first physical KeyDown differences in archived D19 paired traces.

No policy inference, learning, or Motor execution. Input is the D19 JSON artifact.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SCHEMA = "research-a-v2-l0-d20-archived-first-keydown-divergence-v1"
PROTOCOL = "docs/research-a-v2-l0-d20-first-physical-divergence-protocol.md"
SOURCE_RUN = 41485667328
SOURCE_ARTIFACT = 13161377329
D19_SCHEMA = "research-a-v2-l0-d19-paired-motor-v1"
GRID = {"argmax": (9, 144), "sampled": (36, 576)}
ARCHIVED_TOTALS = {
    "argmax": {"before": (0, 0, 0), "after": (0, 0, 0)},
    "sampled": {"before": (5, 44, 49), "after": (4, 46, 50)},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_trace(row):
    events = row["physical_trace"]
    require(isinstance(events, list), "trace must be list")
    require(len(events) == row["physical_keydowns"], "trace length mismatch")
    hits = extras = left = right = 0
    previous_time = -1.0
    for event in events:
        require(isinstance(event, dict), "event must be object")
        require(event.get("key") in ("left", "right"), "invalid key")
        require(type(event.get("hit")) is bool, "invalid hit flag")
        t = event.get("time_s")
        require(type(t) in (int, float) and math.isfinite(t) and t >= 0,
                "invalid event time")
        require(t >= previous_time, "trace time out of order")
        previous_time = t
        for field in ("offset_ms", "immediate_reward"):
            v = event.get(field)
            require(v is None or (type(v) in (int, float) and math.isfinite(v)),
                    "invalid " + field)
        ordinal = event.get("closest_target_ordinal")
        require(ordinal is None or (type(ordinal) is int and ordinal >= 0),
                "invalid closest target")
        hits += int(event["hit"])
        extras += int(not event["hit"])
        left += int(event["key"] == "left")
        right += int(event["key"] == "right")
    require(row["hits"] == hits and row["extra_keydowns"] == extras,
            "trace Hit/Extra mismatch")
    require(row["physical_keydowns"] == hits + extras,
            "physical KeyDown accounting mismatch")
    require(row["physical_left"] == left and row["physical_right"] == right,
            "physical finger accounting mismatch")
    require(row["targets"] == 16, "target count mismatch")
    require(sum(row["controls"]) == row["steps"], "action step count mismatch")
    return (hits, extras, hits + extras)


def first_difference(before, after):
    """Compare ordered physical events by index, not inferred causal identity."""
    for index in range(max(len(before), len(after))):
        a = before[index] if index < len(before) else None
        b = after[index] if index < len(after) else None
        if a != b:
            return {"event_index": index, "before": a, "after": b,
                    "first_divergence_time_s": min(
                        e["time_s"] for e in (a, b) if e is not None)}
    return None


def analyze(data, require_archived=True):
    require(data["schema"] == D19_SCHEMA, "D19 schema mismatch")
    require(data["classification"] == "valid_paired_motor_evaluation",
            "D19 classification invalid")
    require(data.get("arm") == "a0" and data.get("training_seed") == 0,
            "D19 arm/seed mismatch")
    require(data.get("new_motor_episodes") == 135, "D19 episode count mismatch")
    require(data.get("new_training_episodes") == 0 and
            data.get("new_ppo_updates") == 0 and
            data.get("new_adam_steps") == 0, "D19 learning count mismatch")
    output = {}
    for mode, (expected_count, expected_targets) in GRID.items():
        item = data["sets"][mode]
        require(item["negative_control_exact"] is True,
                "D19 negative control mismatch")
        before, after = item["before"], item["after"]
        require(before["episodes"] == after["episodes"] == expected_count and
                before["targets"] == after["targets"] == expected_targets,
                "D19 evaluation grid mismatch")
        b_rows, a_rows = before["rows"], after["rows"]
        require(len(b_rows) == len(a_rows) == expected_count,
                "D19 row count mismatch")
        d19_pairs = item["comparison"]["paired_rows"]
        require(len(d19_pairs) == expected_count, "D19 paired row count mismatch")
        totals = {"before": [0, 0, 0], "after": [0, 0, 0]}
        pairs = []
        for index, (br, ar, saved) in enumerate(zip(b_rows, a_rows, d19_pairs)):
            for row, side in ((br, "before"), (ar, "after")):
                counts = validate_trace(row)
                for k in range(3):
                    totals[side][k] += counts[k]
                require(row["arm"] == "a0" and row["mode"] == mode,
                        "D19 row arm/mode mismatch")
                require(row["case_index"] == index // (4 if mode == "sampled" else 1)
                        and row["repetition"] == (index % 4 if mode == "sampled" else 0),
                        "D19 case/repetition mismatch")
                require(row["interval_ms"] == (210, 240, 250)[row["case_index"] // 3]
                        and row["start_s"] == (0.835, 1.075, 1.290)[row["case_index"] % 3],
                        "D19 interval/start mismatch")
                expected_seed = (320000 + 100 * row["case_index"] + row["repetition"]
                                 if mode == "sampled" else None)
                require(row["sampling_seed"] == expected_seed, "D19 seed mismatch")
            require((br["case_index"], br["repetition"], br["sampling_seed"]) ==
                    (ar["case_index"], ar["repetition"], ar["sampling_seed"]),
                    "paired grid mismatch")
            difference = first_difference(br["physical_trace"], ar["physical_trace"])
            hit_delta = ar["hits"] - br["hits"]
            extra_delta = ar["extra_keydowns"] - br["extra_keydowns"]
            keydown_delta = ar["physical_keydowns"] - br["physical_keydowns"]
            require(saved["case_index"] == br["case_index"] and
                    saved["repetition"] == br["repetition"] and
                    saved["seed"] == br["sampling_seed"] and
                    saved["hit_delta"] == hit_delta and
                    saved["extra_delta"] == extra_delta and
                    saved["physical_keydown_delta"] == keydown_delta and
                    saved["physical_trace_equal"] == (difference is None),
                    "D19 saved paired comparison mismatch")
            pairs.append({
                "case_index": br["case_index"],
                "repetition": br["repetition"],
                "seed": br["sampling_seed"],
                "hit_delta": hit_delta,
                "extra_delta": extra_delta,
                "physical_keydown_delta": keydown_delta,
                "shared_prefix_events": (difference["event_index"] if difference
                                         else len(br["physical_trace"])),
                "first_divergence": difference,
            })
        for side, data_set in (("before", before), ("after", after)):
            hit, extra, keydowns = totals[side]
            require((hit, extra, keydowns) ==
                    (data_set["hits"], data_set["extra_keydowns"],
                     data_set["physical_left"] + data_set["physical_right"]),
                    "D19 set total mismatch")
            if require_archived:
                require(tuple(totals[side]) == ARCHIVED_TOTALS[mode][side],
                        "archived D19 totals mismatch")
        metrics = item["comparison"]["metrics"]
        for k, field in enumerate(("hits", "extra_keydowns", "physical_keydowns")):
            require(metrics[field]["before"] == totals["before"][k] and
                    metrics[field]["after"] == totals["after"][k] and
                    metrics[field]["delta"] == totals["after"][k] - totals["before"][k],
                    "D19 saved metric mismatch")
        equal = sum(p["first_divergence"] is None for p in pairs)
        require(equal == item["comparison"]["physical_traces_equal_pairs"],
                "D19 identical pair count mismatch")
        changed = [p for p in pairs if p["first_divergence"] is not None]
        earliest = min(changed, key=lambda p: (
            p["first_divergence"]["first_divergence_time_s"],
            p["case_index"], p["repetition"])) if changed else None
        output[mode] = {
            "episodes": expected_count,
            "targets": expected_targets,
            "totals": totals,
            "identical_pairs": equal,
            "divergent_pairs": len(changed),
            "hit_decrease_pairs": sum(p["hit_delta"] < 0 for p in pairs),
            "extra_increase_pairs": sum(p["extra_delta"] > 0 for p in pairs),
            "hit_decrease_and_extra_increase_pairs":
                sum(p["hit_delta"] < 0 and p["extra_delta"] > 0 for p in pairs),
            "earliest_divergence": earliest,
            "paired_rows": pairs,
        }
    sampled = output["sampled"]["totals"]
    improvement = (sampled["after"][0] > sampled["before"][0] and
                   sampled["after"][1] <= sampled["before"][1])
    require(data["sampled_hit_increase_without_extra_increase"] == improvement,
            "D19 recorded outcome mismatch")
    return {
        "schema": SCHEMA, "protocol": PROTOCOL,
        "classification": "valid_archived_trace_comparison",
        "source_run": SOURCE_RUN, "source_artifact": SOURCE_ARTIFACT,
        "source_commit": "08d8de523ffae10513645dfe95566060b4e6bf0d",
        "sets": output,
        "sampled_hit_increase_without_extra_increase": improvement,
        "new_training_episodes": 0, "new_ppo_updates": 0,
        "new_adam_steps": 0, "new_motor_episodes": 0,
        "new_physical_keydowns": 0,
        "a1_evaluated": False, "a3_evaluated": False,
        "intervention_performed": False,
        "teacher_free_fingering_acquired": False,
        "human_equivalence_verified": False,
        "single_finger_physical_impossibility_verified": False,
        "formal_study_002_approved": False,
        "limitation": ("D19 stores physical KeyDown traces but not the immediately "
                       "preceding policy observation or per-step action. Event-index "
                       "differences do not establish causal correspondence."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d19-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite D20 output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.d19_json.open(encoding="utf-8") as source:
            data = json.load(source)
        result = analyze(data)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        failure = {"schema": SCHEMA, "protocol": PROTOCOL,
                   "classification": "invalid_archived_trace_comparison",
                   "error": str(exc), "new_training_episodes": 0,
                   "new_ppo_updates": 0, "new_motor_episodes": 0}
        args.out.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False), flush=True)
        raise SystemExit(1)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print("D20_RESULT", json.dumps({
        mode: {k: v for k, v in result["sets"][mode].items()
               if k != "paired_rows"}
        for mode in GRID}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
