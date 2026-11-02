#!/usr/bin/env python3
"""D19: evaluate actual A0 Motor KeyDown before/after archived first PPO update."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

import a002v2_l0_low_speed_ppo as l0
import a002v2_l0_d12_frozen_ppo_cpu as d12
import a002v2_l0_d13_frozen_gradient_stage as d13
import a002v2_l0_d15_first_ppo_clipping as d15
import a002v2_l0_d18_frozen_policy_probability as d18

PROTOCOL = "docs/research-a-v2-l0-d19-paired-motor-protocol.md"
SCHEMA = "research-a-v2-l0-d19-paired-motor-v1"
EXPECTED = {"argmax": (9, 144), "sampled": (36, 576)}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_set(data, mode):
    count, targets = EXPECTED[mode]
    require(data["episodes"] == count and data["targets"] == targets,
            "evaluation count mismatch: " + mode)
    rows = data["rows"]
    require(len(rows) == count, "evaluation row count mismatch")
    expected_pairs = []
    for case in range(9):
        for rep in (range(4) if mode == "sampled" else (0,)):
            expected_pairs.append((case, rep))
    require([(r["case_index"], r["repetition"]) for r in rows] == expected_pairs,
            "evaluation grid/order mismatch")
    totals = {"hits": 0, "extra_keydowns": 0, "physical_keydowns": 0,
              "steps": 0}
    for r in rows:
        require(r["arm"] == "a0" and r["mode"] == mode,
                "evaluation arm/mode mismatch")
        case, rep = r["case_index"], r["repetition"]
        interval = l0.EVAL_INTERVALS_MS[case // 3]
        start = l0.EVAL_STARTS_S[case % 3]
        require(r["interval_ms"] == interval and r["start_s"] == start,
                "evaluation interval/start mismatch")
        seed = l0.eval_seed(case, rep) if mode == "sampled" else None
        require(r["sampling_seed"] == seed, "evaluation seed mismatch")
        require(r["targets"] == 16, "expected 16 targets per evaluation")
        require(r["hits"] + r["extra_keydowns"] == r["physical_keydowns"],
                "physical KeyDown accounting mismatch")
        require(len(r["physical_trace"]) == r["physical_keydowns"],
                "physical trace count mismatch")
        require(r["physical_left"] + r["physical_right"] ==
                r["physical_keydowns"], "physical finger count mismatch")
        require(sum(r["controls"]) == r["steps"],
                "action count does not match steps")
        for e in r["physical_trace"]:
            require(e["key"] in ("left", "right") and type(e["hit"]) is bool,
                    "invalid physical KeyDown trace")
        for key in totals:
            totals[key] += r[key]
    for key in ("hits", "extra_keydowns"):
        require(totals[key] == data[key], "evaluation total mismatch: " + key)
    require(totals["physical_keydowns"] ==
            data["physical_left"] + data["physical_right"],
            "physical KeyDown aggregate mismatch")
    return totals


def compare_sets(before, after, mode):
    b = validate_set(before, mode)
    a = validate_set(after, mode)
    pairs = []
    for br, ar in zip(before["rows"], after["rows"]):
        require((br["case_index"], br["repetition"], br["sampling_seed"]) ==
                (ar["case_index"], ar["repetition"], ar["sampling_seed"]),
                "paired evaluation condition mismatch")
        pairs.append({
            "case_index": br["case_index"],
            "repetition": br["repetition"],
            "seed": br["sampling_seed"],
            "hit_delta": ar["hits"] - br["hits"],
            "extra_delta": ar["extra_keydowns"] - br["extra_keydowns"],
            "physical_keydown_delta":
                ar["physical_keydowns"] - br["physical_keydowns"],
            "physical_trace_equal": br["physical_trace"] == ar["physical_trace"],
        })
    metrics = {}
    for key in ("hits", "extra_keydowns", "physical_keydowns"):
        metrics[key] = {
            "before": b[key], "after": a[key], "delta": a[key] - b[key],
            "before_per_target": b[key] / before["targets"],
            "after_per_target": a[key] / after["targets"],
            "pairs_increase": sum(p[{
                "hits": "hit_delta", "extra_keydowns": "extra_delta",
                "physical_keydowns": "physical_keydown_delta"}[key]] > 0 for p in pairs),
            "pairs_decrease": sum(p[{
                "hits": "hit_delta", "extra_keydowns": "extra_delta",
                "physical_keydowns": "physical_keydown_delta"}[key]] < 0 for p in pairs),
        }
    return {
        "mode": mode, "episodes": len(pairs), "targets": before["targets"],
        "metrics": metrics,
        "physical_traces_equal_pairs":
            sum(p["physical_trace_equal"] for p in pairs),
        "paired_rows": pairs,
    }


def run(d8_dir: Path, d12_dir: Path):
    d12.require_dispatch("native")
    require(os.environ.get("ATEN_CPU_CAPABILITY") is None,
            "D19 must run with native CPU dispatch")
    torch.set_num_threads(1)
    require(torch.get_num_threads() == 1, "D19 requires one CPU thread")
    require(torch.__version__ == "2.5.1+cpu",
            "D19 requires pinned torch 2.5.1+cpu")
    source = d15.load_source(d12_dir)
    capture, native, default, comparison = d13.validate_archived(source)
    model, initial_sha = d12.weights_digest(
        d8_dir / "native-initial-state.pt",
        d8_dir / "native.json", d8_dir / "default.json")
    require(initial_sha == capture["initial_state_sha256"],
            "initial weights SHA mismatch")
    batch = torch.load(d12_dir / "frozen-batch.pt",
                       map_location="cpu", weights_only=True)
    require(d12.batch_digest(batch) == capture["batch_sha256"],
            "frozen batch SHA mismatch")
    require(len(batch) == 8 and
            sum(ep["score"]["targets"] for ep in batch) == 128,
            "archived batch count mismatch")
    for key, value in (("hits", 0), ("extra_keydowns", 9),
                       ("physical_keydowns", 9)):
        require(sum(ep["score"][key] for ep in batch) == value,
                "archived physical record mismatch: " + key)
    require(native["adam_steps"] == 2 if "adam_steps" in native
            else native.get("adam_steps_per_condition", 2) == 2,
            "D12 archived Adam step count mismatch")
    require(d13.compare_parameters(native["weights_after"],
                                   default["weights_after"])["different_elements"]
            == comparison["weights"]["different_elements"] == 10,
            "D12 native/default weight difference mismatch")
    before_bits = d18.weights_bits(model)
    updated = d18.restore_updated(model, native["weights_after"])
    updated_bits = d18.weights_bits(updated)
    sets = {}
    for mode in ("argmax", "sampled"):
        before = l0.evaluate_set(arm="a0", model=model, mode=mode)
        control = l0.evaluate_set(arm="a0", model=model, mode=mode)
        validate_set(before, mode)
        validate_set(control, mode)
        require(before == control,
                "negative control physical/evaluation replay mismatch: " + mode)
        after = l0.evaluate_set(arm="a0", model=updated, mode=mode)
        comparison_row = compare_sets(before, after, mode)
        sets[mode] = {
            "before": before, "after": after,
            "negative_control_exact": True,
            "comparison": comparison_row,
        }
    require(d18.weights_bits(model) == before_bits and
            d18.weights_bits(updated) == updated_bits,
            "evaluation mutated policy weights")
    sampled = sets["sampled"]["comparison"]["metrics"]
    return {
        "schema": SCHEMA, "protocol": PROTOCOL,
        "classification": "valid_paired_motor_evaluation",
        "source_run": d13.SOURCE_RUN,
        "source_artifact": d13.SOURCE_ARTIFACT,
        "initial_state_sha256": initial_sha,
        "batch_sha256": capture["batch_sha256"],
        "runtime": d12.runtime(),
        "arm": "a0", "training_seed": 0,
        "archived_training": {"episodes": 8, "targets": 128,
                              "hits": 0, "extras": 9,
                              "physical_keydowns": 9,
                              "adam_steps": 2},
        "sets": sets,
        "sampled_hit_increase_without_extra_increase":
            (sampled["hits"]["delta"] > 0 and
             sampled["extra_keydowns"]["delta"] <= 0),
        "new_training_episodes": 0, "new_ppo_updates": 0,
        "new_adam_steps": 0,
        "new_motor_episodes": 3 * (9 + 36),
        "new_physical_keydowns": sum(
            validate_set(sets[m][k], m)["physical_keydowns"]
            for m in EXPECTED for k in ("before", "after")) +
            sum(validate_set(sets[m]["before"], m)["physical_keydowns"]
                for m in EXPECTED),
        "a1_evaluated": False, "a3_evaluated": False,
        "intervention_performed": False,
        "teacher_free_fingering_acquired": False,
        "human_equivalence_verified": False,
        "single_finger_physical_impossibility_verified": False,
        "formal_study_002_approved": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d8-dir", type=Path, required=True)
    parser.add_argument("--d12-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite D19 output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args.d8_dir, args.d12_dir)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError,
            json.JSONDecodeError) as exc:
        failure = {
            "schema": SCHEMA, "protocol": PROTOCOL,
            "classification": "invalid_paired_motor_evaluation",
            "error": str(exc),
            "new_training_episodes": 0,
            "new_ppo_updates": 0, "new_adam_steps": 0,
        }
        args.out.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False), flush=True)
        raise SystemExit(1)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print("D19_RESULT", json.dumps({
        "classification": result["classification"],
        "sampled": result["sets"]["sampled"]["comparison"]["metrics"],
        "argmax": result["sets"]["argmax"]["comparison"]["metrics"],
        "new_physical_keydowns": result["new_physical_keydowns"],
    }), flush=True)


if __name__ == "__main__":
    main()
