#!/usr/bin/env python3
"""D5: compare R2 first update in fresh processes, with/without four prior R1 replays.

Preregistered before execution. This script does not change training or scoring.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

import a002v2_l0_low_speed_ppo as l0
from a002v2_l0_d2_onset_advantage import source_models
from a002v2_l0_d3_actual_ppo_update import (
    Recorder, TraceR1, compare_archived, instrument,
)
from a002v2_l0_d4_r2_first_update import (
    first_batch, compare_first_batches, first_difference, select_r2,
)

PROTOCOL = "docs/research-a-v2-l0-d5-r1-prefix-order-protocol.md"
R1_RUN, R2_RUN, D3_RUN = 41309837474, 41312136897, 41369778494
ARMS_SEEDS = [("a0", 0), ("a0", 1), ("a1", 0), ("a1", 1)]


def save_new_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def get_sources(r1_dir: Path, r2_dir: Path, d3_file: Path):
    r1, _ = source_models(r1_dir, "r1")
    r2, r2_archive = source_models(r2_dir, "r2")
    if [(x["arm"], x["seed"]) for x in r1] != ARMS_SEEDS:
        raise ValueError("R1 archival order differs from original D3 replay order")
    diagnostic = json.loads(d3_file.read_text(encoding="utf-8"))
    original, replay = select_r2(r2_archive, diagnostic)
    if replay["original_reproduced"]:
        raise ValueError("D3 R2 is unexpectedly marked exact")
    return r1, original, replay


def physical_summary(recorder: Recorder) -> dict:
    events = recorder.events
    return {
        "total": len(events),
        "hits": sum(bool(e["hit"]) for e in events),
        "extras": sum(not e["hit"] for e in events),
        "unmatched_onsets": sum(e["onset_step"] is None for e in events),
        "reused_onsets": sum(bool(e["onset_reused"]) for e in events),
    }


def run_one(mode: str, r1_dir: Path, r2_dir: Path, d3_file: Path,
            out_dir: Path) -> None:
    if mode not in ("baseline", "preceded"):
        raise ValueError("unknown D5 mode")
    if out_dir.exists():
        raise FileExistsError("D5 output directory already exists: " + str(out_dir))
    torch.set_num_threads(1)
    r1, original, replay = get_sources(r1_dir, r2_dir, d3_file)
    out_dir.mkdir(parents=True, exist_ok=False)
    r1_verified = []
    if mode == "preceded":
        for entry in r1:
            arm, seed = entry["arm"], entry["seed"]
            recorder = Recorder("r1", arm, seed, TraceR1)
            with instrument(recorder):
                trained = l0.train_model(
                    arm=arm, seed=seed,
                    checkpoint_dir=out_dir / "r1-replayed-checkpoints",
                )
            compare_archived(entry, trained, f"d5/r1/{arm}/seed{seed}")
            if len(recorder.updates) != 32:
                raise AssertionError("R1 predecessor did not have 32 PPO updates")
            r1_verified.append({
                "arm": arm, "seed": seed, "updates": len(recorder.updates),
                "checkpoint_sha256": trained["checkpoint_sha256"],
                "exact_archived_replay": True,
            })
            print("D5_R1_PREDECESSOR_EXACT", json.dumps(r1_verified[-1]), flush=True)
    # This call is the same instrumented R2 first-batch path used in D4.
    result = first_batch(True)
    if result["initial_state_sha256"] != original["initial_state_sha256"]:
        raise AssertionError("R2 initial weights differ from original")
    data = {key: result[key] for key in (
        "initial_state_sha256", "stage0", "batch", "first_update", "weights"
    )}
    # Binary trace retains full float32 tensors and observations for exact comparison.
    torch.save(data, out_dir / "first-batch.pt")
    summary = {
        "schema": "research-a-v2-l0-d5-process-order-single-v1",
        "protocol": PROTOCOL, "mode": mode,
        "source_runs": {"r1": R1_RUN, "r2": R2_RUN, "d3": D3_RUN},
        "r1_predecessors_exact": r1_verified,
        "r2_arm": "a0", "r2_seed": 0,
        "r2_episodes": 8, "r2_ppo_updates": 1,
        "r2_initial_state_sha256": result["initial_state_sha256"],
        "original_stage0_difference": first_difference(
            original["evaluations"][0], result["stage0"], "original.stage0"),
        "d3_stage0_difference": first_difference(
            replay["evaluations"][0], result["stage0"], "d3.stage0"),
        "original_update1_difference": first_difference(
            original["training"][0], result["first_update"], "original.update1"),
        "d3_update1_difference": first_difference(
            replay["training_log"][0], result["first_update"], "d3.update1"),
        "physical_keydowns": physical_summary(result["recorder"]),
        "first_update": result["first_update"],
        "historical_r2_gradients_verified": False,
        "formal_study_002_approved": False,
    }
    save_new_json(out_dir / "summary.json", summary)
    print("D5_MODE_SUMMARY", json.dumps({
        k: v for k, v in summary.items() if k not in ("first_update",)
    }, ensure_ascii=False), flush=True)


def classify(*, baseline_matches_original: bool, predecessor_exact: bool,
             initial_digest_equal: bool, first_batch_equal: bool,
             preceded_matches_d3: bool) -> str:
    if not baseline_matches_original or not predecessor_exact or not initial_digest_equal:
        return "invalid_diagnostic"
    if first_batch_equal:
        return "no_r1_predecessor_effect_observed"
    if preceded_matches_d3:
        return "r1_predecessor_state_reproduces_archived_d3_divergence"
    return "r1_predecessor_effect_but_d3_not_fully_reproduced"


def compare(baseline_dir: Path, preceded_dir: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError("D5 comparison output already exists")
    a = torch.load(baseline_dir / "first-batch.pt", map_location="cpu",
                   weights_only=True)
    b = torch.load(preceded_dir / "first-batch.pt", map_location="cpu",
                   weights_only=True)
    sa = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))
    sb = json.loads((preceded_dir / "summary.json").read_text(encoding="utf-8"))
    if sa["mode"] != "baseline" or sb["mode"] != "preceded":
        raise ValueError("comparison inputs are not baseline/preceded")
    first = compare_first_batches(a, b)
    initial_digest_equal = a["initial_state_sha256"] == b["initial_state_sha256"]
    baseline_matches_original = (
        sa["original_stage0_difference"] is None and
        sa["original_update1_difference"] is None
    )
    predecessor_exact = (
        [(x["arm"], x["seed"]) for x in sb["r1_predecessors_exact"]]
        == ARMS_SEEDS and all(x["exact_archived_replay"]
                             for x in sb["r1_predecessors_exact"])
    )
    preceded_matches_d3 = (
        sb["d3_stage0_difference"] is None and
        sb["d3_update1_difference"] is None
    )
    classification = classify(
        baseline_matches_original=baseline_matches_original,
        predecessor_exact=predecessor_exact,
        initial_digest_equal=initial_digest_equal,
        first_batch_equal=first is None,
        preceded_matches_d3=preceded_matches_d3,
    )
    report = {
        "schema": "research-a-v2-l0-d5-r1-prefix-order-comparison-v1",
        "protocol": PROTOCOL, "classification": classification,
        "baseline_vs_preceded_first_difference": first,
        "baseline_matches_original_stage0_and_update1": baseline_matches_original,
        "preceded_matches_archived_d3_stage0_and_update1": preceded_matches_d3,
        "initial_r2_weights_sha_equal": initial_digest_equal,
        "r1_four_models_archival_replay_exact": predecessor_exact,
        "baseline": sa, "preceded": sb,
        "historical_r2_gradients_verified": False,
        "formal_study_002_approved": False,
    }
    save_new_json(out, report)
    print("D5_FINAL", json.dumps({
        k: v for k, v in report.items() if k not in ("baseline", "preceded")
    }, ensure_ascii=False), flush=True)
    if classification == "invalid_diagnostic":
        raise AssertionError("D5 preregistered validity checks failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("baseline", "preceded", "compare"))
    parser.add_argument("--r1", type=Path)
    parser.add_argument("--r2", type=Path)
    parser.add_argument("--d3", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--preceded-dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.mode == "compare":
        if not all((args.baseline_dir, args.preceded_dir, args.out)):
            parser.error("compare requires --baseline-dir --preceded-dir --out")
        compare(args.baseline_dir, args.preceded_dir, args.out)
    else:
        if not all((args.r1, args.r2, args.d3, args.out_dir)):
            parser.error("mode requires --r1 --r2 --d3 --out-dir")
        run_one(args.mode, args.r1, args.r2, args.d3, args.out_dir)


if __name__ == "__main__":
    main()
