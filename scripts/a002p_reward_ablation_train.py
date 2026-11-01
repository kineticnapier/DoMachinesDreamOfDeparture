#!/usr/bin/env python3
"""A002-P-R1 fixed-budget paired RL comparison of ONLY near-miss reward.

Protocol was committed before this code was run:
  docs/research-a-experiment-002p-r1-reward-ablation-protocol.md

R0 reruns the original A002-P exactly, now with explicit environment injection.
R1 changes only positive reward for KeyDown outside the hit window.
Both conditions retain the same Motor, architecture, seeds, training and eval.
This exploratory comparison DOES NOT establish learned human fingering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from a002_exploratory_ppo import (
    ActorCritic, ARMS, SEEDS, EPISODES,
    EVAL_INTERVALS_MS, EVAL_STARTS_S,
    run_seed,
)
from a002p_posthoc_policy_diagnostic import traced_episode, aggregate
from dmdod.research.a002_timing_only_env import TimingOnlyTwoFingerEnv
from dmdod.research.a002p_reward_ablation import NoPositiveNearMissEnv

MODES = (
    ("r0_original", TimingOnlyTwoFingerEnv),
    ("r1_no_positive_near_miss", NoPositiveNearMissEnv),
)
ORIGINAL_SOURCE_RUN = 41296142433
ORIGINAL_SOURCE_COMMIT = "69a8e3b54e8d8c8fa28c1e46e522a897de9ac6e4"
ORIGINAL_SOURCE_ARTIFACT = 13079276880
PROTOCOL = "docs/research-a-experiment-002p-r1-reward-ablation-protocol.md"


def check_original_r0(new_result: dict, reference: dict) -> list[dict]:
    """Compare 0/48/96 tests individually, not just their aggregate averages."""
    errors: list[dict] = []
    if new_result["arm"] != reference["arm"] or new_result["seed"] != reference["seed"]:
        raise ValueError("original control arm/seed mismatch")
    new_evaluations = new_result["evaluations"]
    original_evaluations = reference["evaluations"]
    if len(new_evaluations) != len(original_evaluations):
        return [{"reason": "evaluation_checkpoint_count_mismatch"}]
    for n, (current, old) in enumerate(zip(new_evaluations, original_evaluations)):
        if current["episode"] != old["episode"]:
            errors.append({"checkpoint": n, "reason": "episode_index_mismatch"})
        cur_ev, old_ev = current["evaluation"], old["evaluation"]
        for field in ("hits", "targets", "extra_keydowns", "full_hit_episodes"):
            if cur_ev[field] != old_ev[field]:
                errors.append({
                    "checkpoint": current["episode"], "metric": field,
                    "new": cur_ev[field], "old": old_ev[field],
                })
        if len(cur_ev["rows"]) != len(old_ev["rows"]):
            errors.append({"checkpoint": current["episode"], "reason": "evaluation_length"})
            continue
        for idx, (a, b) in enumerate(zip(cur_ev["rows"], old_ev["rows"])):
            for field in ("interval_ms", "start_s", "hits", "targets",
                          "extra_keydowns", "physical_keydowns", "matched_keys", "full_hit"):
                if a[field] != b[field]:
                    errors.append({
                        "checkpoint": current["episode"],
                        "eval_index": idx, "metric": field,
                        "new": a[field], "old": b[field],
                    })
    return errors


def check_detailed_replay(r: dict, *, env_cls: type, base: Path) -> dict:
    checkpoint = base / r["checkpoint_file"]
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != r["checkpoint_sha256"]:
        raise ValueError("output checkpoint checksum failed")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["arm"] == r["arm"] and state["seed"] == r["seed"]
    model = ActorCritic()
    model.load_state_dict(state["policy"], strict=True)
    model.eval()

    rows = [
        traced_episode(
            model, arm=r["arm"], interval_ms=interval, start_s=start,
            env_cls=env_cls,
        )
        for interval in EVAL_INTERVALS_MS for start in EVAL_STARTS_S
    ]
    original_rows = r["evaluations"][-1]["evaluation"]["rows"]
    if len(rows) != len(original_rows):
        raise AssertionError("traced evaluation length differs")
    for idx, (row, original) in enumerate(zip(rows, original_rows)):
        for key in ("hits", "targets", "extra_keydowns",
                    "physical_keydowns", "matched_keys", "interval_ms", "start_s"):
            if row[key] != original[key]:
                raise AssertionError(
                    f"trace mismatch for {r['arm']} seed={r['seed']} row={idx} key={key}: "
                    f"{row[key]} != {original[key]}"
                )
    return aggregate(rows)


def run(*, previous_pilot_dir: Path, output: Path, checkpoints: Path) -> dict:
    torch.set_num_threads(1)
    reference_path = previous_pilot_dir / "pilot.json"
    previous = json.loads(reference_path.read_text(encoding="utf-8"))
    if previous["schema"] != "research-a002p-pilot-ppo-v1" or previous["quick"]:
        raise ValueError("original pilot source is missing or incompatible")
    if previous["settings"]["seeds"] != list(SEEDS):
        raise ValueError("original seed set changed")
    original_lookup = {
        (r["arm"], r["seed"]): r for r in previous["all_results"]
    }
    if set(original_lookup) != {(arm, seed) for arm in ARMS for seed in SEEDS}:
        raise ValueError("incomplete original pilot source")

    all_results = []
    control_mismatches = []
    # Same seeded initialization and same separate start/BPM RNG per run.
    # The only differing factor for R1 is the near-miss reward.
    for arm in ARMS:
        for seed in SEEDS:
            for mode, env_cls in MODES:
                checkpoint_dir = checkpoints / mode
                r = run_seed(
                    arm=arm, seed=seed, episodes=EPISODES,
                    checkpoint_dir=checkpoint_dir, env_cls=env_cls,
                )
                if mode == "r0_original":
                    control_mismatches.extend(
                        [{"arm": arm, "seed": seed, **x}
                         for x in check_original_r0(r, original_lookup[(arm, seed)])]
                    )
                diagnostics = check_detailed_replay(
                    r, env_cls=env_cls, base=checkpoint_dir
                )
                all_results.append({
                    "reward_condition": mode,
                    "arm": arm,
                    "seed": seed,
                    "original_pilot_reproducible_for_r0": (
                        not any(x["arm"] == arm and x["seed"] == seed
                                for x in control_mismatches)
                        if mode == "r0_original" else None
                    ),
                    "training_record": r,
                    "final_diagnostic": diagnostics,
                })
                print(
                    "REWARD_ABLATION_PROGRESS",
                    json.dumps({
                        "condition": mode, "arm": arm, "seed": seed,
                        "final_hits": r["evaluations"][-1]["evaluation"]["hits"],
                        "final_extras": r["evaluations"][-1]["evaluation"]["extra_keydowns"],
                        "final_keydowns": diagnostics["physical_keydowns"],
                        "actual_keys": diagnostics["physical_down_by_key"],
                        "control_mismatch_count": len(control_mismatches),
                    }, ensure_ascii=False),
                    flush=True,
                )
    return {
        "schema": "research-a002p-r1-paired-reward-ablation-v1",
        "status": "exploratory; original A002-P and formal 002 unchanged",
        "source_protocol": PROTOCOL,
        "original_source_run": ORIGINAL_SOURCE_RUN,
        "original_source_commit": ORIGINAL_SOURCE_COMMIT,
        "original_source_artifact": ORIGINAL_SOURCE_ARTIFACT,
        "control_r0_matches_original_all_evaluations": not control_mismatches,
        "control_mismatches": control_mismatches,
        "training_conditions": {
            "modes": [m for m,_ in MODES],
            "arms": list(ARMS), "seeds": list(SEEDS),
            "episodes_per_seed": EPISODES,
            "eval_intervals_ms": list(EVAL_INTERVALS_MS),
            "eval_start_s": list(EVAL_STARTS_S),
            "physical_motor_modified": False,
            "student_fingering_teacher": False,
        },
        "all_results": all_results,
    }


def brief_report(report: dict) -> dict:
    return {
        "source_protocol": report["source_protocol"],
        "control_r0_matches_original_all_evaluations": report[
            "control_r0_matches_original_all_evaluations"
        ],
        "control_mismatches": report["control_mismatches"][:12],
        "rows": [
            {
                "reward": r["reward_condition"],
                "arm": r["arm"], "seed": r["seed"],
                "hits": r["final_diagnostic"]["hits"],
                "targets": r["final_diagnostic"]["targets"],
                "extras": r["final_diagnostic"]["extra_keydowns"],
                "real_keys": r["final_diagnostic"]["physical_down_by_key"],
                "positive_reward_on_extra": r["final_diagnostic"][
                    "extra_keydowns_with_positive_step_reward"
                ],
                "selection_fraction": r["final_diagnostic"]["action_selection_fraction"],
                "mean_entropy": r["final_diagnostic"]["mean_actor_entropy_nats"],
                "evaluations": [
                    {
                        "episode": e["episode"],
                        "hits": e["evaluation"]["hits"],
                        "extras": e["evaluation"]["extra_keydowns"],
                        "switch_fraction": e["evaluation"]["switch_fraction"],
                        "full_hit_episodes": e["evaluation"]["full_hit_episodes"],
                    }
                    for e in r["training_record"]["evaluations"]
                ],
                "checkpoint_sha256": r["training_record"]["checkpoint_sha256"],
            } for r in report["all_results"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True,
                        help="downloaded *original* A002-P artifact root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite report {args.output}")
    report = run(
        previous_pilot_dir=args.original,
        output=args.output,
        checkpoints=args.checkpoints,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+"\n",
        encoding="utf-8",
    )
    summary = brief_report(report)
    print("A002P_R1_FINAL_SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
    # A failure here keeps the original negative observation, all eight models,
    # and the mismatched control evidence in the uploaded artifacts.
    if not report["control_r0_matches_original_all_evaluations"]:
        raise SystemExit("original R0 not reproducible; causal comparison NOT VALID")


if __name__ == "__main__":
    main()
