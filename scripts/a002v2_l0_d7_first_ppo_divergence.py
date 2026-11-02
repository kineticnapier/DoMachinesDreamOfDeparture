#!/usr/bin/env python3
"""D7: stop at the first exact R1 PPO-update difference, at most 128 episodes.

Precommitted protocol: docs/research-a-v2-l0-d7-first-ppo-divergence-protocol.md
Uses the ORIGINAL train_model and D3 Recorder, never a rewritten PPO.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import sys
from unittest.mock import patch

import torch

import a002v2_l0_low_speed_ppo as l0
from a002v2_l0_d3_actual_ppo_update import Recorder, TraceR1, instrument
from a002v2_l0_d4_r2_first_update import first_difference
from a002v2_l0_d5_r1_prefix_order import get_sources, save_new_json

PROTOCOL = "docs/research-a-v2-l0-d7-first-ppo-divergence-protocol.md"
MAX_UPDATES = 16
ARM, SEED = "a0", 0


class StopComparison(Exception):
    """Intentional early stop; never interpreted as a failed training episode."""


def classify(stage0_difference, first_update_difference, checked_updates):
    if stage0_difference is not None:
        return "initial_evaluation_diverged"
    if first_update_difference is not None:
        return "first_update_divergence_identified"
    if checked_updates == MAX_UPDATES:
        return "no_divergence_through_128"
    return "invalid_diagnostic"


def validate_archives(original, d3):
    if (original["arm"], original["seed"]) != (ARM, SEED):
        raise ValueError("wrong original R1 model identity")
    if (d3["mode"], d3["arm"], d3["seed"]) != ("r1", ARM, SEED):
        raise ValueError("wrong D3 R1 model identity")
    if not d3["original_reproduced"]:
        raise ValueError("D3 did not reproduce original R1")
    if original["checkpoint_sha256"] != d3["source_checkpoint_sha256"]:
        raise ValueError("D3 source checkpoint differs from original R1")
    if original["checkpoint_sha256"] != d3["replay_checkpoint_sha256"]:
        raise ValueError("D3 replay checkpoint differs from original R1")
    if d3["training_log"] != original["training"]:
        raise ValueError("D3 training rows differ from original R1")
    if d3["evaluations"] != original["evaluations"]:
        raise ValueError("D3 evaluation rows differ from original R1")
    if len(original["training"]) != 32 or len(d3["all_updates"]) != 32:
        raise ValueError("expected 32 original and D3 updates")
    if len(original["evaluations"]) != 3:
        raise ValueError("expected three original evaluation stages")
    for i, row in enumerate(d3["all_updates"], 1):
        if row["update"] != i or len(row["actor_gradients_by_epoch"]) != 2:
            raise ValueError("missing or unordered D3 PPO gradients")
    events = defaultdict(list)
    for e in d3["all_training_keydown_events"]:
        if not 1 <= e["update"] <= 32:
            raise ValueError("out-of-range D3 physical event")
        events[e["update"]].append(e)
    return events


def training_row(batch, losses, index):
    return {
        "episode": index * l0.BATCH_EPISODES,
        "train_hit_rate": sum(x["score"]["hits"] for x in batch) /
                          (len(batch) * 16),
        "train_extra_keydowns": sum(x["score"]["extra_keydowns"] for x in batch),
        "mean_train_reward": sum(x["score"]["reward_sum"] for x in batch) /
                             len(batch),
        "periods_ms": [x["interval_ms"] for x in batch],
        "starts_s": [round(x["start_s"], 6) for x in batch],
        **losses,
    }


def weight_digest(model):
    data = BytesIO()
    torch.save(model.state_dict(), data)
    return sha256(data.getvalue()).hexdigest()


def physical_summary(events):
    return {
        "physical_keydowns": len(events),
        "hits": sum(bool(e["hit"]) for e in events),
        "extras": sum(not e["hit"] for e in events),
        "unmatched_onsets": sum(e["onset_step"] is None for e in events),
        "reused_onsets": sum(bool(e["onset_reused"]) for e in events),
    }


def run(r1_dir: Path, r2_dir: Path, d3_file: Path):
    torch.set_num_threads(1)
    r1, _original_r2, _failed_d3_r2 = get_sources(r1_dir, r2_dir, d3_file)
    originals = [m for m in r1 if (m["arm"], m["seed"]) == (ARM, SEED)]
    if len(originals) != 1:
        raise ValueError("missing or duplicated R1 A0 seed0")
    original = originals[0]
    diagnostic = json.loads(d3_file.read_text(encoding="utf-8"))
    matches = [m for m in diagnostic["models"]
               if (m["mode"], m["arm"], m["seed"]) == ("r1", ARM, SEED)]
    if len(matches) != 1:
        raise ValueError("missing or duplicated D3 R1 A0 seed0")
    d3 = matches[0]
    expected_events = validate_archives(original, d3)

    cpu_model = next(
        (line.partition(":")[2].strip() for line in
         Path("/proc/cpuinfo").read_text().splitlines()
         if line.startswith("model name")), "unknown"
    ) if Path("/proc/cpuinfo").exists() else "unknown"
    report = {
        "schema": "research-a-v2-l0-d7-r1-update-localization-v1",
        "protocol": PROTOCOL,
        "source_runs": {"r1": 41309837474, "r2": 41312136897,
                        "d3": 41369778494, "d5": 41380105087,
                        "d6": 41385518062},
        "model": {"arm": ARM, "seed": SEED},
        "max_episodes": 128,
        "new_full_256_episode_training": False,
        "historical_r2_gradients_verified": False,
        "formal_study_002_approved": False,
        "runtime": {
            "python": sys.version, "torch": torch.__version__,
            "platform": platform.platform(), "cpu_model": cpu_model,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "env": {k: os.environ.get(k) for k in
                    ("PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        },
        "stage0_difference": None,
        "first_update_difference": None,
        "updates": [],
        "classification": "in_progress",
    }
    recorder = Recorder("r1", ARM, SEED, TraceR1)
    observed_stage0 = {}
    original_eval = l0.evaluate_set

    def observed_eval(*args, **kwargs):
        result = original_eval(*args, **kwargs)
        if not report["updates"] and kwargs.get("mode") in ("argmax", "sampled"):
            mode = kwargs["mode"]
            if mode in observed_stage0:
                raise AssertionError("duplicate initial evaluation mode")
            observed_stage0[mode] = result
            if len(observed_stage0) == 2:
                report["stage0_difference"] = first_difference(
                    original["evaluations"][0],
                    {"episode": 0, **observed_stage0}, "stage0")
                if report["stage0_difference"] is not None:
                    raise StopComparison()
        return result

    def observed_update(model, optimizer, batch):
        losses = recorder.update_ppo(model, optimizer, batch)
        i = len(recorder.updates)
        if not 1 <= i <= MAX_UPDATES:
            raise AssertionError("D7 exceeded fixed update budget")
        current = recorder.updates[-1]
        events = [e for e in recorder.events if e["update"] == i]
        row = training_row(batch, losses, i)
        diffs = {
            "training_row": first_difference(
                original["training"][i - 1], row, f"update{i}.training"),
            "d3_recorded_update": first_difference(
                d3["all_updates"][i - 1], current, f"update{i}.d3_update"),
            "d3_physical_events": first_difference(
                expected_events[i], events, f"update{i}.physical_events"),
        }
        summary = physical_summary(events)
        if summary["physical_keydowns"] != current["event_summary"]["all_physical_keydowns"]:
            raise AssertionError("physical KeyDown accounting mismatch")
        report["updates"].append({
            "update": i, "episodes_completed": 8 * i,
            "differences": diffs,
            "physical": summary,
            "training_hit_rate": row["train_hit_rate"],
            "training_extra_keydowns": row["train_extra_keydowns"],
            "actor_gradient_l2_by_epoch": [
                e["actor_gradient_l2"] for e in current["actor_gradients_by_epoch"]],
            "policy_sha256_after_update": weight_digest(model),
        })
        if any(v is not None for v in diffs.values()):
            report["first_update_difference"] = {
                "update": i, "episodes_completed": 8 * i,
                "differences": diffs,
            }
            raise StopComparison()
        if i == MAX_UPDATES:
            raise StopComparison()
        return losses

    try:
        with instrument(recorder):
            with patch.object(l0, "evaluate_set", observed_eval), patch.object(
                    l0, "update_ppo", observed_update):
                l0.train_model(
                    arm=ARM, seed=SEED,
                    checkpoint_dir=Path("reports/research_a_v2_l0_d7/unused"),
                )
        raise AssertionError("D7 unexpectedly finished 256 episodes")
    except StopComparison:
        pass
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["classification"] = "invalid_diagnostic"
        return report

    if len(observed_stage0) != 2:
        report["error"] = "missing initial evaluation"
        report["classification"] = "invalid_diagnostic"
    else:
        report["classification"] = classify(
            report["stage0_difference"], report["first_update_difference"],
            len(report["updates"]))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1", type=Path, required=True)
    parser.add_argument("--r2", type=Path, required=True)
    parser.add_argument("--d3", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refuse to overwrite D7 result")
    try:
        result = run(args.r1, args.r2, args.d3)
    except Exception as exc:
        result = {
            "schema": "research-a-v2-l0-d7-r1-update-localization-v1",
            "protocol": PROTOCOL, "classification": "invalid_diagnostic",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    save_new_json(args.out, result)
    print("D7_RESULT", json.dumps({
        "classification": result["classification"],
        "stage0_difference": result.get("stage0_difference"),
        "first_update_difference": result.get("first_update_difference"),
        "updates_checked": len(result.get("updates", [])),
        "physical_keydowns_checked": sum(x["physical"]["physical_keydowns"]
                                          for x in result.get("updates", [])),
        "error": result.get("error"),
    }, ensure_ascii=False), flush=True)
    if result["classification"] == "invalid_diagnostic":
        raise AssertionError("D7 invalid diagnostic; preserve failure artifact")


if __name__ == "__main__":
    main()
