#!/usr/bin/env python3
"""D4: locate R2's first divergence with archived logs and only one PPO batch.

Preregistered in docs/research-a-v2-l0-d4-r2-first-update-protocol.md.
The original 256-episode R2 and D3 outcomes are read-only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import random
from unittest.mock import patch

import torch

import a002v2_l0_low_speed_ppo as l0
from a002v2_l0_r2_penalty_env import PenalizedLoggedLowSpeedEnv
from a002v2_l0_d3_actual_ppo_update import Recorder, TraceR2, instrument

R2_RUN = 41312136897
D3_RUN = 41369778494
R2_ORIGINAL_HEAD = "773ad68c4cb28f4c2ab8aaf2aeaffecdfd8ec95e"
D3_ORIGINAL_HEAD = "5d799475a8fdd471f95ae96c9e0f9cc730cbbd98"
PROTOCOL = "docs/research-a-v2-l0-d4-r2-first-update-protocol.md"
ARM, SEED = "a0", 0


def _scalar(x):
    if isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else {"shape": list(x.shape)}
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return repr(x)


def first_difference(expected, actual, path="$"):
    """Find the first exact difference; never round or use an adjustable tolerance."""
    if isinstance(expected, torch.Tensor) or isinstance(actual, torch.Tensor):
        if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
            return {"path": path, "expected": _scalar(expected), "actual": _scalar(actual)}
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            return {"path": path + ".tensor_spec",
                    "expected": [list(expected.shape), str(expected.dtype)],
                    "actual": [list(actual.shape), str(actual.dtype)]}
        if torch.equal(expected, actual):
            return None
        index = (expected != actual).nonzero(as_tuple=False)[0].tolist()
        ix = tuple(index)
        return {"path": path + "[" + ",".join(map(str, index)) + "]",
                "expected": _scalar(expected[ix]), "actual": _scalar(actual[ix])}
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                return {"path": path + "." + str(key), "expected": _scalar(expected.get(key, "<missing>")),
                        "actual": _scalar(actual.get(key, "<missing>"))}
            mismatch = first_difference(expected[key], actual[key], path + "." + str(key))
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return {"path": path + ".length", "expected": len(expected), "actual": len(actual)}
        for i, (a, b) in enumerate(zip(expected, actual)):
            mismatch = first_difference(a, b, path + f"[{i}]")
            if mismatch is not None:
                return mismatch
        return None
    if expected != actual:
        return {"path": path, "expected": _scalar(expected), "actual": _scalar(actual)}
    return None


def first_update_difference(expected_rows, actual_rows):
    if len(expected_rows) != 32 or len(actual_rows) != 32:
        raise ValueError("both archived training histories must contain 32 updates")
    for index, (a, b) in enumerate(zip(expected_rows, actual_rows), start=1):
        if a.get("episode") != 8 * index or b.get("episode") != 8 * index:
            raise ValueError("nonconsecutive archived PPO updates")
        mismatch = first_difference(a, b, f"training.update{index}")
        if mismatch is not None:
            return {"update": index, "episodes_completed": 8 * index, **mismatch}
    return None


def initial_state_digest(model):
    stream = BytesIO()
    torch.save(model.state_dict(), stream)
    return sha256(stream.getvalue()).hexdigest()


def first_batch_row(batch, losses):
    return {
        "episode": 8,
        "train_hit_rate": sum(x["score"]["hits"] for x in batch) / (8 * 16),
        "train_extra_keydowns": sum(x["score"]["extra_keydowns"] for x in batch),
        "mean_train_reward": sum(x["score"]["reward_sum"] for x in batch) / 8,
        "periods_ms": [x["interval_ms"] for x in batch],
        "starts_s": [round(x["start_s"], 6) for x in batch],
        **losses,
    }


def first_batch(instrumented):
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    model = l0.ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
    digest = initial_state_digest(model)
    recorder = Recorder("r2", ARM, SEED, TraceR2) if instrumented else None
    with (instrument(recorder) if instrumented else
          patch.object(l0, "LoggedLowSpeedEnv", PenalizedLoggedLowSpeedEnv)):
        stage0 = {
            "episode": 0,
            "argmax": l0.evaluate_set(arm=ARM, model=model, mode="argmax"),
            "sampled": l0.evaluate_set(arm=ARM, model=model, mode="sampled"),
        }
        batch = [l0.train_episode(model, arm=ARM, rng=rng) for _ in range(8)]
        losses = l0.update_ppo(model, optimizer, batch)
    if recorder is not None:
        if len(recorder.updates) != 1:
            raise AssertionError("D3 instrumentation did not record exactly one update")
        if len(recorder.events) != sum(x["score"]["physical_keydowns"] for x in batch):
            raise AssertionError("D3 instrumentation lost physical KeyDown events")
    return {
        "initial_state_sha256": digest,
        "stage0": stage0,
        "batch": batch,
        "first_update": first_batch_row(batch, losses),
        "weights": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "recorder": recorder,
    }


def compare_first_batches(a, b):
    for key in ("initial_state_sha256", "stage0", "batch", "first_update", "weights"):
        diff = first_difference(a[key], b[key], key)
        if diff is not None:
            return diff
    return None


def select_r2(original, diagnostic):
    if original.get("schema") != "research-a-v2-l0-r2-penalty-comparison-v1":
        raise ValueError("wrong original R2 schema")
    if diagnostic.get("schema") != "research-a-v2-l0-d3-original-ppo-update-replay-v1":
        raise ValueError("wrong D3 diagnostic schema")
    candidates = [m for m in original["r2_models"]
                  if m["arm"] == ARM and m["seed"] == SEED]
    replay = [m for m in diagnostic["models"]
              if m["mode"] == "r2" and m["arm"] == ARM and m["seed"] == SEED]
    if len(candidates) != 1 or len(replay) != 1:
        raise ValueError("missing or duplicate R2 A0 seed0 model")
    src, d3 = candidates[0], replay[0]
    if d3["source_checkpoint_sha256"] != src["checkpoint_sha256"]:
        raise ValueError("D3 compared against a different original checkpoint")
    if len(src["evaluations"]) != 3 or len(d3["evaluations"]) != 3:
        raise ValueError("missing archived evaluation stage")
    return src, d3


def run(r2_file: Path, d3_file: Path):
    original = json.loads(r2_file.read_text(encoding="utf-8"))
    diagnostic = json.loads(d3_file.read_text(encoding="utf-8"))
    src, d3 = select_r2(original, diagnostic)
    archived_first = first_update_difference(src["training"], d3["training_log"])
    archived_initial = first_difference(src["evaluations"][0], d3["evaluations"][0], "archived.stage0")
    archived_later = first_difference(src["evaluations"], d3["evaluations"], "archived.evaluations")
    baseline = first_batch(False)
    traced = first_batch(True)
    baseline_vs_traced = compare_first_batches(baseline, traced)
    source_initial_vs_baseline = first_difference(
        src["evaluations"][0], baseline["stage0"], "original.stage0_vs_baseline")
    source_initial_vs_traced = first_difference(
        src["evaluations"][0], traced["stage0"], "original.stage0_vs_traced")
    baseline_vs_original = first_difference(
        src["training"][0], baseline["first_update"], "original.update1_vs_baseline")
    traced_vs_original = first_difference(
        src["training"][0], traced["first_update"], "original.update1_vs_traced")
    traced_vs_d3 = first_difference(
        d3["training_log"][0], traced["first_update"], "d3.update1_vs_traced")
    source_digest_ok = (
        baseline["initial_state_sha256"] == traced["initial_state_sha256"] ==
        src["initial_state_sha256"])
    if not source_digest_ok or source_initial_vs_baseline or source_initial_vs_traced:
        classification = "invalid_initial_state_or_evaluation"
    elif baseline_vs_traced is not None:
        classification = "instrumentation_effect_candidate"
    elif baseline_vs_original is not None:
        classification = "original_vs_current_environment_or_runtime_candidate"
    elif archived_first is not None and archived_first["update"] > 1:
        classification = "first_update_exact_later_update_diverges"
    elif archived_first is None:
        classification = "all_archived_training_rows_exact_other_divergence"
    else:
        classification = "first_update_exact_but_archived_d3_update1_differs"
    recorder = traced["recorder"]
    events = recorder.events
    report = {
        "schema": "research-a-v2-l0-d4-r2-first-update-divergence-v1",
        "protocol": PROTOCOL,
        "source_runs": {"r2": R2_RUN, "d3": D3_RUN},
        "source_heads": {"r2": R2_ORIGINAL_HEAD, "d3": D3_ORIGINAL_HEAD},
        "arm": ARM, "seed": SEED, "episodes_per_branch": 8,
        "ppo_updates_per_branch": 1, "new_full_training": False,
        "classification": classification,
        "original_checkpoint_sha256": src["checkpoint_sha256"],
        "d3_checkpoint_sha256": d3["replay_checkpoint_sha256"],
        "initial_state_sha256_match": source_digest_ok,
        "archived_first_training_difference": archived_first,
        "archived_stage0_difference": archived_initial,
        "archived_evaluation_difference": archived_later,
        "baseline_vs_traced_first_difference": baseline_vs_traced,
        "original_stage0_vs_baseline": source_initial_vs_baseline,
        "original_stage0_vs_traced": source_initial_vs_traced,
        "original_update1_vs_baseline": baseline_vs_original,
        "original_update1_vs_traced": traced_vs_original,
        "d3_update1_vs_traced": traced_vs_d3,
        "first_update_baseline": baseline["first_update"],
        "first_update_traced": traced["first_update"],
        "first_update_archived": src["training"][0],
        "first_update_archived_d3": d3["training_log"][0],
        "physical_keydowns_traced": {
            "total": len(events), "hits": sum(e["hit"] for e in events),
            "extras": sum(not e["hit"] for e in events),
            "unmatched_onsets": sum(e["onset_step"] is None for e in events),
            "reused_onsets": sum(e["onset_reused"] for e in events),
        },
        "traced_physical_events": events,
        "d3_original_reproduced": bool(d3["original_reproduced"]),
        "historical_r2_gradients_verified": False,
        "formal_study_002_approved": False,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2-file", type=Path, required=True)
    parser.add_argument("--d3-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refuse to overwrite D4 diagnostic")
    report = run(args.r2_file, args.d3_file)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print("D4_FIRST_DIVERGENCE", json.dumps({
        k: v for k, v in report.items() if k in (
            "classification", "initial_state_sha256_match",
            "archived_first_training_difference", "archived_stage0_difference",
            "baseline_vs_traced_first_difference", "original_update1_vs_baseline",
            "original_update1_vs_traced", "d3_update1_vs_traced",
            "physical_keydowns_traced", "historical_r2_gradients_verified",
        )
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
