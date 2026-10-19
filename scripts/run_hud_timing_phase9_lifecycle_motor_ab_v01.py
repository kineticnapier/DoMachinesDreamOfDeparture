from __future__ import annotations

"""Fixed Train-bool Motor A/B for HUD v3 lifecycle v1.4 versus v1.5."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.planning.hud_v3_composite_forecast_source_v30 import (
    LIFECYCLE_V14,
    LIFECYCLE_V15,
)
if __package__:
    from scripts import run_countdown_hud_v3_motor_phase3 as phase3
else:
    import run_countdown_hud_v3_motor_phase3 as phase3


FORMAT = "dmdod-hud-timing-phase9-lifecycle-motor-ab-v0.1"
VERSIONS = (LIFECYCLE_V14, LIFECYCLE_V15)
EXPECTED_CHART_SHA256 = (
    "04981e92f74288b8aac7286e83420b933036d2caead9c157bc33b706779ad1b1"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "cca1bfcc2d8e2988eeabf602a57211c8d8b66a54bfbbcbe0e3a9409ec167d2d8"
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _lifecycle_contract_violations(frames: list[dict]) -> list[dict]:
    live: set[int] = set()
    seen: set[int] = set()
    violations = []
    for row in frames:
        pending = {item["local_id"] for item in row["pending"]}
        new = set(row["new_ids"])
        reused = set(row["reused_ids"])
        canceled = set(row["canceled_ids"])
        elapsed = set(row["elapsed_ids"])
        reasons = []
        if new & seen:
            reasons.append("id_recreated")
        if new & reused:
            reasons.append("new_reused_overlap")
        if canceled & elapsed:
            reasons.append("canceled_elapsed_overlap")
        if new | reused != pending:
            reasons.append("pending_partition")
        if reused - live:
            reasons.append("reused_not_live")
        if (canceled | elapsed) - live:
            reasons.append("retired_not_live")
        if live - reused != canceled | elapsed:
            reasons.append("disappeared_without_single_retirement")
        if reasons:
            violations.append({"step": row["step"], "reasons": reasons})
        seen |= new
        live = pending
    return violations


def _reservation_violations(frames: list[dict]) -> list[dict]:
    violations = []
    allowed_phases = {
        "reserved", "armed_in_flight", "keydown_observed", "releasing",
        "canceled_in_flight",
    }
    for row in frames:
        for field in ("reservations_after_command", "reservations_after_physics"):
            reservations = row[field]
            ids = [item["local_id"] for item in reservations]
            axes = [axis for item in reservations for axis in item["axes"]]
            reasons = []
            if len(ids) != len(set(ids)):
                reasons.append("duplicate_reservation_id")
            if len(axes) != len(set(axes)):
                reasons.append("axis_double_booked")
            if any(item["phase"] not in allowed_phases for item in reservations):
                reasons.append("unknown_reservation_phase")
            if reasons:
                violations.append({
                    "step": row["step"], "snapshot": field, "reasons": reasons,
                })
    return violations


def _forecast_differences(left: list[dict], right: list[dict]) -> dict:
    if len(left) != len(right):
        return {
            "frame_count_mismatch": [len(left), len(right)],
            "pending_id_steps": [],
            "due": [],
            "uncertainty": [],
        }
    pending_id_steps = []
    due = []
    uncertainty = []
    for a, b in zip(left, right, strict=True):
        a_pending = {item["local_id"]: item for item in a["pending"]}
        b_pending = {item["local_id"]: item for item in b["pending"]}
        if set(a_pending) != set(b_pending):
            pending_id_steps.append(a["step"])
        for local_id in set(a_pending) & set(b_pending):
            x, y = a_pending[local_id], b_pending[local_id]
            if abs(x["due_local_s"] - y["due_local_s"]) > 1e-12:
                due.append({"step": a["step"], "local_id": local_id})
            if x["uncertainty_s"] != y["uncertainty_s"]:
                uncertainty.append({"step": a["step"], "local_id": local_id})
    return {
        "frame_count_mismatch": None,
        "pending_id_steps": pending_id_steps,
        "due": due,
        "uncertainty": uncertainty,
    }


def _physical_differences(left: dict, right: dict) -> dict:
    fields = ("physical_keydowns", "logical_keydowns")
    return {
        field: [] if left[field] == right[field] else {
            LIFECYCLE_V14: left[field], LIFECYCLE_V15: right[field],
        }
        for field in fields
    }


def _lifecycle_counts(frames: list[dict]) -> dict:
    return {
        name: sum(len(row[f"{name}_ids"]) for row in frames)
        for name in ("new", "reused", "canceled", "elapsed")
    }


def evaluate(arms: dict[str, dict]) -> dict:
    old, new = arms[LIFECYCLE_V14], arms[LIFECYCLE_V15]
    forecast_differences = _forecast_differences(
        old["frames"], new["frames"]
    )
    physical_differences = _physical_differences(old, new)
    lifecycle_violations = {
        version: _lifecycle_contract_violations(arms[version]["frames"])
        for version in VERSIONS
    }
    reservation_violations = {
        version: _reservation_violations(arms[version]["frames"])
        for version in VERSIONS
    }
    old_stats, new_stats = old["stats"], new["stats"]
    old_targets = old["target_outcomes_posthoc"]
    new_targets = new["target_outcomes_posthoc"]
    second_safe_abstention = all(
        len(arm["target_outcomes_posthoc"]) > 1
        and arm["target_outcomes_posthoc"][1]["miss"]
        and arm["target_outcomes_posthoc"][1]["keydown_s"] is None
        and arm["scheduling"]["insufficient_lead"] >= 1
        for arm in arms.values()
    )
    gates = {
        "hits_not_decreased": new_stats["hits"] >= old_stats["hits"],
        "too_early_not_increased": (
            new_stats["too_early"] <= old_stats["too_early"]
        ),
        "overload_not_increased": (
            int(new_stats["overloaded"]) <= int(old_stats["overloaded"])
        ),
        "countdown_first_hit_maintained": bool(
            old_targets and new_targets
            and old_targets[0]["hit"] and new_targets[0]["hit"]
        ),
        "forecast_due_unchanged": not forecast_differences["due"],
        "forecast_uncertainty_unchanged": not forecast_differences["uncertainty"],
        "pending_id_sets_unchanged": (
            forecast_differences["frame_count_mismatch"] is None
            and not forecast_differences["pending_id_steps"]
        ),
        "event_id_contract_violations_zero": not any(
            lifecycle_violations.values()
        ),
        "reservation_state_valid": not any(reservation_violations.values()),
        "second_target_safe_abstention_maintained": second_safe_abstention,
    }
    return {
        "verdict": (
            "MOTOR_INTEGRATION_PASS" if all(gates.values())
            else "MOTOR_INTEGRATION_BLOCKED"
        ),
        "gates": gates,
        "forecast_differences": forecast_differences,
        "physical_differences": physical_differences,
        "lifecycle_contract_violations": lifecycle_violations,
        "reservation_violations": reservation_violations,
        "lifecycle_counts": {
            version: _lifecycle_counts(arms[version]["frames"])
            for version in VERSIONS
        },
        "target_differences": [
            {"ordinal": index, LIFECYCLE_V14: a, LIFECYCLE_V15: b}
            for index, (a, b) in enumerate(zip(old_targets, new_targets, strict=True))
            if a != b
        ],
        "stats_delta_v15_minus_v14": {
            "hits": new_stats["hits"] - old_stats["hits"],
            "misses": new_stats["misses"] - old_stats["misses"],
            "too_early": new_stats["too_early"] - old_stats["too_early"],
            "physical_keydowns": (
                new_stats["physical_keydowns"] - old_stats["physical_keydowns"]
            ),
            "x_accuracy_percent": (
                new_stats["x_accuracy_percent"] - old_stats["x_accuracy_percent"]
            ),
            "timing_mae_ms": (
                new_stats["timing_mae_ms"] - old_stats["timing_mae_ms"]
            ),
        },
    }


def run(chart: Path, checkpoint: Path, *, wall_timeout_s: float) -> dict:
    if not chart.is_file() or _hash(chart) != EXPECTED_CHART_SHA256:
        raise ValueError("fixed Train bool chart is missing or has changed")
    if not checkpoint.is_file() or _hash(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("frozen Fingering checkpoint is missing or has changed")
    segment = build_playable_segment(
        compile_adofai(load_adofai(chart)), end_s=5.0
    )
    if len(segment.targets) != 9:
        raise ValueError("fixed smoke must contain exactly 9 targets")
    model = phase3.frozen.load_model(checkpoint)
    arms = {}
    for version in VERSIONS:
        print("starting lifecycle", version, flush=True)
        arms[version] = phase3.simulate(
            segment,
            model,
            "hud_v3_composite",
            wall_timeout_s=wall_timeout_s,
            lifecycle_version=version,
        )
        print(version, arms[version]["stats"], flush=True)
    analysis = evaluate(arms)
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": analysis["verdict"],
        "chart": chart.name,
        "chart_sha256": _hash(chart),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _hash(checkpoint),
        "scope_s": 5.0,
        "target_count": 9,
        "control_dt_s": phase3.CONTROL_DT_S,
        "physics_dt_s": phase3.PHYSICS_DT_S,
        "fixed_policy": {
            "selector": "network",
            "style_alpha": .5,
            "normal_lead_s": .044,
            "decision_lead_s": .069,
        },
        "analysis": analysis,
        "arms": arms,
        "causality": (
            "Both policies received only TimingFrameUpdate plus current Motor. "
            "Target truth and scorer traces were joined only after simulation."
        ),
        "scope_note": (
            "This is a fixed Train-bool safety smoke, not evidence of Motor "
            "performance improvement or generalization."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chart", type=Path, default=phase3.DEFAULT_CHART)
    parser.add_argument(
        "--checkpoint", type=Path, default=phase3.DEFAULT_CHECKPOINT
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-timeout-s", type=float, default=180.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = run(
        args.chart, args.checkpoint, wall_timeout_s=args.wall_timeout_s
    )
    result["run_wall_s"] = round(time.monotonic() - started, 6)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("saved:", args.output)
    print("verdict:", result["verdict"])
    print("gates:", result["analysis"]["gates"])
    if result["verdict"] != "MOTOR_INTEGRATION_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
