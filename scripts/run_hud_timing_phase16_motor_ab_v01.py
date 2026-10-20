from __future__ import annotations

"""Phase 16 Train-only real Motor A/B: frozen v1.5 versus Phase15 opt-ins.

The Motor/Fingering policy, model, scorer, physics, speed and chart are shared.
Only the three explicitly enabled HUD-only Timing recoveries vary.
Evaluation reads target truth AFTER each complete policy/physical replay.
No image recognition, Hold release oracle, training, Validation20 or FINAL.
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from math import isfinite
import json
from pathlib import Path

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
if __package__:
    from scripts import run_countdown_hud_v3_motor_phase3 as phase3
    from scripts.audit_hud_timing_phase11_generalization_v01 import (
        REFERENCE, TRAIN_DIR, UNSEEN_END_S, verify_selection,
    )
    from scripts.run_hud_timing_phase9_lifecycle_motor_ab_v01 import (
        _lifecycle_contract_violations, _reservation_violations,
    )
else:
    import run_countdown_hud_v3_motor_phase3 as phase3
    from audit_hud_timing_phase11_generalization_v01 import (
        REFERENCE, TRAIN_DIR, UNSEEN_END_S, verify_selection,
    )
    from run_hud_timing_phase9_lifecycle_motor_ab_v01 import (
        _lifecycle_contract_violations, _reservation_violations,
    )


FORMAT = "dmdod-phase16-frozen-motor-ab-v0.1"
BASELINE = "v15_recovery_off"
CANDIDATE = "v15_midspin_twirl_checkpoint_on"
FIXED_FINGERING_SHA256 = (
    "cca1bfcc2d8e2988eeabf602a57211c8d8b66a54bfbbcbe0e3a9409ec167d2d8"
)
DEFAULT_CHECKPOINT = phase3.DEFAULT_CHECKPOINT
DEFAULT_MANIFEST = Path("reports/hud-timing-phase11-selection-20261019-r1.json")
BOOL_CHART = "04042 - bool - vo[ICE].adofai"
BOOL_SHA256 = REFERENCE[BOOL_CHART][0]


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _fixed_paths(train_dir: Path, manifest: dict, *, chart_name: str | None,
                 all_unseen: bool) -> tuple[tuple[Path, float], ...]:
    if chart_name is not None and all_unseen:
        raise ValueError("select either one chart or the entire fixed unseen16")
    if not all_unseen and chart_name is None:
        chart_name = BOOL_CHART
    if chart_name == BOOL_CHART and not all_unseen:
        path = train_dir / BOOL_CHART
        if not path.is_file() or _sha256(path) != BOOL_SHA256:
            raise ValueError("frozen Train bool chart missing or changed")
        return ((path, 5.0),)
    chosen = verify_selection(manifest, train_dir)
    if all_unseen:
        return tuple((path, UNSEEN_END_S) for path in chosen)
    matches = [path for path in chosen if path.name == chart_name]
    if len(matches) != 1:
        raise ValueError("chart_name must be an entry in frozen Train16 or bool")
    return ((matches[0], UNSEEN_END_S),)


def _compact(arm: dict, *,
             diagnostic_window_s: tuple[float, float] | None = None) -> dict:
    """Persist ordinary summaries; opt-in scorer/frame traces for diagnosis."""
    compact = {
        "stats": arm["stats"],
        "scheduling": arm["scheduling"],
        "target_outcomes_posthoc": arm["target_outcomes_posthoc"],
        "physical_keydowns": arm["physical_keydowns"],
        "logical_keydowns": arm["logical_keydowns"],
        "decisions": arm["decisions"],
        "recovery_flags": arm["recovery_flags"],
        "source_conflict_count": arm["source_conflict_count"],
        "timing_status_counts": arm["timing_status_counts"],
        "frame_count": arm["frame_count"],
        "run_wall_s": arm["run_wall_s"],
        "lifecycle_contract_violations": (
            _lifecycle_contract_violations(arm["frames"])
        ),
        "reservation_contract_violations": (
            _reservation_violations(arm["frames"])
        ),
    }
    if diagnostic_window_s is not None:
        start_s, end_s = diagnostic_window_s
        # The trace is evaluator truth, accessed only after the replay.
        # Do not align Motor reservations to target IDs by assuming that a
        # physical KeyDown necessarily scores its intended event.
        scorer_trace = arm["judgement_trace_posthoc"]
        compact["diagnostic_window_posthoc"] = {
            "start_s": start_s,
            "end_s": end_s,
            "frames": [
                row for row in arm["frames"]
                if start_s <= row["local_clock_s"] <= end_s
            ],
            "judgement_trace": [
                row for row in scorer_trace
                if start_s <= row["time_s"] <= end_s
            ],
            "target_outcomes": [
                row for row in arm["target_outcomes_posthoc"]
                if start_s <= row["due_s"] <= end_s
            ],
            "decisions": [
                row for row in arm["decisions"]
                if start_s <= row["due_local_s"] <= end_s
            ],
            "physical_keydowns": [
                row for row in arm["physical_keydowns"]
                if start_s <= row["time_s"] <= end_s
            ],
            "logical_keydowns": [
                row for row in arm["logical_keydowns"]
                if start_s <= row["time_s"] <= end_s
            ],
        }
    return compact


def _difference(old: dict, new: dict) -> dict:
    a, b = old["stats"], new["stats"]
    before, after = old["target_outcomes_posthoc"], new["target_outcomes_posthoc"]
    if a["targets"] != b["targets"] or len(before) != len(after):
        return {"valid": False, "reason": "target_population_changed"}
    newly_hit = [i for i, (x, y) in enumerate(zip(before, after, strict=True))
                 if not x["hit"] and y["hit"]]
    lost_hit = [i for i, (x, y) in enumerate(zip(before, after, strict=True))
                if x["hit"] and not y["hit"]]
    checks = {
        "no_hit_regression": not lost_hit,
        "too_early_not_increased": b["too_early"] <= a["too_early"],
        "overload_not_increased": int(b["overloaded"]) <= int(a["overloaded"]),
        "overload_counter_not_increased": (
            b["overload_counter"] <= a["overload_counter"]
        ),
        "reservation_collisions_not_increased": (
            new["scheduling"]["reservation_collisions"]
            <= old["scheduling"]["reservation_collisions"]
        ),
        "inflight_cancellations_not_increased": (
            new["scheduling"]["inflight_cancellations"]
            <= old["scheduling"]["inflight_cancellations"]
        ),
        "source_conflicts_zero": (
            old["source_conflict_count"] == new["source_conflict_count"] == 0
        ),
        "lifecycle_contract_valid": (
            not old["lifecycle_contract_violations"]
            and not new["lifecycle_contract_violations"]
        ),
        "reservation_contract_valid": (
            not old["reservation_contract_violations"]
            and not new["reservation_contract_violations"]
        ),
    }
    return {
        "valid": True, "checks": checks,
        "newly_hit_ordinals_posthoc": newly_hit,
        "lost_hit_ordinals_posthoc": lost_hit,
        "delta": {
            field: (None if a[field] is None or b[field] is None
                    else b[field] - a[field])
            for field in ("hits", "misses", "too_early", "physical_keydowns",
                          "timing_mae_ms", "x_accuracy_percent")
        },
        "same_physical_keydowns": (
            old["physical_keydowns"] == new["physical_keydowns"]
        ),
    }


def run(train_dir: Path, checkpoint: Path, manifest: dict, *,
        chart_name: str | None = None, all_unseen: bool = False,
        per_arm_timeout_s: float = 180.0,
        diagnostic_center_s: float | None = None,
        diagnostic_radius_s: float = .25) -> dict:
    if not checkpoint.is_file() or _sha256(checkpoint) != FIXED_FINGERING_SHA256:
        raise ValueError("frozen Fingering checkpoint missing or changed")
    if per_arm_timeout_s <= 0:
        raise ValueError("per-arm timeout must be positive")
    if not isfinite(diagnostic_radius_s) or diagnostic_radius_s <= 0:
        raise ValueError("diagnostic radius must be finite and positive")
    if diagnostic_center_s is not None and not isfinite(diagnostic_center_s):
        raise ValueError("diagnostic center must be finite")
    diagnostic_window_s = (
        None if diagnostic_center_s is None else (
            max(0.0, diagnostic_center_s - diagnostic_radius_s),
            diagnostic_center_s + diagnostic_radius_s,
        )
    )
    selections = _fixed_paths(train_dir, manifest, chart_name=chart_name,
                              all_unseen=all_unseen)
    model = phase3.frozen.load_model(checkpoint)
    cases = []
    for path, end_s in selections:
        segment = build_playable_segment(
            compile_adofai(load_adofai(path)), end_s=end_s
        )
        old = phase3.simulate(
            segment, model, "hud_v3_composite",
            wall_timeout_s=per_arm_timeout_s, lifecycle_version="v1.5",
            capture_judgement_trace=diagnostic_window_s is not None,
        )
        new = phase3.simulate(
            segment, model, "hud_v3_composite",
            wall_timeout_s=per_arm_timeout_s, lifecycle_version="v1.5",
            midspin_recovery=True,
            twirl_recovery=True,
            checkpoint_recovery=True,
            capture_judgement_trace=diagnostic_window_s is not None,
        )
        # Contract summaries are derived from the raw frame streams.
        # Compare the same compact representation that is persisted in JSON;
        # raw simulate() arms do not contain these derived summary keys.
        baseline_compact = _compact(old, diagnostic_window_s=diagnostic_window_s)
        candidate_compact = _compact(new, diagnostic_window_s=diagnostic_window_s)
        cases.append({
            "chart": path.name,
            "chart_sha256": _sha256(path),
            "scope_s": end_s,
            "target_count": len(segment.targets),
            "arms": {
                BASELINE: baseline_compact,
                CANDIDATE: candidate_compact,
            },
            "difference": _difference(baseline_compact, candidate_compact),
        })
    checks = [
        value for case in cases
        for value in case["difference"].get("checks", {}).values()
    ]
    regressions = sum(
        len(case["difference"].get("lost_hit_ordinals_posthoc", []))
        for case in cases
    )
    comparisons_valid = all(case["difference"].get("valid") for case in cases)
    safety_pass = comparisons_valid and bool(checks) and all(checks)
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": (
            "MOTOR_AB_NONREGRESSION_PASS" if safety_pass else
            "MOTOR_AB_REGRESSION_OR_INCOMPLETE"
        ),
        "scope": "fixed-unseen16" if all_unseen else "single-frozen-Train-chart",
        "evaluated_charts": len(cases),
        "physics_dt_s": phase3.PHYSICS_DT_S,
        "diagnostic_window_s": diagnostic_window_s,
        "control_dt_s": phase3.CONTROL_DT_S,
        "checkpoint_sha256": _sha256(checkpoint),
        "frozen_selection_source": str(DEFAULT_MANIFEST),
        "cases": cases,
        "total_hit_regressions": regressions,
        "safety_checks_all_pass": safety_pass,
        "interpretation": (
            "Physical Motor A/B with independent replays, not Timing "
            "coverage. No automatic policy acceptance from isolated hits."
        ),
    }


def _write_new(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--chart-name")
    scope.add_argument("--all-unseen", action="store_true")
    parser.add_argument("--per-arm-timeout-s", type=float, default=180.0)
    parser.add_argument("--diagnostic-center-s", type=float,
                        help="Opt-in posthoc scorer and frame trace near this time")
    parser.add_argument("--diagnostic-radius-s", type=float, default=.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = run(
        args.train_dir, args.checkpoint, manifest,
        chart_name=args.chart_name, all_unseen=args.all_unseen,
        per_arm_timeout_s=args.per_arm_timeout_s,
        diagnostic_center_s=args.diagnostic_center_s,
        diagnostic_radius_s=args.diagnostic_radius_s,
    )
    _write_new(args.output, result)
    print("verdict:", result["verdict"])
    for case in result["cases"]:
        print(case["chart"], case["difference"])
    print("output:", args.output)


if __name__ == "__main__":
    main()
