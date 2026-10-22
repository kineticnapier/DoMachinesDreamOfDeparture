from __future__ import annotations

"""Phase 23: causal early-due, lead, and reference-intervention audit."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.stock_floor_state_machine import StockFloorStateMachine, StockRunMode, targets_from_playable_segment
from dmdod.adofai.stock_frame_state_machine import StockFrameStateMachine, StockQueuedKeyDown
from dmdod.adofai.stock_judgement_reference import stock_angle_boundaries
from dmdod.adofai.timing import compile_adofai
from dmdod.planning.predictive_timing_observability_v01 import (
    AcceptableStrikeWindow,
    DueInterval,
    VisiblePrefixFrame,
    assess_shared_safe_strike,
    compare_visible_prefixes,
)
from scripts.audit_hud_timing_phase13_setspeed_v01 import (
    _first_time,
    _floor_visible_with_marker,
    _intervene_numeric_setspeed,
    _midspin_pair_safe,
    capture_causal_hud,
)
from scripts.audit_phase20b_stock_floor_state_v01 import _edge_key, _local_ids, _logical_keydowns


FORMAT = "dmdod-phase23-predictive-timing-observability-v0.1"
CHART = "01645 - First town of this Journey remix - Irin.adofai"
CHART_SHA256 = "4a71411d454d3b10a10cd98165a2e9cce99e4b14c310d1e6847836fdc68fc0e1"
CONTROL_DT_S = 0.010
MOTOR_LEAD_S = 0.044


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _visible(rows: list[dict]) -> tuple[VisiblePrefixFrame, ...]:
    return tuple(
        VisiblePrefixFrame(
            float(row["observed_at_s"]),
            tuple(float(value) for value in row["hud263"]),
            tuple(sorted(row["countdown"].items())),
        )
        for row in rows
    )


def _hit_window(target) -> AcceptableStrikeWindow:
    boundary = stock_angle_boundaries(
        target.bpm, pitch=target.pitch_ratio if hasattr(target, "pitch_ratio") else 1.0
    ).counted_deg
    radius_s = boundary / (3.0 * target.bpm)
    return AcceptableStrikeWindow(target.episode_time_s - radius_s, target.episode_time_s + radius_s)


def _run_reference(targets, keydowns: list[dict], extras: list[dict], *, run_mode: StockRunMode, anchored: bool = False) -> dict:
    machine = StockFloorStateMachine(
        targets[36:] if anchored else targets,
        run_mode=run_mode,
        initial_floor=(targets[35].floor_index if anchored else None),
    )
    frames = StockFrameStateMachine(machine, frame_dt_s=CONTROL_DT_S, phase_s=0.0)
    rows = []
    for index, edge in enumerate(keydowns + extras):
        if anchored and float(edge["time_s"]) <= 3.174:
            continue
        frames.queue_keydown(StockQueuedKeyDown(str(edge.get("input_id", f"base:{index}")), float(edge["time_s"]), str(edge.get("contact", "audit"))))
    for frame_index in range(0 if not anchored else 318, 537):
        update = frames.update(frame_index)
        for transition in update.transitions:
            rows.append({
                "frame_time_s": update.frame_time_s,
                "input_id": next((item.keydown.input_id for item in update.processed_keydowns if item.input_transition is transition), None),
                "kind": transition.kind.value,
                "target_ordinal": transition.target_ordinal,
                "judgement": None if transition.judgement is None else getattr(transition.judgement, "value", transition.judgement),
                "automatic_miss": transition.automatic_miss,
                "failed": transition.failed,
            })
        if update.failed:
            break
    base_pairs = [
        row for row in rows
        if isinstance(row["input_id"], str) and row["input_id"].startswith("local:") and row["target_ordinal"] is not None
    ]
    event89 = [row for row in rows if row["target_ordinal"] == 89]
    return {
        "failed": machine.failed,
        "last_frame_s": rows[-1]["frame_time_s"] if rows else None,
        "base_input_mismatch_count": sum(int(row["input_id"].split(":", 1)[1]) != row["target_ordinal"] for row in base_pairs),
        "event36": [row for row in rows if row["target_ordinal"] == 36],
        "event37": [row for row in rows if row["target_ordinal"] == 37],
        "event89": event89,
        "event89_input_hit": any(row["input_id"] is not None and not row["automatic_miss"] and row["judgement"] not in ("too_early", "too_late") for row in event89),
        "selected_intervention_rows": [row for row in rows if row["input_id"] and str(row["input_id"]).startswith(("oracle:", "hud_candidate:"))],
    }


def _generalization(report_paths: list[Path], train_dir: Path) -> dict:
    rows: list[dict] = []
    for report_path in report_paths:
        if not report_path.is_file():
            continue
        data = json.loads(report_path.read_text(encoding="utf-8"))
        for case in data.get("cases", []):
            chart_path = train_dir / case["chart"]
            if not chart_path.is_file():
                continue
            segment = build_playable_segment(compile_adofai(load_adofai(chart_path)), end_s=float(case["scope_s"]))
            arm = case["arms"]["v15_midspin_twirl_checkpoint_on"]
            for decision in arm["decisions"]:
                if decision["reason"] != "insufficient_observable_lead":
                    continue
                ordinal = int(decision["local_id"])
                target = segment.targets[ordinal]
                neighborhood = segment.chart.floors[max(0, target.floor_index - 3): target.floor_index + 1]
                if any("SetSpeed" in floor.event_markers for floor in neighborhood):
                    cause = "future_setspeed_neighborhood"
                elif any(floor.midspin for floor in neighborhood):
                    cause = "midspin_neighborhood"
                else:
                    cause = "other_or_unverified"
                lead_ms = (float(decision["due_local_s"]) - float(decision["decision_s"])) * 1000.0
                rows.append({
                    "chart": case["chart"], "local_id": ordinal,
                    "decision_s": decision["decision_s"], "predicted_due_s": decision["due_local_s"],
                    "observable_lead_ms": lead_ms, "late_vs_44ms_ms": 44.0 - lead_ms,
                    "posthoc_neighborhood": cause,
                })
    counts = Counter(row["posthoc_neighborhood"] for row in rows)
    return {
        "source_reports": [str(path) for path in report_paths if path.is_file()],
        "sample_count": len(rows),
        "classification_counts": dict(counts),
        "lead_ms": {"minimum": min((row["observable_lead_ms"] for row in rows), default=None), "maximum": max((row["observable_lead_ms"] for row in rows), default=None), "all_below_44ms": all(row["observable_lead_ms"] < 44.0 for row in rows)},
        "events": rows,
        "claim_limit": "saved two-chart diagnostics only; neighborhood labels use evaluator chart structure after replay and are not Student inputs",
    }


def run(phase16_path: Path, phase20c_path: Path, train_dir: Path, generalization_reports: list[Path]) -> dict:
    chart_path = train_dir / CHART
    if _hash(chart_path) != CHART_SHA256:
        raise ValueError("fixed First town chart missing or changed")
    phase16 = json.loads(phase16_path.read_text(encoding="utf-8"))
    case = phase16["cases"][0]
    arm = case["arms"]["v15_midspin_twirl_checkpoint_on"]
    diagnostic = arm["diagnostic_window_posthoc"]
    phase20c = json.loads(phase20c_path.read_text(encoding="utf-8"))

    chart = load_adofai(chart_path)
    alternate_chart, intervention = _intervene_numeric_setspeed(chart, floor_index=37)
    original_segment = build_playable_segment(compile_adofai(chart), end_s=12.0)
    alternate_segment = build_playable_segment(compile_adofai(alternate_chart), end_s=12.0)
    original = original_segment.targets[37]
    alternate = alternate_segment.targets[37]
    deadline = original.episode_time_s - MOTOR_LEAD_S
    original_hud = capture_causal_hud(original_segment, end_s=3.20)
    alternate_hud = capture_causal_hud(alternate_segment, end_s=3.20)
    marker_first = _first_time(
        original_hud,
        lambda row: _floor_visible_with_marker(row, original_segment, 37, 9),
    )
    midspin_pair_first = _first_time(
        original_hud,
        lambda row: _midspin_pair_safe(row, original_segment, 38),
    )
    prefix = compare_visible_prefixes(_visible(original_hud), _visible(alternate_hud), deadline_s=deadline)
    first_difference = next((left["observed_at_s"] for left, right in zip(original_hud, alternate_hud, strict=True) if left["hud263"] != right["hud263"] or left["countdown"] != right["countdown"]), None)

    windows = (_hit_window(original), _hit_window(alternate))
    feasibility_predeadline = assess_shared_safe_strike(
        observed_at_s=float(prefix.last_compared_at_s),
        due_interval=DueInterval(original.episode_time_s, alternate.episode_time_s),
        acceptable_windows=windows,
        required_lead_s=MOTOR_LEAD_S,
    )
    feasibility_current = assess_shared_safe_strike(
        observed_at_s=3.19,
        due_interval=DueInterval(original.episode_time_s, alternate.episode_time_s),
        acceptable_windows=windows,
        required_lead_s=MOTOR_LEAD_S,
    )

    event37_frame = next(row for row in diagnostic["frames"] if 37 in row["new_ids"])
    decision37 = next(row for row in diagnostic["decisions"] if int(row["local_id"]) == 37)
    event37_truth = next(row for row in diagnostic["target_outcomes"] if int(row["ordinal"]) == 37)
    id38 = next(row for row in phase20c["no_fail"]["event_37_90"] if int(row["policy_local_id"]) == 38)
    auto36 = next(row for row in phase20c["no_fail"]["automatic_transitions"] if int(row["target_ordinal"]) == 36)

    keydowns = _logical_keydowns(diagnostic, end_s=5.36)
    local_ids = _local_ids(arm, diagnostic)
    base = [{**row, "input_id": f"local:{local_ids[_edge_key(row['time_s'], row['contact'])]}"} for row in keydowns]
    targets = targets_from_playable_segment(build_playable_segment(compile_adofai(chart), end_s=6.0))
    interventions = {
        "baseline": [],
        "oracle_event37_due_only": [{"input_id": "oracle:event37", "time_s": original.episode_time_s, "contact": "audit"}],
        "oracle_event36_and37_due": [
            {"input_id": "oracle:event36", "time_s": original_segment.targets[36].episode_time_s, "contact": "audit"},
            {"input_id": "oracle:event37", "time_s": original.episode_time_s, "contact": "audit"},
        ],
        "oracle_all_pre89_lead_abstentions": [
            {"input_id": f"oracle:event{ordinal}", "time_s": original_segment.targets[ordinal].episode_time_s, "contact": "audit"}
            for ordinal in (36, 37, 42, 47)
        ],
        "hud_candidate_after_virtual_nofail_auto": [{"input_id": "hud_candidate:event37", "time_s": 3.251, "contact": "audit"}],
    }
    reference = {
        name: {
            "no_fail": _run_reference(targets, base, extra, run_mode=StockRunMode.NO_FAIL),
            "normal_fail_anchored_after_event35": _run_reference(targets, base, extra, run_mode=StockRunMode.FAIL, anchored=True),
        }
        for name, extra in interventions.items()
    }

    timeline = [
        {"time_s": marker_first, "item": "future SetSpeed marker first visible (numeric multiplier absent)", "provenance": "HUD263 capture"},
        {"time_s": midspin_pair_first, "item": "Midspin pair first safely visible", "provenance": "HUD263 visible geometry"},
        {"time_s": event37_truth["due_s"], "item": "Event37 true due", "provenance": "evaluator_oracle_only"},
        {"time_s": deadline, "item": "last 44ms reservation deadline", "provenance": "evaluator calculation from true due; not Student input"},
        {"time_s": first_difference, "item": "first counterfactual HUD distinction / numeric BPM arrival", "provenance": "HUD263 capture comparison"},
        {"time_s": event37_frame["local_clock_s"], "item": "Timing local ID37 first generated and point due emitted", "provenance": "HUD Timing lifecycle"},
        {"time_s": decision37["decision_s"], "item": "Fingering decision abstains; earliest reservation if lead rule were ignored", "provenance": "policy decision history"},
        {"time_s": auto36["frame_time_s"], "item": "Event36 NoFail automatic transition", "provenance": "audited Stock reference virtual frame"},
        {"time_s": id38["keydown_at_s"], "item": "ID38 physical KeyDown", "provenance": "Motor history"},
        {"time_s": id38["processed_at_s"], "item": "ID38 processed against Event37 as VeryLate", "provenance": "Stock reference; evaluator checked"},
    ]
    timeline.sort(key=lambda row: row["time_s"])

    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdicts": ["EARLY_DUE_NONIDENTIFIABLE", "LEAD_44MS_INFEASIBLE", "ORACLE_ONLY_INTERVENTION", "INSUFFICIENT_EVIDENCE"],
        "scope": {"chart": CHART, "chart_sha256": CHART_SHA256, "event_ordinal_posthoc": 37, "production_changed": False},
        "timeline": timeline,
        "observability": {
            "future_setspeed_marker_first_visible_s": marker_first,
            "midspin_pair_first_safely_visible_s": midspin_pair_first,
            "first_information_complete_lower_bound_s": first_difference,
            "first_existing_timing_identification_s": event37_frame["local_clock_s"],
            "exact_due_reconstruction_at_first_information_frame": "unverified_and_not_needed_for_predeadline_nonidentifiability",
            "counterfactual": {
                "intervention": intervention,
                "original_due_s_posthoc": original.episode_time_s,
                "alternate_due_s_posthoc": alternate.episode_time_s,
                "due_gap_ms_posthoc": (alternate.episode_time_s - original.episode_time_s) * 1000.0,
                "deadline_44ms_s_posthoc": deadline,
                "compared_frames": prefix.compared_frames,
                "last_compared_at_s": prefix.last_compared_at_s,
                "maximum_hud_difference": prefix.maximum_feature_difference,
                "countdown_equal": prefix.cues_equal,
                "first_difference_at_s": first_difference,
                "status": prefix.status.value,
            },
        },
        "lead": {
            "required_s": MOTOR_LEAD_S,
            "existing_timing_lead_ms": (float(decision37["due_local_s"]) - float(decision37["decision_s"])) * 1000.0,
            "original_stock_counted_window_s": [windows[0].earliest_s, windows[0].latest_s],
            "alternate_stock_counted_window_s": [windows[1].earliest_s, windows[1].latest_s],
            "predeadline_shared_assessment": {"status": feasibility_predeadline.status.value, "reason": feasibility_predeadline.reason, "shared_window": None},
            "current_timing_assessment": {"status": feasibility_current.status.value, "reason": feasibility_current.reason, "shared_window": None},
            "too_early_or_wrong_world_risk": "the two Stock counted windows are disjoint; one precommitted strike cannot be valid in both visible-prefix-equivalent worlds",
        },
        "reference_interventions": reference,
        "intervention_boundary": {
            "oracle_rows_are_evaluator_only": True,
            "hud_candidate_inputs": ["ID37 point forecast at 3.190s", "declared NoFail mode", "virtual Event36 auto-transition predicted at 3.250s"],
            "hud_candidate_reservation_lead_ms": 61.0,
            "why_not_safe_control": ["live Unity frame phase is unverified", "automatic FailMiss visibility is unverified", "normal-fail mode does not advance Event36 at 3.250s", "no physical Motor replay was authorized"],
        },
        "generalization": _generalization(generalization_reports, train_dir),
        "student_boundary": {
            "oracle_use": "counterfactual truth and target ordinals used only after prefix capture",
            "production_modules_modified": [],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase16", type=Path, default=Path("reports/phase16-firsttown12-event89-causal-20261021-r2.json"))
    parser.add_argument("--phase20c", type=Path, default=Path("reports/phase20c-frame-stock-state-20261022-r1.json"))
    parser.add_argument("--train-dir", type=Path, default=Path("data/DMDOD-v140-tuf-p11-p20/Train"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {args.output}")
    reports = [Path("reports/phase16-firsttown12-motor-ab-20261020-r1.json"), Path("reports/phase16-bool5-motor-ab-20261020-r1.json")]
    result = run(args.phase16, args.phase20c, args.train_dir, reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdicts": result["verdicts"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
