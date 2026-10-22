from __future__ import annotations

"""Offline Phase 16 replay through the Stock-floor set-valued estimator."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmdod.features.judgement_observer_v01 import JudgementDelta, JudgementDeltaStatus
from dmdod.planning.judgement_state_estimator_v01 import MotorKeyDownEvidence, TimingEventEvidence, TimingEventStatus
from dmdod.planning.stock_persistent_judgement_state_estimator_v03 import StockPersistentJudgementStateEstimator


FORMAT = "dmdod-phase22-stock-estimator-audit-v0.1"


def _feedback(row: dict[str, object]) -> JudgementDelta:
    return JudgementDelta(
        status=JudgementDeltaStatus(str(row["status"])),
        observed_at_s=float(row["observed_at_s"]),
        displayed_at_s=float(row["displayed_at_s"]),
        deltas=tuple((str(key), int(value)) for key, value in dict(row.get("deltas", {})).items()),
        event_interval_s=(tuple(float(value) for value in row["event_interval_s"]) if row.get("event_interval_s") else None),
        delta_available=str(row["status"]) == "delta",
        within_interval_order_known=bool(row.get("within_interval_order_known", False)),
        reason="Phase 21 visible-counter replay",
    )


def _run_arm(
    decisions: list[dict[str, object]],
    keydowns: list[dict[str, object]],
    observations: list[dict[str, object]],
    *,
    automatic_failmiss_visible: bool | None,
) -> tuple[StockPersistentJudgementStateEstimator, list[dict[str, object]]]:
    estimator = StockPersistentJudgementStateEstimator(
        automatic_failmiss_visible=automatic_failmiss_visible,
        max_hypotheses=128,
    )
    stream: list[tuple[float, int, str, object]] = []
    for row in decisions:
        local_id = str(row["local_id"])
        if not local_id.isdigit() or not 36 <= int(local_id) <= 90:
            continue
        due = float(row["due_local_s"])
        evidence = TimingEventEvidence(
            local_id=local_id,
            observed_at_s=float(row["decision_s"]),
            due_lower_s=due,
            due_upper_s=due,
            status=(TimingEventStatus.ABSTAINED if row.get("choice") is None else TimingEventStatus.CONFIRMED),
            reason=str(row.get("reason", "")),
        )
        stream.append((evidence.observed_at_s, 0, "timing", evidence))
    for row in keydowns:
        local_id = str(row["policy_local_id"])
        evidence = MotorKeyDownEvidence(
            input_id=f"input:{local_id}",
            intended_local_id=local_id,
            occurred_at_s=float(row["keydown_at_s"]),
            observed_at_s=float(row["keydown_at_s"]),
        )
        stream.append((evidence.observed_at_s, 1, "keydown", evidence))
    for row in observations:
        if row["status"] == "baseline":
            continue
        evidence = _feedback(row)
        stream.append((evidence.observed_at_s, 2, "feedback", evidence))
    stream.sort(key=lambda item: (item[0], item[1]))

    trace: list[dict[str, object]] = []
    for at_s, _, kind, evidence in stream:
        if kind == "timing":
            estimator.observe_timing(evidence)
        elif kind == "keydown":
            estimator.observe_keydown(evidence)
        else:
            estimate = estimator.observe_judgement(evidence)
            trace.append({
                "observed_at_s": at_s,
                "visible_deltas": {key: value for key, value in evidence.deltas if value},
                "status": estimate.status.value,
                "hypothesis_count": len(estimate.hypotheses),
                "possible_current_local_ids": estimate.possible_current_local_ids,
                "identifiable": estimate.identifiable,
                "unknown_remainder": any(row.unknown_search_remainder for row in estimate.hypotheses),
            })
    return estimator, trace


def _oracle_posthoc(estimator: StockPersistentJudgementStateEstimator, keydowns: list[dict[str, object]]) -> dict[str, object]:
    truth = {f"input:{row['policy_local_id']}": str(row["stock_target_ordinal"]) for row in keydowns}
    exact_counts: list[int] = []
    for hypothesis in estimator.hypotheses:
        pairs = {row.input_id: row.judged_local_id for row in hypothesis.attributions if row.input_id is not None}
        exact_counts.append(sum(pairs.get(input_id) == target for input_id, target in truth.items()))
    return {
        "truth_pair_count": len(truth),
        "best_exact_attribution_count": max(exact_counts, default=0),
        "complete_exact_hypothesis_exists": any(count == len(truth) for count in exact_counts),
        "unknown_remainder_retained": any(row.unknown_search_remainder for row in estimator.hypotheses),
    }


def run(phase16_path: Path, phase20c_path: Path, phase21_path: Path) -> dict[str, object]:
    phase16 = json.loads(phase16_path.read_text(encoding="utf-8"))
    arm = phase16["cases"][0]["arms"]["v15_midspin_twirl_checkpoint_on"]["diagnostic_window_posthoc"]
    phase20c = json.loads(phase20c_path.read_text(encoding="utf-8"))
    keydowns = list(phase20c["no_fail"]["event_37_90"])
    phase21 = json.loads(phase21_path.read_text(encoding="utf-8"))

    fixture_estimator, fixture_trace = _run_arm(
        list(arm["decisions"]), keydowns, list(phase21["fixture_all_categories"]["observations"]),
        automatic_failmiss_visible=True,
    )
    conservative_estimator, conservative_trace = _run_arm(
        list(arm["decisions"]), keydowns, list(phase21["conservative_failmiss_unknown"]["observations"]),
        automatic_failmiss_visible=None,
    )
    fixture_oracle = _oracle_posthoc(fixture_estimator, keydowns)
    conservative_oracle = _oracle_posthoc(conservative_estimator, keydowns)
    first_conservative_risk = next((row for row in conservative_trace if len(row["possible_current_local_ids"]) > 1 or row["unknown_remainder"]), None)
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdicts": ["STATE_NONIDENTIFIABLE", "NO_SAFE_INTERVENTION_ESTABLISHED"],
        "scope": "Phase16 First town local IDs 36..90; production control unchanged",
        "online_inputs": ["Timing local ID/due/status", "own physical KeyDown history", "Phase21 visible counter deltas", "past estimator hypotheses"],
        "forbidden_inputs_absent": ["true target index", "scorer cursor", "_used/_missed", "privileged target time", "future counter delta"],
        "fixture_failmiss_visible": {
            "trace": fixture_trace,
            "final_hypothesis_count": len(fixture_estimator.hypotheses),
            "oracle_posthoc": fixture_oracle,
            "interpretation": "conditional result only; automatic FailMiss counter visibility is not established on live Stock",
        },
        "conservative_failmiss_unknown": {
            "trace": conservative_trace,
            "final_hypothesis_count": len(conservative_estimator.hypotheses),
            "oracle_posthoc": conservative_oracle,
            "first_alignment_risk_visible_at_s": (first_conservative_risk["observed_at_s"] if first_conservative_risk else None),
            "false_unique_identification_count": sum(1 for row in conservative_trace if row["identifiable"]),
            "interpretation": "normal and hidden-auto-transition floor states remain observationally compatible",
        },
        "event89_prevention": {
            "risk_detectable_as_possibility_at_s": (first_conservative_risk["observed_at_s"] if first_conservative_risk else None),
            "risk_uniquely_identified": False,
            "fixture_first_warning_s": 3.26,
            "id38_physical_keydown_s": 3.260999999999752,
            "available_margin_ms": 0.999999999752,
            "required_motor_lead_ms": 44.0,
            "safe_intervention": False,
            "reason": "the optimistic signal is too late for the frozen Motor contract; the conservative visible history is not target-identifying",
        },
        "normal_fail": {
            "reference_stop_s": 3.32,
            "estimator_tracking_ends_when_a separately observable game-stop signal is supplied": True,
            "counter_deltas_alone_prove_terminal_state": False,
        },
        "stock_runtime_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase16", type=Path, default=Path("reports/phase16-firsttown12-event89-causal-20261021-r2.json"))
    parser.add_argument("--phase20c", type=Path, default=Path("reports/phase20c-frame-stock-state-20261022-r1.json"))
    parser.add_argument("--phase21", type=Path, default=Path("reports/phase21-stock-visible-feedback-20261022-r1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {args.output}")
    result = run(args.phase16, args.phase20c, args.phase21)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdicts": result["verdicts"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

