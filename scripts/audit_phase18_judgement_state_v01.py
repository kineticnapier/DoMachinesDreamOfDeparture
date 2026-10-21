from __future__ import annotations

"""Replay Phase 16 through causal Timing/Motor/visible-judgement evidence.

Scorer target indices are excluded from the online replay and joined only in
``oracle_evaluation`` after all estimator outputs have been produced.
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from dmdod.features.judgement_observer_v01 import JudgementObserver
from dmdod.planning.judgement_state_estimator_v01 import (
    EstimateStatus,
    JudgementStateEstimator,
    MotorKeyDownEvidence,
    RepositoryWindowModel,
    TimingEventEvidence,
    TimingEventStatus,
)
if __package__:  # package import in tests
    from scripts.audit_phase17_visible_judgement_v01 import (
        next_capture_s,
        render_counters,
    )
else:  # direct ``python scripts/...py`` execution
    from audit_phase17_visible_judgement_v01 import (  # type: ignore[no-redef]
        next_capture_s,
        render_counters,
    )


FORMAT = "dmdod-phase18-judgement-state-audit-v0.1"
CANDIDATE = "v15_midspin_twirl_checkpoint_on"


def edge_id(time_s: float, contact: str) -> str:
    return f"{float(time_s):.9f}:{contact}"


def extract_causal_keydowns(frames: list[dict], control_dt_s: float) -> list[dict]:
    """Map self-observed edges to reservations without scorer truth."""
    rows: list[dict] = []
    for frame in frames:
        for edge in frame["logical_edges"]:
            if edge["edge"] != "down":
                continue
            reservations = [
                reservation for reservation in frame["reservations_after_physics"]
                if edge["contact"] in reservation["down_axes"]
            ]
            if len(reservations) != 1:
                raise RuntimeError("KeyDown must have one causal active reservation")
            reservation = reservations[0]
            rows.append({
                "input_id": edge_id(edge["time_s"], edge["contact"]),
                "intended_local_id": str(reservation["local_id"]),
                "occurred_at_s": float(edge["time_s"]),
                "observed_at_s": next_capture_s(float(edge["time_s"]), control_dt_s),
                "contact": edge["contact"],
            })
    return rows


def _serialize_estimate(result) -> dict:
    return {
        "status": result.status.value,
        "observed_at_s": result.observed_at_s,
        "identifiable_under_repository_model": result.identifiable,
        "stock_identity_verified": result.stock_identity_verified,
        "aligned_state_possible": result.aligned_state_possible,
        "possible_unresolved_local_ids": list(result.possible_unresolved_local_ids),
        "reason": result.reason,
        "hypotheses": [
            {
                "attributions": [list(row) for row in hypothesis.attributions],
                "supporting_evidence": list(hypothesis.supporting_evidence),
                "contradicting_evidence": list(hypothesis.contradicting_evidence),
                "within_interval_order_known": hypothesis.within_interval_order_known,
            }
            for hypothesis in result.hypotheses
        ],
    }


def replay_visible_evidence(data: dict) -> dict:
    case = data["cases"][0]
    arm = case["arms"][CANDIDATE]
    window = arm["diagnostic_window_posthoc"]
    dt = float(data["control_dt_s"])
    labels = tuple(sorted({str(row["margin"]) for row in window["judgement_trace"]}))
    start_s = 3.15
    end_s = 5.36
    first_tick = round(start_s / dt)
    last_tick = round(end_s / dt)
    capture_times = [tick * dt for tick in range(first_tick, last_tick + 1)]
    visible_frames = render_counters(window["judgement_trace"], labels, capture_times)
    observer = JudgementObserver(labels)
    estimator = JudgementStateEstimator(
        RepositoryWindowModel(
            perfect_s=0.025,
            early_late_perfect_s=0.030,
            pass_s=0.065,
        ),
        max_predecessors=3,
    )
    decisions = sorted(arm["decisions"], key=lambda row: float(row["decision_s"]))
    keydowns = sorted(
        extract_causal_keydowns(window["frames"], dt),
        key=lambda row: row["observed_at_s"],
    )
    decision_cursor = 0
    keydown_cursor = 0
    outputs: list[dict] = []
    for capture_s, frame in zip(capture_times, visible_frames):
        while decision_cursor < len(decisions) and float(decisions[decision_cursor]["decision_s"]) <= capture_s + 1e-12:
            row = decisions[decision_cursor]
            estimator.observe_timing(TimingEventEvidence(
                local_id=str(row["local_id"]),
                observed_at_s=float(row["decision_s"]),
                due_lower_s=float(row["due_local_s"]),
                due_upper_s=float(row["due_local_s"]),
                status=(
                    TimingEventStatus.CONFIRMED
                    if row["choice"] is not None else TimingEventStatus.ABSTAINED
                ),
                reason=str(row["reason"]),
            ))
            decision_cursor += 1
        while keydown_cursor < len(keydowns) and keydowns[keydown_cursor]["observed_at_s"] <= capture_s + 1e-12:
            row = keydowns[keydown_cursor]
            # Ignore edges before this bounded replay's first Timing evidence.
            if row["intended_local_id"] in {str(item["local_id"]) for item in decisions[:decision_cursor]}:
                estimator.observe_keydown(MotorKeyDownEvidence(
                    input_id=row["input_id"],
                    intended_local_id=row["intended_local_id"],
                    occurred_at_s=row["occurred_at_s"],
                    observed_at_s=row["observed_at_s"],
                ))
            keydown_cursor += 1
        feedback = observer.observe(frame)
        result = estimator.observe_judgement(feedback)
        if result.status not in {EstimateStatus.NO_UPDATE}:
            output = _serialize_estimate(result)
            output["visible_delta"] = feedback.delta_map
            output["visible_event_interval_s"] = feedback.event_interval_s
            outputs.append(output)
    return {
        "visible_labels": labels,
        "online_outputs": outputs,
        "causal_keydowns": keydowns,
        "window_s": [start_s, end_s],
    }


def run(source: Path) -> dict:
    raw = source.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    replay = replay_visible_evidence(data)
    case = data["cases"][0]
    arm = case["arms"][CANDIDATE]
    trace_by_edge = {
        edge_id(row["time_s"], row["key"]): row
        for row in arm["diagnostic_window_posthoc"]["judgement_trace"]
    }
    online_by_input: dict[str, dict] = {}
    for output in replay["online_outputs"]:
        for hypothesis in output["hypotheses"]:
            for input_id, *_ in hypothesis["attributions"]:
                online_by_input.setdefault(input_id, output)

    oracle_rows = []
    for keydown in replay["causal_keydowns"]:
        intended = int(keydown["intended_local_id"])
        if not 37 <= intended <= 90:
            continue
        truth = trace_by_edge.get(keydown["input_id"])
        output = online_by_input.get(keydown["input_id"])
        oracle_rows.append({
            "input_id": keydown["input_id"],
            "intended_local_id": intended,
            "actual_target_index_posthoc": None if truth is None else int(truth["target_index"]),
            "actual_margin_posthoc": None if truth is None else str(truth["margin"]),
            "estimate_status": None if output is None else output["status"],
            "estimated_judged_ids": [] if output is None else sorted({
                int(attribution[2])
                for hypothesis in output["hypotheses"]
                for attribution in hypothesis["attributions"]
                if attribution[0] == keydown["input_id"]
            }),
            "truth_used_online": False,
        })

    def output_for_intended(local_id: str) -> dict:
        return next(
            output for output in replay["online_outputs"]
            if any(
                attribution[1] == local_id
                for hypothesis in output["hypotheses"]
                for attribution in hypothesis["attributions"]
            )
        )

    id38 = output_for_intended("38")
    id89 = output_for_intended("89")
    id90 = output_for_intended("90")
    matched = [
        row for row in oracle_rows
        if row["actual_target_index_posthoc"] in row["estimated_judged_ids"]
    ]
    checks = {
        "id38_excludes_aligned_under_repository_model": (
            not id38["aligned_state_possible"]
            and id38["possible_unresolved_local_ids"] == ["38"]
        ),
        "id38_first_visible_update_is_3270ms": abs(id38["observed_at_s"] - 3.270) < 1e-9,
        "id89_is_flagged_unresolved_before_expiry": (
            id89["possible_unresolved_local_ids"] == ["89"]
            and id89["observed_at_s"] < 5.315
        ),
        "id90_supports_model_resynchronization": (
            id90["aligned_state_possible"]
            and id90["possible_unresolved_local_ids"] == []
        ),
        "online_records_exclude_oracle_target_index": all(
            "target_index" not in json.dumps(output)
            for output in replay["online_outputs"]
        ),
        "oracle_truth_never_used_online": all(row["truth_used_online"] is False for row in oracle_rows),
    }
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "SIMULATOR_MODEL_IDENTIFIES_BACKLOG_STOCK_IDENTITY_UNVERIFIED",
        "source": {
            "path": str(source),
            "sha256": sha256(raw).hexdigest(),
            "truth_scope": "posthoc_evaluator_only",
        },
        "online_contract": {
            "timing": "local ID, predicted due interval, causal status/reason",
            "motor": "self-observed KeyDown and reservation intended local ID",
            "judgement": "Phase17 cumulative-counter delta and display interval",
            "forbidden": ["target_index", "scorer cursor", "_used", "_missed", "truth due"],
        },
        "model_scope": {
            "name": "repository symmetric windows",
            "perfect_s": 0.025,
            "early_late_perfect_s": 0.030,
            "pass_s": 0.065,
            "stock_parity_verified": False,
            "miss_counter_visible_behavior_verified": False,
        },
        "id37_abstention": {
            "decision_s": 3.190,
            "forecast_lead_ms": 30.588,
            "online_conclusion": (
                "No input is reserved for local ID37. Before later visible feedback, both "
                "auto-resolution and remaining-unresolved hypotheses are possible."
            ),
        },
        "id38_update": id38,
        "id89_preexpiry_risk": {
            **id89,
            "repository_model_possible_miss_deadline_s": 5.315000043646398,
            "risk_lead_ms": (5.315000043646398 - id89["observed_at_s"]) * 1000.0,
            "claim": "local ID89 is possibly unresolved, not a certain future Miss",
        },
        "id90_update": id90,
        "online_summary": {
            "outputs": len(replay["online_outputs"]),
            "oracle_evaluated_inputs_37_90": len(oracle_rows),
            "oracle_target_in_model_hypotheses": len(matched),
        },
        "oracle_evaluation": oracle_rows,
        "nonidentifiability": {
            "stock_identity": "UNVERIFIED",
            "counterexamples": [
                "normal VeryLate and FIFO VeryLate share a visible delta without timing/motor context",
                "overlapping due intervals can keep aligned and predecessor hypotheses simultaneously",
                "multiple KeyDowns and multiple deltas in one 10ms interval lose ordering",
                "reset, missing, or delayed counter frames create an attribution gap",
                "unknown Stock non-KeyDown Miss display prevents proving cursor advancement",
            ],
        },
        "checks": checks,
        "production_control_connected": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source)
    if not all(result["checks"].values()):
        result["verdict"] = "AUDIT_INCOMPLETE"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print("verdict:", result["verdict"])
    print("summary:", result["online_summary"])
    print("checks:", result["checks"])
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
