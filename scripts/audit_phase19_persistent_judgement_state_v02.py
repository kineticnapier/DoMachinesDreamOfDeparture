from __future__ import annotations

"""Replay Phase 16 with the Phase 19 persistent judgement-state estimator.

Privileged scorer target indices are joined only after ``online_replay`` has
returned.  The online path consumes Timing local IDs/due intervals, causal
Motor reservations and KeyDowns, visible counter deltas, and prior estimator
state.
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from dmdod.features.judgement_observer_v01 import JudgementDeltaStatus, JudgementObserver
from dmdod.planning.judgement_state_estimator_v01 import (
    MotorKeyDownEvidence,
    RepositoryWindowModel,
    TimingEventEvidence,
    TimingEventStatus,
)
from dmdod.planning.persistent_judgement_state_estimator_v02 import (
    AllocationRule,
    MotorReservationEvidence,
    PersistentJudgementStateEstimator,
    ReservationStatus,
)

if __package__:
    from scripts.audit_phase17_visible_judgement_v01 import next_capture_s, render_counters
    from scripts.audit_phase18_judgement_state_v01 import CANDIDATE, extract_causal_keydowns
else:
    from audit_phase17_visible_judgement_v01 import next_capture_s, render_counters
    from audit_phase18_judgement_state_v01 import CANDIDATE, extract_causal_keydowns


FORMAT = "dmdod-phase19-persistent-judgement-state-audit-v0.2"


def _serialize_hypothesis(row) -> dict:
    return {
        "hypothesis_id": row.hypothesis_id,
        "attributions": [
            {
                "input_id": item.input_id,
                "intended_local_id": item.intended_local_id,
                "judged_local_id": item.judged_local_id,
                "label": item.label,
                "visible_interval_s": list(item.visible_interval_s),
                "resolves_target": item.resolves_target,
                "ambiguous_order": item.ambiguous_order,
            }
            for item in row.attributions
        ],
        "resolutions": [
            {
                "local_id": item.local_id,
                "kind": item.kind,
                "evidence_at_s": item.evidence_at_s,
                "input_id": item.input_id,
            }
            for item in row.resolutions
        ],
        "pending_input_ids": list(row.pending_input_ids),
        "possible_unresolved_local_ids": list(row.possible_unresolved_local_ids),
        "automatic_miss_possible": list(row.automatic_miss_possible),
        "observation_consistent": row.observation_consistent,
        "observation_gaps": list(row.observation_gaps),
        "model_assumptions": list(row.model_assumptions),
        "unknown_search_remainder": row.unknown_search_remainder,
    }


def _serialize_estimate(result, *, include_history: bool = True) -> dict:
    rows = []
    for hypothesis in result.hypotheses:
        serialized = _serialize_hypothesis(hypothesis)
        if not include_history:
            serialized["attributions"] = serialized["attributions"][-2:]
            serialized["resolutions"] = serialized["resolutions"][-3:]
        rows.append(serialized)
    return {
        "status": result.status.value,
        "observed_at_s": result.observed_at_s,
        "search_complete": result.search_complete,
        "identifiable_under_model": result.identifiable,
        "stock_identity_verified": result.stock_identity_verified,
        "reason": result.reason,
        "hypotheses": rows,
    }


def online_replay(data: dict, *, allocation_rule: AllocationRule) -> dict:
    case = data["cases"][0]
    arm = case["arms"][CANDIDATE]
    window = arm["diagnostic_window_posthoc"]
    dt = float(data["control_dt_s"])
    labels = tuple(sorted({str(row["margin"]) for row in window["judgement_trace"]}))
    start_s, end_s = 3.15, 5.36
    capture_times = [
        tick * dt
        for tick in range(round(start_s / dt), round(end_s / dt) + 1)
    ]
    observer = JudgementObserver(labels)
    visible_frames = render_counters(window["judgement_trace"], labels, capture_times)
    estimator = PersistentJudgementStateEstimator(
        RepositoryWindowModel(0.025, 0.030, 0.065),
        allocation_rule=allocation_rule,
        max_predecessors=3,
        max_hypotheses=128,
        automatic_miss_visible=None,
    )
    decisions = sorted(arm["decisions"], key=lambda row: float(row["decision_s"]))
    keydowns = sorted(
        [
            row for row in extract_causal_keydowns(window["frames"], dt)
            if float(row["occurred_at_s"]) >= start_s - 1e-12
        ],
        key=lambda row: (row["observed_at_s"], row["occurred_at_s"]),
    )
    decision_cursor = 0
    keydown_cursor = 0
    observed_ids: set[str] = set()
    updates: list[dict] = []
    milestones: dict[str, dict] = {}

    for capture_s, frame in zip(capture_times, visible_frames):
        evidence_batch: list[tuple[float, int, str, dict]] = []
        while (
            decision_cursor < len(decisions)
            and float(decisions[decision_cursor]["decision_s"]) <= capture_s + 1e-12
        ):
            row = decisions[decision_cursor]
            evidence_batch.append((float(row["decision_s"]), 0, "timing", row))
            decision_cursor += 1

        while (
            keydown_cursor < len(keydowns)
            and float(keydowns[keydown_cursor]["observed_at_s"]) <= capture_s + 1e-12
        ):
            row = keydowns[keydown_cursor]
            evidence_batch.append((float(row["observed_at_s"]), 1, "keydown", row))
            keydown_cursor += 1

        for _, _, evidence_kind, row in sorted(
            evidence_batch, key=lambda item: (item[0], item[1])
        ):
            if evidence_kind == "timing":
                local_id = str(row["local_id"])
                evidence = TimingEventEvidence(
                    local_id=local_id,
                    observed_at_s=float(row["decision_s"]),
                    due_lower_s=float(row["due_local_s"]),
                    due_upper_s=float(row["due_local_s"]),
                    status=(
                        TimingEventStatus.CONFIRMED
                        if row["choice"] is not None else TimingEventStatus.ABSTAINED
                    ),
                    reason=str(row["reason"]),
                )
                estimator.observe_timing(evidence)
                observed_ids.add(local_id)
                if row["choice"] is not None:
                    estimator.observe_reservation(MotorReservationEvidence(
                        reservation_id=f"timing-{local_id}",
                        local_id=local_id,
                        observed_at_s=float(row["decision_s"]),
                        status=ReservationStatus.RESERVED,
                        reason="Phase16 Motor reservation inferred from own decision record",
                    ))
                if local_id == "37":
                    milestones["id37_abstention"] = {
                        "decision_s": float(row["decision_s"]),
                        "choice": row["choice"],
                        "reason": row["reason"],
                        "state": _serialize_hypothesis(estimator.hypotheses[0]),
                    }
                continue

            if row["intended_local_id"] in observed_ids:
                estimator.observe_keydown(MotorKeyDownEvidence(
                    input_id=row["input_id"],
                    intended_local_id=row["intended_local_id"],
                    occurred_at_s=float(row["occurred_at_s"]),
                    observed_at_s=float(row["observed_at_s"]),
                ))

        estimator.advance_time(capture_s)
        feedback = observer.observe(frame)
        estimate = estimator.observe_judgement(feedback)
        if feedback.status is JudgementDeltaStatus.DELTA:
            serialized = _serialize_estimate(estimate, include_history=False)
            serialized["visible_delta"] = feedback.delta_map
            serialized["visible_event_interval_s"] = list(feedback.event_interval_s or ())
            updates.append(serialized)
            for hypothesis in estimate.hypotheses:
                for attribution in hypothesis.attributions[-2:]:
                    if attribution.intended_local_id in {"38", "89", "90"}:
                        milestones.setdefault(f"id{attribution.intended_local_id}_update", (
                            _serialize_estimate(estimate)
                        ))

    final = estimator._estimate(  # research snapshot; no mutation
        end_s,
        default_status=updates and estimate.status or estimate.status,
        reason="end of bounded replay",
    )
    return {
        "allocation_rule": allocation_rule.value,
        "window_s": [start_s, end_s],
        "visible_labels": list(labels),
        "visible_updates": updates,
        "milestones": milestones,
        "final_state": _serialize_estimate(final),
        "causal_keydowns": keydowns,
    }


def _oracle_evaluate(data: dict, replay: dict) -> dict:
    arm = data["cases"][0]["arms"][CANDIDATE]
    trace = {
        f'{float(row["time_s"]):.9f}:{row["key"]}': row
        for row in arm["diagnostic_window_posthoc"]["judgement_trace"]
    }
    final_hypotheses = replay["final_state"]["hypotheses"]
    rows = []
    for keydown in replay["causal_keydowns"]:
        intended = int(keydown["intended_local_id"])
        if not 37 <= intended <= 90:
            continue
        truth = trace.get(keydown["input_id"])
        estimated = sorted({
            int(item["judged_local_id"])
            for hypothesis in final_hypotheses
            for item in hypothesis["attributions"]
            if item["input_id"] == keydown["input_id"]
            and item["judged_local_id"] is not None
        })
        actual = None if truth is None else int(truth["target_index"])
        rows.append({
            "input_id": keydown["input_id"],
            "intended_local_id": intended,
            "estimated_judged_local_ids": estimated,
            "actual_target_index_posthoc": actual,
            "actual_margin_posthoc": None if truth is None else str(truth["margin"]),
            "truth_in_hypothesis_set": actual in estimated,
            "truth_used_online": False,
        })
    return {
        "rows": rows,
        "evaluated": len(rows),
        "truth_retained": sum(row["truth_in_hypothesis_set"] for row in rows),
        "all_truth_retained": all(row["truth_in_hypothesis_set"] for row in rows),
    }


def run(source: Path) -> dict:
    raw = source.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    fifo = online_replay(data, allocation_rule=AllocationRule.REPOSITORY_FIFO)
    non_fifo = online_replay(data, allocation_rule=AllocationRule.COMPATIBLE_ANY)
    oracle = _oracle_evaluate(data, fifo)

    def milestone(name: str) -> dict:
        return fifo["milestones"].get(name, {})

    id38 = milestone("id38_update")
    id89 = milestone("id89_update")
    id90 = milestone("id90_update")
    id38_targets = {
        item["judged_local_id"]
        for hypothesis in id38.get("hypotheses", [])
        for item in hypothesis["attributions"]
        if item["intended_local_id"] == "38"
    }
    id89_targets = {
        item["judged_local_id"]
        for hypothesis in id89.get("hypotheses", [])
        for item in hypothesis["attributions"]
        if item["intended_local_id"] == "89"
    }
    id90_targets = {
        item["judged_local_id"]
        for hypothesis in id90.get("hypotheses", [])
        for item in hypothesis["attributions"]
        if item["intended_local_id"] == "90"
    }
    final_histories = fifo["final_state"]["hypotheses"]
    fifo_history = [
        item
        for hypothesis in final_histories
        if not hypothesis["unknown_search_remainder"]
        for item in hypothesis["attributions"]
    ]
    offset_persisted = all(
        int(item["judged_local_id"]) == int(item["intended_local_id"]) - 1
        for item in fifo_history
        if 38 <= int(item["intended_local_id"]) <= 89
    )
    checks = {
        "id37_abstention_retains_unresolved_state": "37" in fifo["milestones"]
        ["id37_abstention"]["state"]["possible_unresolved_local_ids"],
        "id38_transitions_to_37": id38_targets == {"37"},
        "offset_persists_38_through_89": offset_persisted,
        "id89_flags_unresolved_risk": id89_targets == {"88"} and any(
            "89" in hypothesis["possible_unresolved_local_ids"]
            for hypothesis in id89.get("hypotheses", [])
        ),
        "id90_resynchronizes_after_auto_resolution": id90_targets == {"90"},
        "oracle_truth_retained": oracle["all_truth_retained"],
        "no_false_identification_after_truncation": all(
            not update["identifiable_under_model"]
            for update in fifo["visible_updates"]
            if not update["search_complete"]
        ),
        "online_payload_excludes_forbidden_oracle_fields": all(
            token not in json.dumps(fifo["visible_updates"])
            for token in ("target_index", "scorer_cursor", "_used", "_missed", "truth_time_s")
        ),
    }
    verdict = (
        "REPOSITORY_FIFO_PERSISTENT_STATE_TRACKED_STOCK_PARITY_PARTIAL"
        if all(checks.values()) else "AUDIT_INCOMPLETE"
    )
    non_fifo_summary = {
        "allocation_rule": non_fifo["allocation_rule"],
        "window_s": non_fifo["window_s"],
        "visible_update_count": len(non_fifo["visible_updates"]),
        "ever_truncated": any(
            not update["search_complete"] for update in non_fifo["visible_updates"]
        ),
        "final_hypothesis_count": len(non_fifo["final_state"]["hypotheses"]),
        "final_state": {
            **non_fifo["final_state"],
            "hypotheses": [
                {
                    **hypothesis,
                    "attributions": hypothesis["attributions"][-2:],
                    "resolutions": hypothesis["resolutions"][-3:],
                }
                for hypothesis in non_fifo["final_state"]["hypotheses"]
            ],
        },
        "claim": (
            "independent countermodel only; omitted per-interval histories are "
            "not used for the repository FIFO verdict"
        ),
    }
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "source": {
            "path": str(source),
            "sha256": sha256(raw).hexdigest(),
            "truth_scope": "posthoc_evaluator_only",
        },
        "online_contract": {
            "allowed": [
                "Timing local ID and predicted due interval/status history",
                "own Motor reservation/cancellation and KeyDown",
                "Phase17 visible judgement deltas",
                "prior estimator hypotheses",
                "monotonic elapsed time",
            ],
            "forbidden": ["target_index", "scorer cursor", "_used", "_missed", "truth due"],
        },
        "model_scope": {
            "repository_fifo": "independent analysis, not assumed Stock-equivalent",
            "compatible_any": "non-FIFO countermodel analyzed independently",
            "stock_parity": "PARTIAL_DLL_VERIFICATION_RUNTIME_UNVERIFIED",
        },
        "repository_fifo_replay": fifo,
        "non_fifo_countermodel_replay": non_fifo_summary,
        "oracle_evaluation": oracle,
        "phase18_comparison": {
            "phase18": "reconstructed each visible interval with bounded predecessor search",
            "phase19": "transitions only surviving cumulative histories and retains UNKNOWN on truncation/gaps",
            "precision_priority": "never convert incomplete search into MODEL_IDENTIFIED",
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
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("verdict:", result["verdict"])
    print("oracle:", {
        "evaluated": result["oracle_evaluation"]["evaluated"],
        "truth_retained": result["oracle_evaluation"]["truth_retained"],
    })
    print("checks:", result["checks"])
    return 0 if result["verdict"] != "AUDIT_INCOMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
