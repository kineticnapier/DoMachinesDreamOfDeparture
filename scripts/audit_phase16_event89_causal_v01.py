from __future__ import annotations

"""Posthoc-only audit of Phase 16 reservation-to-scorer attribution.

Target ordinals, target times, judgement traces, and scorer windows are used
only after both policy/physics replays have completed.  Nothing from this
module is imported by Timing, Fingering, or Motor policy code.
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_chart_16k import CoupledContactRealChartEnv


FORMAT = "dmdod-phase16-event89-causal-audit-v0.1"
BASELINE = "v15_recovery_off"
CANDIDATE = "v15_midspin_twirl_checkpoint_on"
CHART = "01645 - First town of this Journey remix - Irin.adofai"
CHART_SHA256 = "4a71411d454d3b10a10cd98165a2e9cce99e4b14c310d1e6847836fdc68fc0e1"


def _edge_key(time_s: float, contact: str) -> tuple[float, str]:
    return round(float(time_s), 9), str(contact)


def attribute_keydowns(arm: dict) -> list[dict]:
    """Join policy reservations to scorer trace without using truth in policy."""
    window = arm["diagnostic_window_posthoc"]
    judgements = {
        _edge_key(row["time_s"], row["key"]): row
        for row in window["judgement_trace"]
    }
    outcomes = arm["target_outcomes_posthoc"]
    decisions = {row["local_id"]: row for row in arm["decisions"]}
    rows: list[dict] = []
    for frame in window["frames"]:
        for edge in frame["logical_edges"]:
            if edge["edge"] != "down":
                continue
            if not window["start_s"] <= edge["time_s"] <= window["end_s"]:
                continue
            reservations = [
                row for row in frame["reservations_after_physics"]
                if edge["contact"] in row["axes"]
                and edge["contact"] in row["down_axes"]
            ]
            if len(reservations) != 1:
                raise RuntimeError(
                    "logical KeyDown must map to exactly one active reservation: "
                    f"time={edge['time_s']} contact={edge['contact']} "
                    f"local_ids={[row['local_id'] for row in reservations]}"
                )
            reservation = reservations[0]
            local_id = reservation["local_id"]
            decision = decisions.get(local_id)
            if decision is None:
                raise RuntimeError("KeyDown reservation has no Fingering decision")
            distances = sorted(
                (abs(row["due_s"] - decision["due_local_s"]), row["ordinal"])
                for row in outcomes
            )
            # Countdown start is interval-valued and differs by ~4.4 ms here;
            # the next truth event is much farther away. This is evaluator-only
            # identity attribution, never a policy matching threshold.
            if not distances or distances[0][0] > 0.010:
                raise RuntimeError(
                    "posthoc intended target due-time has no unique match: "
                    f"local_id={local_id} due={decision['due_local_s']} "
                    f"nearest={distances[0] if distances else None}"
                )
            if len(distances) > 1 and distances[1][0] == distances[0][0]:
                raise RuntimeError("posthoc intended target due-time match is ambiguous")
            judgement = judgements.get(_edge_key(edge["time_s"], edge["contact"]))
            if judgement is None:
                raise RuntimeError("logical KeyDown is absent from scorer trace")
            intended = distances[0][1]
            actual = judgement["target_index"]
            rows.append({
                "time_s": edge["time_s"],
                "contact": edge["contact"],
                "local_id": local_id,
                "intended_ordinal_posthoc": intended,
                "judged_ordinal_posthoc": actual,
                "ordinal_delta": intended - actual,
                "due_local_s": decision["due_local_s"],
                "judgement": judgement["margin"],
            })
    return rows


def _target(env: CoupledContactRealChartEnv, ordinal: int) -> dict:
    target = env.segment.targets[ordinal]
    pass_s = env._windows(target).pass_s  # evaluator-only audit
    return {
        "ordinal": ordinal,
        "due_s": target.episode_time_s,
        "late_pass_s": pass_s,
        "expires_s": target.episode_time_s + pass_s,
    }


def run(source: Path, train_dir: Path) -> dict:
    raw = source.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    case = data["cases"][0]
    if case["chart"] != CHART or case["chart_sha256"] != CHART_SHA256:
        raise ValueError("Phase 16 chart identity changed")
    if data.get("diagnostic_window_s") != [0.0, 6.0]:
        raise ValueError("audit requires the fixed 0-6s posthoc window")
    chart_path = train_dir / CHART
    if sha256(chart_path.read_bytes()).hexdigest() != CHART_SHA256:
        raise ValueError("fixed Train chart missing or changed")
    segment = build_playable_segment(
        compile_adofai(load_adofai(chart_path)), end_s=case["scope_s"]
    )
    env = CoupledContactRealChartEnv(segment)
    arms = case["arms"]
    candidate = arms[CANDIDATE]
    mappings = attribute_keydowns(candidate)
    mismatches = [row for row in mappings if row["ordinal_delta"] != 0]
    first = mismatches[0]
    last = mismatches[-1]
    resync = next(
        row for row in mappings
        if row["time_s"] > last["time_s"] and row["ordinal_delta"] == 0
    )
    decisions = {row["local_id"]: row for row in candidate["decisions"]}
    predecessor = decisions[first["intended_ordinal_posthoc"] - 1]
    physical = candidate["diagnostic_window_posthoc"]["physical_keydowns"]
    logical = candidate["diagnostic_window_posthoc"]["logical_keydowns"]
    edge_parity = [
        (row["time_s"], row["axis"]) for row in physical
    ] == [
        (row["time_s"], row["contact"]) for row in logical
    ]
    target_rows = {
        ordinal: _target(env, ordinal)
        for ordinal in (36, 37, 38, 88, 89, 90)
    }
    checks = {
        "source_ab_still_has_only_lost_hit_89": (
            case["difference"]["lost_hit_ordinals_posthoc"] == [89]
        ),
        "first_mismatch_is_id38_to_event37": (
            first["local_id"] == 38
            and first["intended_ordinal_posthoc"] == 38
            and first["judged_ordinal_posthoc"] == 37
        ),
        "event37_was_insufficient_lead": (
            predecessor["local_id"] == 37
            and predecessor["reason"] == "insufficient_observable_lead"
            and predecessor["choice"] is None
        ),
        "id38_arrived_after_36_expired_before_37_expired": (
            target_rows[36]["expires_s"] < first["time_s"]
            < target_rows[37]["expires_s"]
        ),
        "id89_was_fifo_consumed_by_event88": (
            last["local_id"] == 89
            and last["intended_ordinal_posthoc"] == 89
            and last["judged_ordinal_posthoc"] == 88
        ),
        "id90_arrived_after_event89_expired_and_resynced": (
            target_rows[89]["expires_s"] < resync["time_s"]
            and resync["local_id"] == 90
            and resync["judged_ordinal_posthoc"] == 90
        ),
        "physical_logical_keydown_parity": edge_parity,
        "lifecycle_contract_violations_zero": not candidate["lifecycle_contract_violations"],
        "reservation_contract_violations_zero": not candidate["reservation_contract_violations"],
    }
    event_80_95 = [
        row for row in candidate["target_outcomes_posthoc"]
        if 80 <= row["ordinal"] <= 95
    ]
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "CAUSE_CONFIRMED_SAFE_FIX_BLOCKED",
        "source": {
            "path": str(source),
            "sha256": sha256(raw).hexdigest(),
            "chart": CHART,
            "chart_sha256": CHART_SHA256,
            "truth_scope": "posthoc_evaluator_only",
        },
        "checks": checks,
        "first_attribution_mismatch": first,
        "causal_predecessor": {
            **predecessor,
            "forecast_lead_s": predecessor["due_local_s"] - predecessor["decision_s"],
            "required_motor_lead_s": 0.044,
        },
        "mismatch_span": {
            "first": first,
            "last": last,
            "resynchronized": resync,
            "mismatched_keydown_count": len(mismatches),
        },
        "scorer_windows_posthoc": target_rows,
        "event_80_95_candidate": event_80_95,
        "event_80_95_attribution": [
            row for row in mappings
            if 80 <= row["intended_ordinal_posthoc"] <= 95
        ],
        "responsibility": {
            "timing": {
                "status": "CONFIRMED_CAUSAL_LIMITATION_NOT_WRONG_DUE_OR_ID",
                "evidence": "ID37 due was accurate but first actionable lead was 30.588ms, below frozen 44ms",
            },
            "fingering": {
                "status": "NEGATED_AS_DIRECT_CAUSE",
                "evidence": "assigned IDs had one stable choice; ID37 was never passed as assignable",
            },
            "motor": {
                "status": "NEGATED_AS_DIRECT_CAUSE",
                "evidence": "physical/logical edges agree and reservation violations are zero",
            },
            "scorer": {
                "status": "CONFIRMED_ATTRIBUTION_MECHANISM_STOCK_PARITY_UNVERIFIED",
                "evidence": "repository scorer expires old targets then consumes the oldest still-unresolved target",
            },
            "combined": {
                "status": "CONFIRMED",
                "evidence": "late-observed ID37 abstention plus FIFO late window creates a one-event backlog through ID89",
            },
        },
        "implementation": {
            "production_change": False,
            "reason": (
                "Recovery would require a late best-effort input, scorer-aware delay, "
                "or scorer reassignment. The first changes the frozen 44ms safety "
                "contract; the second requires verified Stock late-window semantics; "
                "the third would alter evaluation. None is currently justified."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--train-dir", type=Path,
        default=Path("data/DMDOD-v140-tuf-p11-p20/Train"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source, args.train_dir)
    if not all(result["checks"].values()):
        result["verdict"] = "DIAGNOSIS_INCOMPLETE"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print("verdict:", result["verdict"])
    print("first mismatch:", result["first_attribution_mismatch"])
    print("mismatch span:", result["mismatch_span"])
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
