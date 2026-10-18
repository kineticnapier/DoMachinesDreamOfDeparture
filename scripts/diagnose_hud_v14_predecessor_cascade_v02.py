from __future__ import annotations

"""Saved-trace predecessor inspection, without Motor or Timing reruns.

Inspect the *preceding* target(s) that absorbed a missed target's timely
KeyDown. Uses the exact scorer replay as an integrity gate, and labels
selected-axis KeyDown associations as time-nearest (not proof of causation).
No changes to chart, policy, model, physical parameters or existing files.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_hud_v14_failed_cases_v02 import ARMS, CASES, get_case
from replay_hud_v14_scorer_routing_v02 import replay_scorer
from run_hud_v14_fingering_motor_integration_v02 import DEFAULT_TRAIN_CHART


def _nearby(xs: list[dict], due_s: float, window_s: float = .100) -> list[dict]:
    return sorted(
        (e for e in xs if abs(float(e["time_s"]) - due_s) <= window_s),
        key=lambda e: (abs(float(e["time_s"]) - due_s), e["time_s"]),
    )


def target_view(record: dict, scoring: dict, ordinal: int) -> dict:
    """A fully posthoc comparison of an intended target and scorer routing."""
    truth = record["target_outcomes_posthoc"]
    if type(ordinal) is not int or not 0 <= ordinal < len(truth):
        raise ValueError("invalid target ordinal")
    target = truth[ordinal]
    due = float(target["due_s"])
    token = min(
        (x for x in record["first_seen_predictions"]
         if abs(x["due_local_s"] - due) <= .020),
        key=lambda x: (abs(x["due_local_s"] - due), x["local_id"]),
        default=None,
    )
    decision = next((d for d in record["decisions"]
                     if token is not None and d["local_id"] == token["local_id"]),
                    None)
    axes = tuple(
        part.split("/", 1)[0]
        for part in decision["choice"].split(" + ")
    ) if decision is not None and decision["choice"] else ()
    physical_downs = []
    for axis in axes:
        downs = _nearby(
            [x for x in record["physical_switch_trace"]
             if x["axis"] == axis and x["edge"] == "down"], due
        )
        physical_downs.append({
            "axis": axis,
            "closest_down": (
                {"time_s": downs[0]["time_s"],
                 "offset_ms": round((downs[0]["time_s"] - due) * 1000, 3)}
                if downs else None
            ),
            "nearest_is_only_temporal_association": True,
        })
    routed = [
        {
            "physical_key": x["key"],
            "time_s": x["time_s"],
            "offset_ms": round((x["time_s"] - due) * 1000, 3),
            "consumed_ordinal": x["consumed_target_ordinal"],
            "assigned_error_ms": x["assigned_error_ms"],
            "margin": x["margin"],
        }
        for x in _nearby(
            scoring["logical_down_target_routing"], due, .070
        )
    ]
    return {
        "ordinal": ordinal,
        "due_s": due,
        "hit": target["hit"],
        "miss": target["miss"],
        "forecast": (
            {"local_id": token["local_id"],
             "timing_error_ms": round((token["due_local_s"] - due)*1000, 3),
             "first_seen_lead_ms": round(
                 (token["due_local_s"] - token["first_seen_s"])*1000, 3)}
            if token is not None else None
        ),
        "decision": (
            {"choice": decision["choice"], "reason": decision["reason"],
             "at_step": decision["step"],
             "extra_lead_ms": round(decision["extra_lead_s"] * 1000, 3)}
            if decision is not None else None
        ),
        "chosen_axis_nearest_physical_downs": physical_downs,
        "scorer_logical_down_routing_within_70ms": routed,
    }


def implicated_predecessors(scoring: dict, ordinal: int, due_s: float) -> tuple[int, ...]:
    """Only earlier scorer consumers of a KeyDown close to missed due."""
    return tuple(sorted({
        x["consumed_target_ordinal"]
        for x in scoring["logical_down_target_routing"]
        if x["consumed_target_ordinal"] is not None
        and x["consumed_target_ordinal"] < ordinal
        and abs(x["time_s"] - due_s) <= .025
    }))


def predecessor_chain(
    record: dict, scoring: dict, missed: int,
) -> tuple[int, ...]:
    """Walk back through actual earlier-target KeyDown consumption.

    This is a temporal/ordered-scorer ancestry, NOT proof that a forecast
    for target k intentionally generated that physical key. Stop when no
    preceding consumer appears near that target's due, or on ambiguity.
    """
    due_by_ordinal = tuple(
        float(t["due_s"]) for t in record["target_outcomes_posthoc"]
    )
    chain: list[int] = []
    current = missed
    while len(chain) < len(due_by_ordinal):
        candidates = implicated_predecessors(
            scoring, current, due_by_ordinal[current]
        )
        if len(candidates) != 1:
            break
        parent = candidates[0]
        if parent >= current or parent in chain:
            break
        chain.append(parent)
        current = parent
    return tuple(chain)


def inspect_case(arms: dict) -> list[dict]:
    """Cross-arm cascade from a Miss back to its earliest scorer consumer."""
    invalid = [name for name, (_, scorer) in arms.items()
               if not scorer["score_replay_exact_match"]]
    if invalid:
        raise ValueError(f"cannot attribute with scorer replay mismatch: {invalid}")
    misses = sorted({
        index
        for record, _ in arms.values()
        for index, target in enumerate(record["target_outcomes_posthoc"])
        if target["miss"]
    })
    results = []
    for missed in misses:
        parents = sorted({
            parent
            for record, scoring in arms.values()
            for parent in implicated_predecessors(
                scoring, missed, record["target_outcomes_posthoc"][missed]["due_s"]
            )
        })
        chains = {
            name: predecessor_chain(record, scorer, missed)
            for name, (record, scorer) in arms.items()
        }
        focus = sorted(set([missed, *parents,
                            *(index for chain in chains.values() for index in chain)]))
        # Also display one predecessor before the terminal parent to reveal
        # a missing stroke even when no further scored KeyDown exists.
        focus = sorted(set(focus + [max(0, min(chain) - 1)
                                    for chain in chains.values() if chain]))
        results.append({
            "missed_target_ordinal": missed,
            "earlier_scorer_consumers": parents,
            "ordered_scorer_ancestry_by_arm": {
                name: list(chain) for name, chain in chains.items()
            },
            "comparison": {
                str(index): {
                    name: target_view(record, scorer, index)
                    for name, (record, scorer) in arms.items()
                }
                for index in focus
            },
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--train-chart", type=Path, default=DEFAULT_TRAIN_CHART)
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error("input directory does not exist")
    for case in CASES:
        segment = get_case(case, chart=args.train_chart)
        arms = {}
        for arm_name in ARMS:
            filename = args.input_dir / case / (arm_name + ".json")
            if not filename.is_file():
                parser.error(f"missing saved arm result: {filename}")
            record = json.loads(filename.read_text(encoding="utf-8"))
            scorer = replay_scorer(segment, record)
            if not scorer["score_replay_exact_match"]:
                raise RuntimeError(
                    f"refusing precursor analysis: {case}/{arm_name} scorer mismatch"
                )
            arms[arm_name] = (record, scorer)
        print("===", case, "===", flush=True)
        for result in inspect_case(arms):
            print(json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                  flush=True)


if __name__ == "__main__":
    main()
