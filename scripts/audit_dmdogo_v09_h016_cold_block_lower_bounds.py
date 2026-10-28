from __future__ import annotations

"""H0.16: strengthen, but never relax, H0.15's admissible suffix bound.

Cold-start *one- and two-chunk exact relaxations* retain genuine
intra-chunk hand exposure, internal reuse feasibility and two-chunk
carryover while omitting only pre-block state. No changed beam policy,
no five-chunk tuning, no Motor, no real chart/Validation20/FINAL.
"""

import argparse
import json
from math import isfinite
from pathlib import Path

from dmdod.planning.gesture_chunk_planner_v08 import (
    GestureChunkPlannerV08, _candidate_gestures,
)
from audit_dmdogo_v09_h04_beam_decisions import _chunk_record
from audit_dmdogo_v09_h05_nonquick_suffix import _next, _seed_state
from audit_dmdogo_v09_h08_protected_quota import (
    BEAM_WIDTH, trace_quota,
)
from audit_dmdogo_v09_h012_synthetic_counterexamples import (
    audit_one, generate, original_precise_cost, select_alternative,
)
from audit_dmdogo_v09_h014_regret_autopsy import (
    TARGET_IDS, replay_with_snapshots, verify_h013, verify_replay,
)
from audit_dmdogo_v09_h015_admissible_suffix_bound import (
    FORMAT as H015_FORMAT, QUOTA, EPS, SURVEY_DISCOVERY_IDS,
    SURVEY_EVALUATION_IDS, verify_frozen_cost_source,
    suffix_lower_bounds, certificate, evaluate_trace,
)
from dmdogo_v08_provenance import sha256_file

FORMAT = "dmdogo-v09-h016-cold-block-lower-bounds-v1"
FUTURE_BLOCK_MAX = 2
ABS_TOL = 1e-7
GRACE = 5


def cold_block_lower_bound(windows, start: int, end: int) -> float:
    """Minimize frozen incremental cost with EMPTY predecessor state.

    The first chunk receives zero inherited exposure, no previous axis
    or axis-use history. Inside a block, exact causal carryover is
    preserved. For an actual feasible history with nonnegative previous
    exposure and possibly previous axis uses, the same candidate sequence
    cannot cost *less* than its cold-start counterpart. Thus the minimum
    is an admissible lower bound regardless of the incoming state.

    Only one or two chunks are supported; no approximation or beam cut
    is permitted in this local minimum search.
    """
    if not (0 <= start < end <= len(windows) and end-start <= FUTURE_BLOCK_MAX):
        raise ValueError("cold block must contain one or two valid chunks")
    branches = [_seed_state(windows[start].events[0].due_s)]
    for wi in range(start, end):
        win = windows[wi]
        previous_due = windows[wi-1].events[-1].due_s if wi > start else None
        candidates = _candidate_gestures(win)
        if not candidates:
            raise ValueError("empty frozen grammar for future chunk")
        expanded = []
        for parent in branches:
            for candidate in candidates:
                nxt = _next(parent, win, candidate, previous_due,
                            frozen_candidate=False)
                if nxt is not None:
                    expanded.append(nxt)
        if not expanded:
            raise ValueError(
                f"no feasible cold block candidates at {start}:{end}")
        # Deliberately no beam limit and no score-based pruning.
        branches = expanded
    value = min(x.score for x in branches)
    if not isfinite(value) or value < 0:
        raise ValueError("non-finite/negative future cost floor")
    return value


def cold_suffix_bounds(windows) -> dict[str, tuple[float, ...]]:
    """Build exact cold 1-block and disjoint cold 2-block suffix bounds.

    For any start i:
       L1[i] = cost_cold(i:i+1) + L1[i+1]
       L2[i] = cost_cold(i:i+2) + L2[i+2]
    with L2 of a final singleton equal to cold 1-block. Each pair
    is reset to zero state, retaining within-pair carryover.
    Since each omitted cross-block term is nonnegative, both L1 and
    L2 are valid lower bounds on the full remaining continuation.
    L2 >= L1 up to numerical tolerance.
    """
    n = len(windows)
    old = suffix_lower_bounds(windows)
    one = [0.0]*(n+1)
    pair = [0.0]*(n+1)
    cold_one = [cold_block_lower_bound(windows, i, i+1)
                for i in range(n)]
    cold_two = [cold_block_lower_bound(windows, i, i+2)
                for i in range(n-1)]
    for i in range(n-1, -1, -1):
        one[i] = cold_one[i] + one[i+1]
        pair[i] = (cold_two[i] + pair[i+2]
                   if i+1 < n else cold_one[i])
    for i in range(n+1):
        if (one[i] + ABS_TOL < old[i]
                or pair[i] + ABS_TOL < one[i]):
            raise ValueError(
                f"cold bound violates lower-bound strengthening at {i}")
    # The maximum of several independently admissible bounds is itself
    # admissible. Using max also avoids making a weaker floating-point
    # representation of an existing H0.15 certificate.
    tight = tuple(max(old[i], one[i], pair[i]) for i in range(n+1))
    return {
        "intrinsic_h015": tuple(old),
        "cold_one": tuple(one),
        "cold_two": tuple(pair),
        "maximum": tight,
    }


def inspect_frontier(snapshots: list[dict], windows, *, join: int,
                     incumbent_cost: float,
                     known_tagged_final_cost: float | None,
                     baseline_records: list[dict]) -> dict:
    """Evaluate H0.15 plus two new admissible bounds on unchanged path."""
    if [s["window_index"] for s in snapshots] != list(
            range(join, len(windows))):
        raise ValueError("missing/noncontiguous quota-8 window trace")
    if [s["window_index"] for s in snapshots] != [
            r["window_index"] for r in baseline_records]:
        raise ValueError("saved H0.15 rows differ from replayed window census")
    floors = cold_suffix_bounds(windows)
    by_method = {key: [] for key in floors}
    for row, old in zip(snapshots, baseline_records):
        wi = row["window_index"]
        tagged = row["retained_best_tagged"]
        score = tagged["score"] if tagged is not None else None
        if (old["best_tagged_current_score"] !=
                (round(score, 9) if score is not None else None)):
            raise ValueError("replayed tagged score differs from H0.15 row")
        for key, suffix in floors.items():
            outcome = certificate(score, suffix[wi+1], incumbent_cost)
            if key == "intrinsic_h015":
                # Exact baseline parity, not merely equal certificate counts.
                if outcome != {
                        k: old[k] for k in (
                            "applicable", "certified", "lower_bound",
                            "incumbent_cost", "slack",
                            "margin_lb_minus_incumbent")}:
                    raise ValueError("H0.15 baseline certificate differs")
            by_method[key].append({
                "window_index": wi,
                "offset_from_join": wi-join,
                "future_lb": round(suffix[wi+1], 9),
                **outcome,
            })
    summary = {}
    for method, rows in by_method.items():
        certified = [x for x in rows if x["certified"]]
        early = [x for x in certified if x["offset_from_join"] < GRACE]
        summary[method] = {
            "tested_windows": len(rows),
            "certified_window_count": len(certified),
            "certified_target": bool(certified),
            "early_certificate": bool(early),
            "first_certificate_window": (
                certified[0]["window_index"] if certified else None),
            "first_certificate_offset": (
                certified[0]["offset_from_join"] if certified else None),
            "first_early_certificate_offset": (
                early[0]["offset_from_join"] if early else None),
            "window_records": rows,
        }
    for i in range(len(snapshots)):
        # Stronger lower bounds cannot revoke old, certified claims.
        if (by_method["intrinsic_h015"][i]["certified"]
                and not by_method["maximum"][i]["certified"]):
            raise ValueError("strengthened bound revoked existing certificate")
    if (known_tagged_final_cost is not None
            and known_tagged_final_cost < incumbent_cost-EPS):
        if any(value["certified_window_count"] for value in summary.values()):
            raise ValueError(
                "false certificate: surviving protected lineage "
                "actually beats frozen incumbent")
    return summary


def _source_trial_map(h015: dict, name: str) -> dict[int, dict]:
    if name == "known_regret_cases":
        expected = list(TARGET_IDS)
        source = h015[name]
    else:
        expected = list(
            SURVEY_DISCOVERY_IDS if name == "discovery_fixed"
            else SURVEY_EVALUATION_IDS)
        source = h015["fixed_survey"][name]
        if source["trial_indices"] != expected:
            raise ValueError("saved H0.15 fixed survey indices changed")
    by_id = {row["trial_index"]: row for row in source["trials"]}
    if len(by_id) != len(source["trials"]) or set(by_id) != set(expected):
        raise ValueError("saved H0.15 trial/source census mismatch")
    return by_id


def inspect_trial(trial_id: int, saved: dict) -> dict:
    """Reproduce original H0.15 outcomes, then compute stronger bounds."""
    original = audit_one(trial_id)
    events, spec = generate(trial_id)
    if (original["generator"] != spec or saved["generator"] != spec
            or saved["target_count"] != len(original["targets"])):
        raise ValueError("synthetic trial generator or target count changed")
    planner = GestureChunkPlannerV08()
    plan = planner.plan(events)
    if not plan.feasible:
        raise ValueError("frozen synthetic plan infeasible")
    windows = planner._windows(events)
    chunks = [_chunk_record(x) for x in plan.chunks]
    incumbent = original_precise_cost(events, plan, windows, chunks)
    by_join = {row["window_index"]: row for row in saved["targets"]}
    if (len(by_join) != len(saved["targets"])
            or set(by_join) != {
                row["window_index"] for row in original["targets"]}):
        raise ValueError("H0.15 targeted join windows changed")
    outputs = []
    for source in original["targets"]:
        join = source["window_index"]
        historical = by_join[join]
        target = select_alternative(events, chunks, windows, join)
        if (target != source["target"]
                or target["flag_pair"] != historical["flag_pair"]
                or historical["original_v08_cost"] != round(incumbent, 9)
                or historical["release_rule_regret_in_existing_trial"]
                != source["counterexample"]["rule_regret"]):
            raise ValueError("saved H0.15 target or incumbent identity changed")
        trace = replay_with_snapshots(
            events, chunks, target, policy="full_quota8")
        canonical = trace_quota(
            events, chunks, target, quota=QUOTA,
            expected_cost=plan.metrics["beam_cost"],
            expected_evaluated=plan.candidates_evaluated,
            reference_final_cost=incumbent)
        verify_replay(trace, canonical, adaptive=False)
        tagged = canonical["best_protected_terminal"]
        if (canonical["beam_winner"]["final_cost"]
                != historical["full_quota8_final_cost"]
                or (tagged["final_cost"] if tagged is not None else None)
                != historical["full_quota8_tagged_terminal_cost"]):
            raise ValueError("saved H0.15 full quota8 final costs differ")
        old = evaluate_trace(
            trace["snapshots"], windows, target_join=join,
            incumbent_cost=incumbent,
            full_protected_terminal_cost=(
                tagged["final_cost"] if tagged is not None else None))
        if old != historical["certificate"]:
            raise ValueError("saved H0.15 original certificate diverged")
        computed = inspect_frontier(
            trace["snapshots"], windows, join=join,
            incumbent_cost=incumbent,
            known_tagged_final_cost=(
                tagged["final_cost"] if tagged is not None else None),
            baseline_records=old["records"])
        outputs.append({
            "window_index": join,
            "flag_pair": target["flag_pair"],
            "tagged_final_beats_incumbent": (
                tagged is not None
                and tagged["final_cost"] < incumbent-EPS),
            "original_v08_cost": round(incumbent, 9),
            "tagged_final_cost": (
                tagged["final_cost"] if tagged is not None else None),
            "five_chunk_rule_regret":
                source["counterexample"]["rule_regret"],
            "methods": computed,
        })
    return {
        "trial_index": trial_id,
        "generator": spec,
        "target_count": len(outputs),
        "targets": outputs,
    }


def summarize(rows: list[dict]) -> dict:
    targets = [x for row in rows for x in row["targets"]]
    output = {
        "trial_count": len(rows),
        "target_count": len(targets),
        "known_regret_target_count": sum(
            t["five_chunk_rule_regret"] for t in targets),
        "tagged_final_beats_incumbent": sum(
            t["tagged_final_beats_incumbent"] for t in targets),
        "methods": {},
    }
    for method in ("intrinsic_h015", "cold_one", "cold_two", "maximum"):
        summaries = [row["methods"][method] for row in targets]
        output["methods"][method] = {
            "certifiable_targets": sum(
                x["certified_target"] for x in summaries),
            "early_certifiable_targets": sum(
                x["early_certificate"] for x in summaries),
            "total_certified_windows": sum(
                x["certified_window_count"] for x in summaries),
            "first_certificate_offsets": [
                x["first_certificate_offset"] for x in summaries
                if x["first_certificate_offset"] is not None],
        }
    return output


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("h012-report", "h013-report", "h014-report",
                   "h015-report", "output"):
        p.add_argument("--"+option, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("refusing to overwrite previous H0.16 results")
    root = Path(__file__).resolve().parents[1]
    source_blobs = verify_frozen_cost_source(root)
    source = {}
    for name, path in (
        ("h012", a.h012_report),
        ("h013", a.h013_report),
        ("h014", a.h014_report),
        ("h015", a.h015_report),
    ):
        source[name] = json.loads(path.read_text(encoding="utf-8"))
    original_regrets = verify_h013(
        source["h013"], source["h012"], h012_path=a.h012_report)
    h015 = source["h015"]
    if (h015.get("format") != H015_FORMAT
            or h015.get("source_h012_sha256") != sha256_file(a.h012_report)
            or h015.get("source_h013_sha256") != sha256_file(a.h013_report)
            or h015.get("source_h014_sha256") != sha256_file(a.h014_report)
            or h015.get("pinned_frozen_source_blobs") != source_blobs
            or h015.get("beam_width") != BEAM_WIDTH
            or h015.get("protected_quota") != QUOTA
            or h015.get("release_grace_chunks") != GRACE
            or h015.get("physical_motor_executed") is not False):
        raise ValueError("H0.15 provenance or certificate contract differs")
    if (list(original_regrets) != list(TARGET_IDS)
            or h015["known_regret_cases"]["trial_ids"] != list(TARGET_IDS)
            or h015["known_regret_cases"]["aggregate"][
                "any_false_certificate_before_tagged_improvement"] != 0):
        raise ValueError("known counterexample control/census mismatch")
    from audit_dmdogo_v09_h014_regret_autopsy import FORMAT as H014_FORMAT
    h014 = source["h014"]
    if (h014.get("format") != H014_FORMAT
            or h014.get("source_h012_sha256") != sha256_file(a.h012_report)
            or h014.get("source_h013_sha256") != sha256_file(a.h013_report)
            or h014.get("target_trial_indices") != list(TARGET_IDS)
            or h014.get("physical_motor_executed") is not False):
        raise ValueError("H0.14 historical source identity mismatch")
    groups = {}
    for group in ("known_regret_cases", "discovery_fixed", "evaluation_fixed"):
        saved = _source_trial_map(h015, group)
        rows = [inspect_trial(trial_id, entry)
                for trial_id, entry in saved.items()]
        result = summarize(rows)
        old_aggregate = (h015["known_regret_cases"]["aggregate"]
                         if group == "known_regret_cases"
                         else h015["fixed_survey"][group]["aggregate"])
        if (result["target_count"] != old_aggregate["target_count"]
                or result["methods"]["intrinsic_h015"]["certifiable_targets"]
                != old_aggregate["certifiable_target_count"]
                or result["methods"]["intrinsic_h015"]["early_certifiable_targets"]
                != old_aggregate["certifiable_within_first_five_count"]
                or result["methods"]["intrinsic_h015"]["total_certified_windows"]
                != old_aggregate["total_certified_frontier_windows"]):
            raise ValueError("H0.15 preserved benchmark summary differs")
        if group == "known_regret_cases" and any(
                r["methods"]["maximum"]["certified_target"]
                for trial in rows for r in trial["targets"]):
            raise ValueError("unsafe certificate on prior regret counterexample")
        groups[group] = {
            "trial_indices": list(saved),
            "aggregate": result,
            "trials": rows,
        }
    report = {
        "format": FORMAT,
        "source_h012_sha256": sha256_file(a.h012_report),
        "source_h013_sha256": sha256_file(a.h013_report),
        "source_h014_sha256": sha256_file(a.h014_report),
        "source_h015_sha256": sha256_file(a.h015_report),
        "frozen_source_blobs": source_blobs,
        "beam_width": BEAM_WIDTH,
        "quota": QUOTA,
        "five_chunk_rule_unchanged": True,
        "no_new_pruning_policy": True,
        "methods": {
            "intrinsic_h015": "original H0.15 independent minimum per chunk",
            "cold_one": "exact local cost with zero inherited state, reset every chunk",
            "cold_two": "exact min over each adjacent cold-start pair, reset between pairs",
            "maximum": "pointwise max of all three admissible suffix bounds",
        },
        "bound_scope": (
            "Relative to a completed v0.8 incumbent, certify that no "
            "continuation from current full-quota8 retained tagged frontier "
            "can beat incumbent; cannot certify that altering width-24 "
            "beam selection leaves its final winner unchanged."
        ),
        "groups": groups,
        "physical_motor_executed": False,
        "frozen_v08_changed": False,
        "human_likeness_verified": False,
        "limitations": [
            "All cases are the same previously examined procedural seeds as H0.15; no independent holdout.",
            "Cold block resets previous exposure, last axis and reuse history to zero; optimistic cost lower bounds follow only from the pinned frozen nonnegative scorer.",
            "Within each two-chunk block exact feasible local candidates are enumerated without beam pruning; blocks discard boundary coupling.",
            "Positive certificates refer only to the current tagged frontier against an offline feasible incumbent, never to a guaranteed unchanged global Beam winner.",
            "Stronger or earlier certificates do not estimate CPU speedups or actual safe new beam-pruning behavior.",
            "No real Train/Validation20/FINAL, Motor, HUD observation, or new learning.",
        ],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({
        "groups": {name: group["aggregate"]
                   for name, group in groups.items()},
        "saved": str(a.output),
        "no_new_pruning_policy": True,
        "physical_motor_executed": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
