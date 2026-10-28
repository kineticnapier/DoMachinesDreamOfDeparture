from __future__ import annotations

"""H0.15: admissible suffix-bound study; NO modified pruning policy.

Mathematical scope: relative to an already known complete, feasible
frozen v0.8 path, a retained tagged prefix with score S and admissible
remaining cost L cannot yield a strictly cheaper complete tagged
descendant if S + L >= incumbent cost + conservative slack.
This is a certificate for *that tagged frontier versus that incumbent*,
NOT proof that a different width-24 beam or the five-chunk early release
preserves the full quota-8 winner. The known H0.14 regret cases MUST
remain uncertified before their full quota-8 tagged winner.

Every coefficient/source used by this bound is pinned to the read-only
v0.8/H0.4 code. The bound never affects beam selection or learning.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from dmdod.planning.gesture_chunk_planner_v08 import (
    GestureChunkPlannerV08, _axis_parts, _candidate_gestures,
)
from audit_dmdogo_v09_h04_beam_decisions import _chunk_record
from audit_dmdogo_v09_h012_synthetic_counterexamples import (
    generate, original_precise_cost, select_alternative, audit_one,
)
from audit_dmdogo_v09_h014_regret_autopsy import (
    FORMAT as H014_FORMAT, TARGET_IDS, replay_with_snapshots,
    verify_h013, verify_replay, QUOTA, EPS,
)
from audit_dmdogo_v09_h08_protected_quota import trace_quota
from dmdogo_v08_provenance import sha256_file

FORMAT = "dmdogo-v09-h015-admissible-suffix-bound-v1"
BOUND_SLACK_ABS = 1e-6
BOUND_SLACK_REL = 1e-12
SURVEY_DISCOVERY_IDS = tuple(range(4096, 4104))
SURVEY_EVALUATION_IDS = tuple(range(8192, 8200))
PINNED_GIT_BLOBS = {
    "src/dmdod/planning/gesture_chunk_planner_v08.py":
        "4ce02c2e4450dc7a5181ab74cbaa0c1accea94f5",
    "scripts/audit_dmdogo_v09_h04_beam_decisions.py":
        "0d07bd34ec3ffa0b2acb1beabcede659318a71bb",
}
# These constants are *read from* the frozen implementation and are
# checked with source Git blob SHA before any certificate is reported.
HAND_CHANGE_COST = 0.35
EXCESS_DEPTH_COST = 0.10


def git_blob_sha(path: Path) -> str:
    # pathlib text-mode normalizes Windows CRLF to frozen repo LF.
    body = path.read_text(encoding="utf-8").encode("utf-8")
    return hashlib.sha1(
        b"blob " + str(len(body)).encode("ascii") + b"\0" + body).hexdigest()


def verify_frozen_cost_source(root: Path) -> dict:
    observed = {}
    for rel, required in PINNED_GIT_BLOBS.items():
        path = root / rel
        actual = git_blob_sha(path)
        if actual != required:
            raise ValueError(
                f"frozen nonnegative-cost assumption not verified: {rel} "
                f"expected {required}, found {actual}")
        observed[rel] = actual
    return observed


def intrinsic_cost(candidate) -> float:
    """Exact context-free lower-bound terms of v0.8 six-term objective.

    Other frozen components (hand-exposure, reuse margin, seam join)
    are nonnegative and are safely relaxed to zero.
    """
    if candidate.preference < 0:
        raise ValueError("negative preference contradicts admissible bound")
    flat = [axis for row in candidate.axes for axis in row]
    changes = sum(
        _axis_parts(a)[0] != _axis_parts(b)[0]
        for a, b in zip(flat, flat[1:]))
    depth = sum(
        max(0, abs(_axis_parts(a)[1] - _axis_parts(b)[1]) - 1)
        for a, b in zip(flat, flat[1:])
        if _axis_parts(a)[0] == _axis_parts(b)[0])
    result = candidate.preference + HAND_CHANGE_COST*changes + EXCESS_DEPTH_COST*depth
    if result < 0:
        raise ValueError("negative future lower-bound component")
    return result


def suffix_lower_bounds(windows) -> tuple[float, ...]:
    """Bound indexed by start of a suffix: L[i]=sum(min intrinsic[j],j>=i)."""
    min_local = []
    for wi, window in enumerate(windows):
        candidates = _candidate_gestures(window)
        if not candidates:
            raise ValueError(f"no frozen grammar candidates at window {wi}")
        min_local.append(min(intrinsic_cost(c) for c in candidates))
    suffix = [0.0]*(len(windows)+1)
    for i in range(len(windows)-1, -1, -1):
        suffix[i] = suffix[i+1] + min_local[i]
    return tuple(suffix)


def certificate(prefix_score: float | None, future_lb: float,
                known_complete_cost: float) -> dict:
    """Conservative proof of *no strict improvement over known incumbent*.

    A negative outcome is "not certified", not "will improve".
    """
    from math import isfinite
    if (future_lb < 0 or not isfinite(future_lb)
            or not isfinite(known_complete_cost)):
        raise ValueError("non-finite/negative lower bound")
    slack = BOUND_SLACK_ABS + BOUND_SLACK_REL * abs(known_complete_cost)
    if prefix_score is None:
        return {
            "applicable": False, "certified": False,
            "lower_bound": None, "incumbent_cost": known_complete_cost,
            "slack": slack, "margin_lb_minus_incumbent": None,
        }
    if not isfinite(prefix_score) or prefix_score < 0:
        raise ValueError("invalid frozen prefix score")
    bound = prefix_score + future_lb
    margin = bound - known_complete_cost
    return {
        "applicable": True,
        "certified": margin > slack,
        "lower_bound": round(bound, 9),
        "incumbent_cost": round(known_complete_cost, 9),
        "slack": slack,
        "margin_lb_minus_incumbent": round(margin, 9),
    }


def evaluate_trace(snapshots: list[dict], windows, *, target_join: int,
                   incumbent_cost: float,
                   full_protected_terminal_cost: float | None) -> dict:
    """Post-hoc certificate census on ONE unmodified quota8 trajectory."""
    if not 0 <= target_join < len(windows):
        raise ValueError("target window outside source")
    if [s["window_index"] for s in snapshots] != list(
            range(target_join, len(windows))):
        raise ValueError("noncontiguous or missing full-quota8 snapshots")
    suffix = suffix_lower_bounds(windows)
    rows = []
    first_certified = None
    for snap in snapshots:
        wi = snap["window_index"]
        tagged = snap["retained_best_tagged"]
        result = certificate(
            tagged["score"] if tagged is not None else None,
            suffix[wi+1], incumbent_cost)
        if result["certified"] and first_certified is None:
            first_certified = wi
        rows.append({
            "window_index": wi,
            "offset_from_join": wi-target_join,
            "tagged_retained_count": snap["tagged_retained_count"],
            "tagged_expanded_rank": snap["best_tagged_expanded_rank"],
            "best_tagged_current_score": (
                tagged["score"] if tagged is not None else None),
            "optimistic_future_cost": round(suffix[wi+1], 9),
            **result,
        })
    positive_certificate_count = sum(x["certified"] for x in rows)
    if full_protected_terminal_cost is not None:
        # A certified bound cannot coexist with a known tagged
        # terminal result better than the feasible frozen incumbent,
        # *unless that terminal lineage was separately pruned before
        # the snapshot*. For full quota8, all descendant lineages of
        # previous best can change; however the minimum tagged score
        # lower bound is over retained candidates. Any future tagged
        # terminal descends from the retained frontier, so safety holds.
        if (full_protected_terminal_cost < incumbent_cost - EPS
                and positive_certificate_count):
            raise ValueError(
                "false optimistic suffix certificate: tagged descendant "
                "later beats the incumbent")
    return {
        "tested_windows": len(rows),
        "certified_frontier_windows": positive_certificate_count,
        "first_certified_window": first_certified,
        "certified_before_or_at_five_chunk_release": any(
            r["certified"] and r["offset_from_join"] <= 4 for r in rows),
        "earliest_grace_window": rows[0]["window_index"],
        "grace_end_window": target_join+4,
        "last_tagged_cost_vs_incumbent": rows[-1][
            "margin_lb_minus_incumbent"],
        "records": rows,
    }


def inspect_trial(trial_id: int, *, selected_join: int | None = None) -> dict:
    """Deterministic post-hoc study without editing original beam policies."""
    sample = audit_one(trial_id)
    events, source_spec = generate(trial_id)
    if sample["generator"] != source_spec:
        raise ValueError("synthetic generator provenance diverged")
    plan = GestureChunkPlannerV08().plan(events)
    if not plan.feasible or not sample["frozen_plan_feasible"]:
        raise ValueError("inspected synthetic plan infeasible")
    planner = GestureChunkPlannerV08()
    windows = planner._windows(events)
    chunks = [_chunk_record(row) for row in plan.chunks]
    incumbent = original_precise_cost(events, plan, windows, chunks)
    results = []
    for saved in sample["targets"]:
        join = saved["window_index"]
        if selected_join is not None and selected_join != join:
            continue
        target = select_alternative(events, chunks, windows, join)
        if saved["target"] != target:
            raise ValueError("frozen target identity changed")
        full = replay_with_snapshots(
            events, chunks, target, policy="full_quota8")
        kwargs = {
            "expected_cost": plan.metrics["beam_cost"],
            "expected_evaluated": plan.candidates_evaluated,
            "reference_final_cost": incumbent,
        }
        canonical = trace_quota(events, chunks, target, quota=QUOTA, **kwargs)
        verify_replay(full, canonical, adaptive=False)
        if (full["terminal_winner"]["final_cost"]
                != saved["baseline"]["full_quota8_cost"]
                or round(incumbent, 9)
                != saved["baseline"]["v08_cost"]):
            raise ValueError("full quota8 vs H0.12/H0.13 control differs")
        bound = evaluate_trace(
            full["snapshots"], windows, target_join=join,
            incumbent_cost=incumbent,
            full_protected_terminal_cost=(
                canonical["best_protected_terminal"]["final_cost"]
                if canonical["best_protected_terminal"] is not None else None))
        results.append({
            "window_index": join,
            "flag_pair": target["flag_pair"],
            "full_quota8_beats_incumbent":
                full["terminal_winner"]["final_cost"] < incumbent - EPS,
            "protected_terminal_beats_incumbent": (
                canonical["best_protected_terminal"] is not None
                and canonical["best_protected_terminal"]["final_cost"]
                < incumbent - EPS),
            "original_v08_cost": round(incumbent, 9),
            "full_quota8_final_cost": canonical["beam_winner"]["final_cost"],
            "full_quota8_tagged_terminal_cost": (
                canonical["best_protected_terminal"]["final_cost"]
                if canonical["best_protected_terminal"] is not None else None),
            "release_rule_regret_in_existing_trial":
                saved["counterexample"]["rule_regret"],
            "certificate": bound,
        })
    return {
        "trial_index": trial_id,
        "generator": source_spec,
        "target_count": len(results),
        "targets": results,
    }


def _aggregate(rows: list[dict]) -> dict:
    targets = [t for row in rows for t in row["targets"]]
    return {
        "synthetic_trial_count": len(rows),
        "target_count": len(targets),
        "certifiable_target_count": sum(
            t["certificate"]["first_certified_window"] is not None
            for t in targets),
        "certifiable_within_first_five_count": sum(
            t["certificate"]["certified_before_or_at_five_chunk_release"]
            for t in targets),
        "tagged_winner_better_than_incumbent": sum(
            t["protected_terminal_beats_incumbent"] for t in targets),
        "any_false_certificate_before_tagged_improvement": sum(
            t["protected_terminal_beats_incumbent"]
            and t["certificate"]["certified_frontier_windows"] > 0
            for t in targets),
        "released_rule_regret_cases": sum(
            t["release_rule_regret_in_existing_trial"] for t in targets),
        "total_certified_frontier_windows": sum(
            t["certificate"]["certified_frontier_windows"] for t in targets),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h012-report", type=Path, required=True)
    p.add_argument("--h013-report", type=Path, required=True)
    p.add_argument("--h014-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("refusing to overwrite H0.15 saved results")
    root = Path(__file__).resolve().parents[1]
    source_blobs = verify_frozen_cost_source(root)
    h012, h013, h014 = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (a.h012_report, a.h013_report, a.h014_report)]
    from audit_dmdogo_v09_h014_regret_autopsy import verify_h013
    original_regrets = verify_h013(
        h013, h012, h012_path=a.h012_report)
    if (h014.get("format") != H014_FORMAT
            or h014.get("source_h012_sha256") != sha256_file(a.h012_report)
            or h014.get("source_h013_sha256") != sha256_file(a.h013_report)
            or h014.get("target_trial_indices") != list(TARGET_IDS)
            or len(h014["cases"]) != 2
            or h014.get("beam_width") != BEAM_WIDTH
            or h014.get("protected_quota") != QUOTA
            or h014.get("release_grace_chunks") != 5
            or h014.get("physical_motor_executed") is not False):
        raise ValueError("H0.14 source/provenance mismatch")
    regret_rows = []
    for i, entry in enumerate(h014["cases"]):
        trial_id = TARGET_IDS[i]
        if (entry["trial_index"] != trial_id
                or entry["summary"]["regret_early_release_minus_full"] <= EPS
                or entry["summary"]["target_window"] !=
                original_regrets[trial_id]["window_index"]):
            raise ValueError("H0.14 counterexample identity changed")
        fresh = inspect_trial(trial_id,
                              selected_join=entry["summary"]["target_window"])
        if len(fresh["targets"]) != 1:
            raise ValueError("missing known H0.14 regret target")
        record = fresh["targets"][0]
        if (not record["release_rule_regret_in_existing_trial"]
                or not record["protected_terminal_beats_incumbent"]
                or record["certificate"]["certified_frontier_windows"]):
            raise ValueError(
                "optimistic bound illegally discards the known winning "
                "protected lineage in H0.14")
        regret_rows.append(fresh)
    survey_groups = {}
    for name, ids in (
        ("discovery_fixed", SURVEY_DISCOVERY_IDS),
        ("evaluation_fixed", SURVEY_EVALUATION_IDS),
    ):
        recorded = {r["trial_index"]: r
                    for r in h013[name.split("_")[0]]["trials"]}
        if not set(ids).issubset(recorded):
            raise ValueError("H0.13 survey trial IDs not in source cohort")
        rows = []
        for idx in ids:
            fresh = audit_one(idx)
            if fresh != recorded[idx]:
                raise ValueError("H0.13 survey source trial failed replay")
            rows.append(inspect_trial(idx))
        survey_groups[name] = {
            "trial_indices": list(ids),
            "aggregate": _aggregate(rows),
            "trials": rows,
        }
    report = {
        "format": FORMAT,
        "source_h012_sha256": sha256_file(a.h012_report),
        "source_h013_sha256": sha256_file(a.h013_report),
        "source_h014_sha256": sha256_file(a.h014_report),
        "pinned_frozen_source_blobs": source_blobs,
        "beam_width": BEAM_WIDTH,
        "protected_quota": QUOTA,
        "release_grace_chunks": 5,
        "certification_claim": (
            "Only the currently retained tagged descendants cannot "
            "strictly beat an already known fully feasible incumbent "
            "when their cumulative frozen cost + an admissible "
            "suffix lower bound exceeds incumbent by >1e-6+relative slack. "
            "Does not imply equal beam search results under changed pruning."
        ),
        "known_regret_cases": {
            "trial_ids": list(TARGET_IDS),
            "aggregate": _aggregate(regret_rows),
            "trials": regret_rows,
        },
        "fixed_survey": survey_groups,
        "no_pruning_policy_implemented": True,
        "production_change": False,
        "physical_motor_executed": False,
        "frozen_v08_changed": False,
        "limitations": [
            "The bound is conditioned on the exact pinned nonnegative six-term heuristic.",
            "The incumbent is the complete offline v0.8 path: not necessarily available in a live HUD-only controller.",
            "Certificates apply only to the retained tagged frontier of a particular full-quota8 trajectory and its possible continuations.",
            "A frontier certificate cannot prove that removing protected slots preserves the global winner under beam search with other competing lineages.",
            "Non-certificate does not imply a future improvement; it simply supplies no safe pruning proof.",
            "H0.13/14 data and fixed 8+8 survey seeds are previously observed in this research and cannot establish independent generalization.",
            "No old Train, Validation20, FINAL, Motor, or timing training.",
        ],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2, ensure_ascii=False)
        out.write("\n")
    print(json.dumps({
        "known_regret_cases": report["known_regret_cases"]["aggregate"],
        "fixed_survey": {k: v["aggregate"] for k, v in survey_groups.items()},
        "saved": str(a.output),
        "no_pruning_policy_implemented": True,
        "physical_motor_executed": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
