from __future__ import annotations

"""H0.14: exact beam-population autopsy of H0.13 regret IDs 4115/4127.

No threshold search, no new candidates, no production changes. Replays
the full quota-8 and original five-chunk release independently, tracing
both populations and every component of the original six-term heuristic.
Data are artificial, deterministic H0.12 generator events, not ADOFAI
Train/Validation charts. No Motor/Timing/FINAL/training.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from dmdod.planning.gesture_chunk_planner_v08 import (
    GestureChunkPlannerV08, _candidate_gestures,
)
from audit_dmdogo_v09_h04_beam_decisions import _chunk_record
from audit_dmdogo_v09_h05_nonquick_suffix import _cost_terms, _next, _seed_state
from audit_dmdogo_v09_h07_shadow_lane import (
    Node, Proposal, BeamPath, _chunk_record_from,
)
from audit_dmdogo_v09_h08_protected_quota import (
    BEAM_WIDTH, TARGET_TAG, _describe, choose_quota, trace_quota,
)
from audit_dmdogo_v09_h011_adaptive_release import (
    OBSERVE_CHUNKS, stop_after_grace, trace_adaptive,
)
from audit_dmdogo_v09_h012_synthetic_counterexamples import (
    generate, original_precise_cost, select_alternative, audit_one,
    classify, summarize,
)
from audit_dmdogo_v09_h013_late_reversal import (
    FORMAT as H013_FORMAT, DISCOVERY_START, DEFAULT_DISCOVERY,
    EVALUATION_START, DEFAULT_EVALUATION,
    check_h012_source, evaluate_counterexamples,
)
from dmdogo_v08_provenance import sha256_file

FORMAT = "dmdogo-v09-h014-regret-beam-population-autopsy-v1"
TARGET_IDS = (4115, 4127)
QUOTA = 8
EPS = 1e-7


@dataclass(slots=True)
class Tracked:
    state: BeamPath
    key: tuple[int, ...]  # Exact chosen candidate enumeration per chunk


def _terms_difference(left: dict | None, right: dict | None) -> dict | None:
    if left is None or right is None:
        return None
    if set(left) != set(right):
        raise ValueError("cost term sets differ")
    return {k: round(left[k] - right[k], 9) for k in left}


def _best(entries: list[Proposal], tagged: bool) -> Proposal | None:
    return next((p for p in entries if p.protected == tagged), None)


def _brief(branch) -> dict | None:
    if branch is None:
        return None
    return {
        "score": round(branch.score, 9),
        "terms": _cost_terms(branch),
        "last_axis": branch.last_axis,
        "tags": sorted(branch.tags),
    }


def replay_with_snapshots(events, frozen_chunks: list[dict],
                          target: dict, *, policy: str) -> dict:
    """A fresh replay with unchanged ordering and detailed beam snapshots.

    Distinguish "expanded" from "retained" so first-pruning loss can be
    identified exactly; track path identity by candidate-index tuple.
    """
    if policy not in ("full_quota8", "five_chunk_release"):
        raise ValueError("unknown replay policy")
    windows = GestureChunkPlannerV08()._windows(events)
    if len(windows) != len(frozen_chunks):
        raise ValueError("source/frozen chunk boundaries differ")
    join = target["window_index"]
    if (not 0 < join < len(windows)
            or windows[join - 1].events[-1].event_index != target["flag_pair"][0]
            or windows[join].events[0].event_index != target["flag_pair"][1]):
        raise ValueError("target seam differs")
    states = [Tracked(BeamPath(_seed_state(events[0].due_s), None), ())]
    tagged_rank1_ever = False
    observed = 0
    released = False
    release_at = None
    admission_rank = None
    rescue_windows = []
    displaced_slots = 0
    snapshots = []
    for wi, window in enumerate(windows):
        previous_due = windows[wi-1].events[-1].due_s if wi else None
        axes = tuple(tuple(a) for a in frozen_chunks[wi]["axes"])
        family = frozen_chunks[wi]["gesture_family"]
        choices = []
        keys = {}
        for rank, parent in enumerate(states, 1):
            for ci, candidate in enumerate(_candidate_gestures(window)):
                nxt = _next(
                    parent.state.state, window, candidate, previous_due,
                    frozen_candidate=(
                        candidate.axes == axes and candidate.family == family))
                if nxt is None:
                    continue
                if (wi == join and parent.state.state.frozen_prefix
                        and ci == target["nonquick_candidate_index"]):
                    nxt.tags = frozenset({TARGET_TAG})
                prop = Proposal(
                    nxt, parent.state.node, _chunk_record_from(window, candidate),
                    rank, ci)
                choices.append(prop)
                keys[id(prop)] = parent.key + (ci,)
        if not choices:
            raise ValueError(f"all feasible hypotheses disappeared in window {wi}")
        choices.sort(key=lambda p: p.state.score)
        if wi < join:
            selected = choices[:BEAM_WIDTH]
            if sum(p.state.frozen_prefix for p in selected) != 1:
                raise ValueError("original selected predecessor was pruned")
        else:
            if wi == join:
                tagged = [(rank, p) for rank, p in enumerate(choices, 1)
                          if p.protected]
                if len(tagged) != 1:
                    raise ValueError("target admission is not unique")
                admission_rank, admitted = tagged[0]
                if (admission_rank != target["nonquick_join_expanded_rank"]
                        or admitted.candidate_index
                        != target["nonquick_candidate_index"]
                        or admitted.chunk["axes"] != target["nonquick_axes"]):
                    raise ValueError("target admission cost rank/identity differs")
            current_was_released = released
            expanded_tagged = sum(p.protected for p in choices)
            expanded_top_tagged = sum(
                p.protected for p in choices[:BEAM_WIDTH])
            best_tagged = _best(choices, True)
            best_untagged = _best(choices, False)
            if policy == "full_quota8" or not released:
                selected, status = choose_quota(choices, quota=QUOTA)
                replacements = status["displaced_nonprotected_count"]
                if replacements:
                    rescue_windows.append(wi)
                    displaced_slots += replacements
                if policy == "five_chunk_release":
                    tagged_rank1_ever |= choices[0].protected
                    observed += 1
                    if stop_after_grace(
                            chunks_seen=observed,
                            ever_protected_rank1=tagged_rank1_ever):
                        released = True
                        release_at = wi
            else:
                selected = choices[:BEAM_WIDTH]
                replacements = 0
            retained_best_tagged = _best(selected, True)
            retained_best_untagged = _best(selected, False)
            tagged_expanded_rank = next((
                i for i, p in enumerate(choices, 1)
                if p.protected), None)
            snapshots.append({
                "window_index": wi,
                "offset_from_join": wi - join,
                "event_range": [window.events[0].event_index,
                                window.events[-1].event_index],
                "release_already_active_before_selection":
                    current_was_released,
                "quota_released_after_window": released,
                "expanded_count": len(choices),
                "expanded_tagged_count": expanded_tagged,
                "natural_tagged_count": expanded_top_tagged,
                "tagged_retained_count": sum(p.protected for p in selected),
                "displaced_slots": replacements,
                "best_tagged_expanded_rank": tagged_expanded_rank,
                "expanded_best_tagged": _brief(
                    best_tagged.state if best_tagged else None),
                "expanded_best_untagged": _brief(
                    best_untagged.state if best_untagged else None),
                "expanded_tagged_minus_untagged_terms":
                    _terms_difference(
                        _cost_terms(best_tagged.state) if best_tagged else None,
                        _cost_terms(best_untagged.state) if best_untagged else None),
                "cutoff_score": round(choices[
                    min(BEAM_WIDTH, len(choices))-1].state.score, 9),
                "retained_best_tagged": _brief(
                    retained_best_tagged.state if retained_best_tagged else None),
                "retained_best_untagged": _brief(
                    retained_best_untagged.state if retained_best_untagged else None),
                "retained_best_global": _brief(selected[0].state),
                "retained_keys": [list(keys[id(p)]) for p in selected],
                "tagged_expanded_but_pruned": (
                    expanded_tagged > 0
                    and not any(p.protected for p in selected)),
            })
        states = [Tracked(
            BeamPath(p.state, Node(p.parent, p.chunk)), keys[id(p)])
            for p in selected]

    winner = states[0].state
    tagged_terminal = [(i, t) for i, t in enumerate(states, 1)
                       if TARGET_TAG in t.state.state.tags]
    protected = tagged_terminal[0][1].state if tagged_terminal else None
    if not snapshots:
        raise ValueError("no target or downstream snapshots")
    return {
        "policy": policy,
        "flag_pair": target["flag_pair"],
        "window_index": join,
        "admission_expanded_rank": admission_rank,
        "release_decision_window": release_at,
        "quota_released": released,
        "ever_tagged_rank1_during_grace": tagged_rank1_ever
        if policy == "five_chunk_release" else None,
        "rescue_windows": rescue_windows,
        "displaced_slots": displaced_slots,
        "terminal_best_tagged_rank": (
            tagged_terminal[0][0] if tagged_terminal else None),
        "terminal_winner": _describe(events, frozen_chunks, winner, 0.0),
        "terminal_best_tagged": _describe(
            events, frozen_chunks, protected, 0.0),
        "snapshots": snapshots,
        "physical_motor_executed": False,
    }


def verify_replay(actual: dict, reference: dict, *, adaptive: bool) -> None:
    """Compare all terminal properties and the exact rescue interventions."""
    if (actual["admission_expanded_rank"] !=
            reference["admission_expanded_rank"]):
        raise ValueError("different candidate was admitted at target")
    expected_rescue = (
        reference["rescue_windows"] if adaptive
        else reference["slot_replacement_windows"])
    expected_displaced = (
        reference["displaced_untagged_slots"] if adaptive
        else reference["total_displaced_nonprotected_slots"])
    expected_winner = reference["beam_winner"]
    expected_tagged = reference["best_protected_terminal"]
    found_winner = actual["terminal_winner"]
    found_tagged = actual["terminal_best_tagged"]
    # _describe uses 0.0 instead of original v0.8 cost in this trace,
    # therefore delta_cost_vs_frozen_v08 is intentionally ignored.
    def normalized(row: dict | None):
        if row is None:
            return None
        return {k: v for k, v in row.items()
                if k != "delta_cost_vs_frozen_v08"}
    if (actual["rescue_windows"] != expected_rescue
            or actual["displaced_slots"] != expected_displaced
            or normalized(found_winner) != normalized(expected_winner)
            or normalized(found_tagged) != normalized(expected_tagged)
            or actual["terminal_best_tagged_rank"] != reference[
                "best_protected_terminal_rank" if not adaptive
                else "best_protected_terminal_rank"]):
        raise ValueError("H0.14 traced policy diverged from canonical H0.13 replay")
    if adaptive and (
            actual["quota_released"] != reference["quota_released"]
            or actual["release_decision_window"] !=
            reference["release_decision_window"]):
        raise ValueError("adaptive release decision does not reproduce source")


def _first_window(windows: list[dict], predicate):
    return next((w["window_index"] for w in windows if predicate(w)), None)


def _window_lookup(windows: list[dict], wi: int | None):
    return next((row for row in windows if row["window_index"] == wi), None)


def explain_failure(full: dict, adaptive: dict, original_cost: float,
                    saved_case: dict) -> dict:
    """Describe observed loss and cumulative-cost decomposition, not causation."""
    a, b = full["snapshots"], adaptive["snapshots"]
    if ([r["window_index"] for r in a]
            != [r["window_index"] for r in b]):
        raise ValueError("different chunk segmentation in compared policies")
    join = full["window_index"]
    grace_end = join + OBSERVE_CHUNKS - 1
    if (adaptive["release_decision_window"] != grace_end
            or not adaptive["quota_released"]):
        raise ValueError("expected released counterexample")
    for full_row, adapted_row in zip(a[:OBSERVE_CHUNKS],
                                     b[:OBSERVE_CHUNKS]):
        if full_row["retained_keys"] != adapted_row["retained_keys"]:
            raise ValueError("full/adaptive traces differ before release")
    first_population_difference = next((
        x["window_index"] for x, y in zip(a, b)
        if x["retained_keys"] != y["retained_keys"]), None)
    first_tagged_absent = _first_window(
        b, lambda row: row["tagged_retained_count"] == 0)
    first_tagged_pruned = _first_window(
        b, lambda row: row["tagged_expanded_but_pruned"])
    first_tagged_rank1 = _first_window(
        a, lambda row: row["best_tagged_expanded_rank"] == 1)
    if first_population_difference is not None and (
            first_population_difference <= grace_end):
        raise ValueError("beam populations diverged before release")
    terminal_full = full["terminal_winner"]
    terminal_adaptive = adaptive["terminal_winner"]
    regret = terminal_adaptive["final_cost"] - terminal_full["final_cost"]
    if not regret > EPS:
        raise ValueError("known H0.13 regret was not reproduced")
    if (not terminal_full["is_from_protected_lineage"]
            or not terminal_full["final_cost"] < original_cost - EPS
            or not terminal_adaptive["final_cost"] >= original_cost - EPS):
        raise ValueError("known H0.13 missed improvement not reproduced")
    component_deltas = _terms_difference(
        terminal_adaptive["score_components"],
        terminal_full["score_components"])
    if abs(sum(component_deltas.values()) - regret) > 1e-6:
        raise ValueError("six-term objective decomposition inconsistent")
    lost = _window_lookup(b, first_tagged_absent)
    boundary = _window_lookup(b, grace_end)
    win_late = _window_lookup(a, first_tagged_rank1)
    return {
        "flag_pair": full["flag_pair"],
        "target_window": join,
        "grace_ends_at": grace_end,
        "first_post_release_window": grace_end+1,
        "first_different_retained_population_window":
            first_population_difference,
        "first_tagged_absent_adaptive_window": first_tagged_absent,
        "first_tagged_pruned_adaptive_window": first_tagged_pruned,
        "first_tagged_loss_reason": (
            "expanded_then_pruned_by_unprotected_top24"
            if lost is not None and lost["tagged_expanded_but_pruned"]
            else "no_tagged_feasible_expansion"
            if lost is not None else None),
        "first_tagged_rank1_full_window": first_tagged_rank1,
        "full_rank1_offset_from_join": (
            first_tagged_rank1-join if first_tagged_rank1 is not None else None),
        "release_window_population": boundary,
        "first_adaptive_tagged_absence_population": lost,
        "full_first_rank1_population": win_late,
        "terminal_v08_cost": round(original_cost, 9),
        "terminal_full_quota8_cost": terminal_full["final_cost"],
        "terminal_early_release_cost": terminal_adaptive["final_cost"],
        "regret_early_release_minus_full": round(regret, 9),
        "regret_six_component_deltas": component_deltas,
        "changed_events_full_vs_v08": terminal_full[
            "changed_event_assignment_count"],
        "changed_events_early_release_vs_v08": terminal_adaptive[
            "changed_event_assignment_count"],
        "full_new_quick_flag_locations": terminal_full["quick"]["added"],
        "adaptive_new_quick_flag_locations": terminal_adaptive["quick"]["added"],
        "verified_h013_counterexample": saved_case["counterexample"],
        "interpretation_limit": (
            "Differences in six cumulative cost components compare two "
            "different terminal paths; they do not isolate a causal "
            "contribution from any one chunk or prove a safe stopping bound."),
    }


def verify_h013(h013: dict, h012: dict, *,
                h012_path: Path) -> dict[int, dict]:
    if (h013.get("format") != H013_FORMAT
            or h013.get("source_h012_sha256") != sha256_file(h012_path)
            or h013.get("release_grace_chunks") != OBSERVE_CHUNKS
            or h013.get("beam_width") != BEAM_WIDTH
            or h013.get("quota") != QUOTA
            or h013.get("physical_motor_executed") is not False):
        raise ValueError("H0.13 source/provenance mismatch")
    check_h012_source(h012)
    for label, start, n in (
        ("discovery", DISCOVERY_START, DEFAULT_DISCOVERY),
        ("evaluation", EVALUATION_START, DEFAULT_EVALUATION),
    ):
        cohort = h013[label]
        if (cohort["trial_indices"] != list(range(start, start+n))
                or [r["trial_index"] for r in cohort["trials"]]
                != cohort["trial_indices"]
                or cohort["aggregate"] != summarize(cohort["trials"])
                or h013[label+"_falsification"]
                != evaluate_counterexamples(cohort)):
            raise ValueError(f"H0.13 {label} full source census mismatch")
        for row in cohort["trials"]:
            _, spec = generate(row["trial_index"])
            if row["generator"] != spec:
                raise ValueError(f"H0.13 {label} generator SHA mismatch")
    counterexamples = {}
    for trial in h013["discovery"]["trials"]:
        for case in trial["targets"]:
            if case["counterexample"]["rule_regret_after_release"]:
                key = trial["trial_index"]
                if key in counterexamples:
                    raise ValueError("multiple regret targets for one trial")
                counterexamples[key] = case
    if (tuple(sorted(counterexamples)) != TARGET_IDS
            or h013["discovery"]["aggregate"]["rule_regret_count"]
            != len(TARGET_IDS)
            or h013["evaluation"]["aggregate"]["rule_regret_count"] != 0):
        raise ValueError("H0.13 discovery/evaluation counterexample identities differ")
    return counterexamples


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h012-report", type=Path, required=True)
    p.add_argument("--h013-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("refusing to overwrite an H0.14 audit")
    h012 = json.loads(args.h012_report.read_text(encoding="utf-8"))
    h013 = json.loads(args.h013_report.read_text(encoding="utf-8"))
    cases = verify_h013(h013, h012, h012_path=args.h012_report)
    results = []
    for trial_id in TARGET_IDS:
        recorded = cases[trial_id]
        fresh = audit_one(trial_id)
        historical = next(
            row for row in h013["discovery"]["trials"]
            if row["trial_index"] == trial_id)
        if fresh != historical:
            raise ValueError("H0.13 counterexample trial no longer reproducible")
        matching = [
            row for row in fresh["targets"]
            if row["window_index"] == recorded["window_index"]]
        if len(matching) != 1 or matching[0] != recorded:
            raise ValueError("H0.13 target record differs on recomputation")

        events, spec = generate(trial_id)
        planner = GestureChunkPlannerV08()
        original = planner.plan(events)
        if not original.feasible:
            raise ValueError("frozen v0.8 synthetic plan no longer feasible")
        windows = planner._windows(events)
        chunks = [_chunk_record(chunk) for chunk in original.chunks]
        original_cost = original_precise_cost(
            events, original, windows, chunks)
        target = select_alternative(
            events, chunks, windows, recorded["window_index"])
        if target != recorded["target"]:
            raise ValueError("candidate selection was changed")
        opts = {
            "expected_cost": original.metrics["beam_cost"],
            "expected_evaluated": original.candidates_evaluated,
            "reference_final_cost": original_cost,
        }
        full_reference = trace_quota(
            events, chunks, target, quota=QUOTA, **opts)
        adaptive_reference = trace_adaptive(
            events, chunks, target, **opts)
        if classify(full_reference, adaptive_reference,
                    # The historical saved classification is reproduced
                    # by audit_one above; detailed trace is a separate replay.
                    {"summary": {"first_tagged_rank_one_window":
                        (recorded["counterexample"]["first_rank1_offset"]
                         + recorded["window_index"])
                        if recorded["counterexample"]["first_rank1_offset"]
                        is not None else None}}
                    ) != recorded["counterexample"]:
            raise ValueError("H0.13 original regret classification changed")

        full_trace = replay_with_snapshots(
            events, chunks, target, policy="full_quota8")
        adaptive_trace = replay_with_snapshots(
            events, chunks, target, policy="five_chunk_release")
        verify_replay(full_trace, full_reference, adaptive=False)
        verify_replay(adaptive_trace, adaptive_reference, adaptive=True)
        if (full_reference["beam_winner"]["final_cost"]
                != recorded["baseline"]["full_quota8_cost"]
                or adaptive_reference["beam_winner"]["final_cost"]
                != recorded["baseline"]["adaptive_cost"]):
            raise ValueError("H0.13 saved counterexample costs differ")
        explanation = explain_failure(
            full_trace, adaptive_trace, original_cost, recorded)
        results.append({
            "trial_index": trial_id,
            "generator": spec,
            "target": target,
            "summary": explanation,
            "full_quota8": full_trace,
            "five_chunk_release": adaptive_trace,
            "physical_motor_executed": False,
        })
    report = {
        "format": FORMAT,
        "source_h012_sha256": sha256_file(args.h012_report),
        "source_h013_sha256": sha256_file(args.h013_report),
        "target_trial_indices": list(TARGET_IDS),
        "target_count": len(results),
        "beam_width": BEAM_WIDTH,
        "protected_quota": QUOTA,
        "release_grace_chunks": OBSERVE_CHUNKS,
        "cases": results,
        "physical_motor_executed": False,
        "frozen_v08_changed": False,
        "release_rule_changed": False,
        "human_likeness_verified": False,
        "limitations": [
            "Only two post-hoc selected synthetic failures are diagnosed; not an unbiased success-rate estimate.",
            "The six cost terms are frozen, untrained cumulative heuristic components rather than anatomical/physical measurements.",
            "Best-of-beam snapshots may reflect different candidate ancestries at different windows; their differences are not causal estimates.",
            "The five-chunk rule is already falsified on this procedural generator, and no threshold is tuned here.",
            "Two policies have different search populations after release, so a late full-quota8 rank1 need not appear in adaptive search.",
            "No Train/Validation20/FINAL charts, physical Motor or new training are used.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({
        "target_count": len(results),
        "cases": [{
            "trial_index": row["trial_index"],
            "target_window": row["summary"]["target_window"],
            "first_population_difference_window":
                row["summary"]["first_different_retained_population_window"],
            "first_tagged_absent_window":
                row["summary"]["first_tagged_absent_adaptive_window"],
            "tagged_loss_reason": row["summary"]["first_tagged_loss_reason"],
            "full_first_tagged_rank1_window":
                row["summary"]["first_tagged_rank1_full_window"],
            "regret": row["summary"]["regret_early_release_minus_full"],
            "cost_component_differences": row[
                "summary"]["regret_six_component_deltas"],
        } for row in results],
        "saved": str(args.output),
        "physical_motor_executed": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
