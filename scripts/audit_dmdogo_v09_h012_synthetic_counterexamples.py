from __future__ import annotations

"""H0.12: deterministic synthetic falsification of H0.11's five-chunk release.

No H0.1–H0.11 training chart, validation chart, FINAL, physical Motor or
learned timing is loaded. Candidates are selected before outcomes by the
frozen-beam expanded rank at two prespecified chunk locations per chart.
Report all generated trials, infeasible attempts and null counterexample
counts; never silently optimize the five-chunk threshold on test cases.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import random
from pathlib import Path
from dmdogo_h0_source_provenance import capture_audit_source_provenance
from statistics import mean

from dmdod.planning.gesture_chunk_planner_v08 import (
    GestureChunkPlannerV08, GestureEvent, _candidate_gestures,
)
from audit_dmdogo_v09_h04_beam_decisions import _chunk_record
from audit_dmdogo_v09_h05_nonquick_suffix import _next, _seed_state
from audit_dmdogo_v09_h08_protected_quota import (
    BEAM_WIDTH, QUOTAS, trace_quota,
)
from audit_dmdogo_v09_h09_quota_mechanism import (
    trace_window_mechanism, compare_exact_h08,
)
from audit_dmdogo_v09_h011_adaptive_release import (
    QUOTA, OBSERVE_CHUNKS, trace_adaptive,
)

FORMAT = "dmdogo-v09-h012-synthetic-counterexamples-v1"
GENERATOR_VERSION = "four-phase-timed-chunks-v1"
BASE_SEED = 20261028
DEFAULT_TRIALS = 32
EPS = 1e-7
MODES = ("constant", "accelerate", "alternating", "regime_shift")


def generate(trial_index: int) -> tuple[tuple[GestureEvent, ...], dict]:
    """Deterministic, declared generators, independent of planner outcomes.

    Each trial has 64/80/96 presses, four within-chart tempo regimes and
    optional isolated 2-key chord insertions; no real chart is sampled.
    Values are generated without consulting the v0.8 planner or labels.
    """
    if not isinstance(trial_index, int) or trial_index < 0:
        raise ValueError("nonnegative trial index required")
    rng = random.Random(BASE_SEED + 7919 * trial_index)
    mode = MODES[trial_index % len(MODES)]
    n = (64, 80, 96)[(trial_index // len(MODES)) % 3]
    levels = (0.026, 0.035, 0.047, 0.062, 0.085, 0.11)
    base = levels[rng.randrange(len(levels))]
    time = 1.0
    rows = []
    for i in range(n):
        phase = 4*i // n
        if mode == "constant":
            interval = base
        elif mode == "accelerate":
            interval = base * (1.6, 1.3, 1.0, 0.8)[phase]
        elif mode == "alternating":
            interval = base * ((0.85, 1.6, 0.9, 1.55)[phase])
        else:
            interval = base * (1.0, 1.0, 0.55, 1.35)[phase]
        if i:
            # A small number of salient phrase boundaries; these are
            # not tied to target seams or observed successes.
            if i % 16 == 0:
                time += 0.19 + (trial_index % 3) * 0.06
            else:
                jitter = 1.0 + 0.035*(rng.randrange(5)-2)
                time += interval * jitter
        chord_size = 2 if (trial_index % 6 == 0 and i % 29 == 14) else 1
        rows.append(GestureEvent(i, round(time, 9), chord_size))
    config = {
        "seed": BASE_SEED + 7919 * trial_index,
        "mode": mode,
        "event_count": n,
        "base_ioi_s": base,
        "chord_event_count": sum(e.chord_size > 1 for e in rows),
        "events_sha256": hashlib.sha256(json.dumps([
            asdict(e) for e in rows], sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(),
    }
    return tuple(rows), config


def selected_seams(window_count: int) -> tuple[int, ...]:
    """Two predeclared positions; require at least five future chunks."""
    if window_count < 9:
        return ()
    choices = (1, max(2, window_count // 2 - 5))
    return tuple(dict.fromkeys(j for j in choices
                               if 0 < j and j + OBSERVE_CHUNKS < window_count))


def original_precise_cost(events, plan, windows, chunks) -> float:
    if not plan.feasible or len(chunks) != len(windows):
        raise ValueError("synthetic frozen plan must be feasible")
    branch = _seed_state(events[0].due_s)
    for wi, window in enumerate(windows):
        matches = [
            candidate for candidate in _candidate_gestures(window)
            if candidate.family == chunks[wi]["gesture_family"]
            and [list(row) for row in candidate.axes] == chunks[wi]["axes"]
        ]
        if len(matches) != 1:
            raise ValueError("frozen chunk candidate not unique")
        branch = _next(
            branch, window, matches[0],
            windows[wi-1].events[-1].due_s if wi else None,
            frozen_candidate=True)
        if branch is None:
            raise ValueError("frozen v0.8 path became reuse-infeasible")
    if round(branch.score, 6) != plan.metrics["beam_cost"]:
        raise ValueError("frozen v0.8 precision check failed")
    return branch.score


def select_alternative(events, chunks, windows, join: int):
    """Prefer best outside top24, otherwise best non-original alternative.

    The selection uses ONLY the original frozen prefix, frozen candidate
    grammar and cost/rank *at the targeted join*, not any future outcome.
    It is not restricted to a quick-jump flag or human non-quick pattern;
    this is deliberately a broader falsification of the stop rule.
    """
    states = [_seed_state(events[0].due_s)]
    if not 0 < join < len(windows):
        raise ValueError("invalid synthetic join")
    for wi, window in enumerate(windows[:join+1]):
        axes = tuple(tuple(x) for x in chunks[wi]["axes"])
        family = chunks[wi]["gesture_family"]
        prev_due = windows[wi-1].events[-1].due_s if wi else None
        expanded = []
        for parent in states:
            for ci, candidate in enumerate(_candidate_gestures(window)):
                nxt = _next(
                    parent, window, candidate, prev_due,
                    frozen_candidate=(
                        candidate.axes == axes and candidate.family == family))
                if nxt is None:
                    continue
                expanded.append((nxt, parent.frozen_prefix, ci, candidate))
        if not expanded:
            raise ValueError("frozen prefix has no feasible expansions")
        expanded.sort(key=lambda x: x[0].score)
        if wi < join:
            states = [x[0] for x in expanded[:BEAM_WIDTH]]
            if sum(x.frozen_prefix for x in states) != 1:
                raise ValueError("original path diverged before synthetic join")
            continue
        candidates = [
            (rank, item) for rank, item in enumerate(expanded, 1)
            if item[1]
            and (item[3].family != family
                 or item[3].axes != axes)
        ]
        if not candidates:
            return None
        preferred = next(((rank, row) for rank, row in candidates
                          if rank > BEAM_WIDTH), candidates[0])
        rank, (_, _, ci, candidate) = preferred
        return {
            "flag_pair": [windows[join-1].events[-1].event_index,
                          window.events[0].event_index],
            "window_index": join,
            "nonquick_candidate_index": ci,
            "nonquick_join_expanded_rank": rank,
            "nonquick_axes": [list(x) for x in candidate.axes],
            "gesture_family": candidate.family,
            "selection": (
                "first_rank_outside_top24" if rank > BEAM_WIDTH
                else "first_nonoriginal_in_original_predecessor"),
        }
    raise RuntimeError("unreachable")


def classify(full: dict, adaptive: dict, diagnostic: dict) -> dict:
    """Distinguish late reversals from actual missed final improvements."""
    first = diagnostic["summary"]["first_tagged_rank_one_window"]
    join = full["window_index"]
    late = first is not None and first - join >= OBSERVE_CHUNKS
    released = adaptive["quota_released"]
    full_cost = full["beam_winner"]["final_cost"]
    adaptive_cost = adaptive["beam_winner"]["final_cost"]
    costs_difference = round(adaptive_cost - full_cost, 9)
    full_protected_winner = full["beam_winner"]["is_from_protected_lineage"]
    return {
        "late_rank1_after_grace": late,
        "ever_rank1": first is not None,
        "first_rank1_offset": first - join if first is not None else None,
        "quota_released": released,
        "full_quota8_winner_protected": full_protected_winner,
        "adaptive_winner_protected": adaptive[
            "beam_winner"]["is_from_protected_lineage"],
        "full_quota8_lower_cost_than_v08": full[
            "beam_winner"]["delta_cost_vs_frozen_v08"] < -EPS,
        "adaptive_lower_cost_than_v08": adaptive[
            "beam_winner"]["delta_cost_vs_frozen_v08"] < -EPS,
        "adaptive_minus_full_final_cost": costs_difference,
        "rule_regret": costs_difference > EPS,
        "rule_regret_after_release": released and costs_difference > EPS,
        "missed_improvement_vs_v08": (
            released and
            full["beam_winner"]["delta_cost_vs_frozen_v08"] < -EPS
            and adaptive["beam_winner"]["delta_cost_vs_frozen_v08"] >= -EPS),
        "released_then_late_rank1": released and late,
    }


def audit_one(trial: int) -> dict:
    events, spec = generate(trial)
    planner = GestureChunkPlannerV08()
    plan = planner.plan(events)
    outcome = {
        "trial_index": trial, "generator": spec,
        "frozen_plan_feasible": plan.feasible,
        "frozen_failure_reason": plan.failure_reason,
        "chunk_count": plan.metrics.get("chunk_count"),
        "targets": [],
        "skipped_targets": [],
    }
    if not plan.feasible:
        return outcome
    windows = planner._windows(events)
    chunks = [_chunk_record(c) for c in plan.chunks]
    original_score = original_precise_cost(events, plan, windows, chunks)
    outcome["frozen_exact_cost"] = round(original_score, 9)
    seams = selected_seams(len(windows))
    if not seams:
        outcome["skipped_targets"].append({
            "reason": "too_few_chunks_with_five_future_windows"})
        return outcome
    for join in seams:
        target = select_alternative(events, chunks, windows, join)
        if target is None:
            outcome["skipped_targets"].append({
                "window_index": join, "reason": "no_distinct_feasible_alternative"})
            continue
        full = trace_quota(
            events, chunks, target, quota=QUOTA,
            expected_cost=plan.metrics["beam_cost"],
            expected_evaluated=plan.candidates_evaluated,
            reference_final_cost=original_score)
        if not full["protected_descendants_reached_terminal"]:
            # The H0.9 diagnostic requires a tagged terminal survivor.
            # Count this predeclared untestable case explicitly instead of
            # silently excluding or misclassifying it as a successful rule.
            outcome["skipped_targets"].append({
                "window_index": join,
                "reason": "full_quota8_protected_lineage_extinct",
                "candidate_rank": target["nonquick_join_expanded_rank"],
            })
            continue
        diagnostic = trace_window_mechanism(
            events, chunks, target, quota=QUOTA)
        compare_exact_h08(diagnostic, full)
        adaptive = trace_adaptive(
            events, chunks, target,
            expected_cost=plan.metrics["beam_cost"],
            expected_evaluated=plan.candidates_evaluated,
            reference_final_cost=original_score)
        if adaptive["admission_expanded_rank"] != full["admission_expanded_rank"]:
            raise ValueError("synthetic source admission rank differs")
        first5 = diagnostic["windows"][:OBSERVE_CHUNKS]
        witnessed_rank1 = any(x["best_tagged_expanded_rank"] == 1
                               for x in first5)
        if adaptive["ever_protected_rank1_before_decision"] != witnessed_rank1:
            raise ValueError("synthetic online decision differs from diagnostic")
        if not adaptive["quota_released"]:
            if (adaptive["beam_winner"] != full["beam_winner"]
                    or adaptive["best_protected_terminal"]
                    != full["best_protected_terminal"]
                    or adaptive["rescue_windows"] != full["slot_replacement_windows"]
                    or adaptive["displaced_untagged_slots"]
                    != full["total_displaced_nonprotected_slots"]):
                raise ValueError("unreleased full quota8 case diverged")
        outcome["targets"].append({
            "window_index": join,
            "target": target,
            "counterexample": classify(full, adaptive, diagnostic),
            "baseline": {
                "v08_cost": round(original_score, 9),
                "full_quota8_cost": full["beam_winner"]["final_cost"],
                "adaptive_cost": adaptive["beam_winner"]["final_cost"],
                "full_quota8_rescue_windows": full["slot_replacement_count"],
                "adaptive_rescue_windows": adaptive["rescue_window_count"],
                "full_quota8_displaced_slots":
                    full["total_displaced_nonprotected_slots"],
                "adaptive_displaced_slots":
                    adaptive["displaced_untagged_slots"],
                "full_terminal_tagged_survives":
                    full["protected_descendants_reached_terminal"],
                "adaptive_terminal_tagged_survives":
                    adaptive["protected_terminal_survived"],
                "new_quick_full": full["beam_winner"]["quick"]["added_count"],
                "new_quick_adaptive": adaptive["beam_winner"]["quick"]["added_count"],
                "new_rapid_full": full["beam_winner"]["rapid"]["added_count"],
                "new_rapid_adaptive": adaptive["beam_winner"]["rapid"]["added_count"],
            },
            "diagnostic": {
                "first_tagged_rank1_window": diagnostic[
                    "summary"]["first_tagged_rank_one_window"],
                "first_natural_quota_window": diagnostic[
                    "summary"]["first_natural_quota_window"],
                "quota8_rescue_windows": diagnostic["summary"]["rescue_windows"],
                "first_five_chunks": [{
                    "window_index": x["window_index"],
                    "best_tagged_expanded_rank": x["best_tagged_expanded_rank"],
                    "best_tagged_minus_best_untagged":
                        x["best_tagged_minus_best_untagged"],
                    "natural_tagged_count": x["natural_tagged_count"],
                    "replaced_slots": x["replaced_slots"],
                } for x in diagnostic["windows"][:OBSERVE_CHUNKS]],
                "last_five_chunks": [{
                    "window_index": x["window_index"],
                    "best_tagged_expanded_rank": x["best_tagged_expanded_rank"],
                    "best_tagged_minus_best_untagged":
                        x["best_tagged_minus_best_untagged"],
                } for x in diagnostic["windows"][-OBSERVE_CHUNKS:]],
            },
        })
    return outcome


def summarize(trials: list[dict]) -> dict:
    targets = [t for trial in trials for t in trial["targets"]]
    reasons = {}
    for trial in trials:
        for skipped in trial["skipped_targets"]:
            reason = skipped["reason"]
            reasons[reason] = reasons.get(reason, 0) + 1
        if not trial["frozen_plan_feasible"]:
            reason = trial["frozen_failure_reason"] or "unknown"
            reasons["frozen_" + reason] = reasons.get("frozen_" + reason, 0) + 1
    outcomes = [t["counterexample"] for t in targets]
    original = sum(t["baseline"]["full_quota8_displaced_slots"] for t in targets)
    adapted = sum(t["baseline"]["adaptive_displaced_slots"] for t in targets)
    return {
        "synthetic_trials_generated": len(trials),
        "feasible_frozen_trials": sum(
            trial["frozen_plan_feasible"] for trial in trials),
        "targets_evaluated": len(targets),
        "skipped_reasons": reasons,
        "released_count": sum(x["quota_released"] for x in outcomes),
        "late_rank1_after_grace_count": sum(
            x["late_rank1_after_grace"] for x in outcomes),
        "released_then_late_rank1_count": sum(
            x["released_then_late_rank1"] for x in outcomes),
        "full_quota8_protected_winner_count": sum(
            x["full_quota8_winner_protected"] for x in outcomes),
        "rule_regret_count": sum(x["rule_regret"] for x in outcomes),
        "rule_regret_after_release_count": sum(
            x["rule_regret_after_release"] for x in outcomes),
        "missed_improvement_vs_v08_count": sum(
            x["missed_improvement_vs_v08"] for x in outcomes),
        "full_quota8_displaced_slots": original,
        "adaptive_displaced_slots": adapted,
        "avoided_displacements": original - adapted,
        "mean_regret_over_all_targets": round(mean(
            [x["adaptive_minus_full_final_cost"] for x in outcomes]
            or [0.0]), 9),
        "counterexample_trial_indices": sorted(set(
            trial["trial_index"] for trial in trials
            for t in trial["targets"] if t["counterexample"]["rule_regret"])),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.trials <= 128:
        p.error("--trials must be between 1 and 128")
    if args.output.exists():
        p.error("refusing to overwrite existing H0.12 output")
    if (GestureChunkPlannerV08().beam_width != BEAM_WIDTH
            or QUOTA != 8 or OBSERVE_CHUNKS != 5):
        raise ValueError("frozen beam width or stop rule changed")
    trials = [audit_one(i) for i in range(args.trials)]
    summary = summarize(trials)
    if summary["targets_evaluated"] == 0:
        raise ValueError("no viable synthetic counterfactuals; refusing empty result")
    report = {
        "format": FORMAT,
        "audit_source_provenance": capture_audit_source_provenance(Path(__file__).resolve().parents[1]),
        "generator_version": GENERATOR_VERSION,
        "base_seed": BASE_SEED,
        "tested_trial_count": args.trials,
        "generator_modes": list(MODES),
        "beam_width": BEAM_WIDTH, "quota": QUOTA,
        "grace_chunks": OBSERVE_CHUNKS,
        "target_seams": (
            "chunk 1 and chunk max(2, window_count // 2 - 5), when "
            "at least five future chunks remain"),
        "candidate_preselection": (
            "first feasible non-original frozen-predecessor candidate "
            "outside natural top24 by expanded score, else first "
            "feasible non-original candidate"),
        "aggregate": summary,
        "trials": trials,
        "uses_saved_train_charts": False,
        "physical_motor_executed": False,
        "v08_frozen_code_changed": False,
        "human_likeness_verified": False,
        "limitations": [
            "Synthetic events are not independent real ADOFAI charts or human demonstrations.",
            "The five-chunk rule was derived from H0.9 and is tested as-is without threshold tuning.",
            "Not all alternative seam candidates are genuine non-quick fingering corrections; target selection deliberately tests a broader failure mode.",
            "No counterexample in a finite search cannot establish the rule is globally safe.",
            "A late tagged rank1 is not sufficient to establish a better final protected winner.",
            "Candidate selection uses only the original frozen predecessor and local join costs; outcome-based sampling is prohibited.",
            "Counts of displaced slots are not CPU timing improvements or combined chart effects.",
            "Only frozen heuristic, abstract 80ms reuse, no physical Motor, Validation20, FINAL, or new training.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({
        "tested_trial_count": report["tested_trial_count"],
        "aggregate": summary, "saved": str(args.output),
        "physical_motor_executed": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
