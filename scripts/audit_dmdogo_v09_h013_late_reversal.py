from __future__ import annotations

"""H0.13: historical late-reversal autopsy + disjoint-seed synthetic falsification.

Predeclared 5-chunk release rule unchanged. Reuse H0.12's frozen
generator and source-verified counterfactuals, with discovery seed IDs
starting at 4096 and evaluation IDs starting at 8192. No search
adaptation, threshold adjustment or filtering based on outcomes.
No original Train, Validation20, FINAL, Motor, or learning.
"""

import argparse
import hashlib
import json
from pathlib import Path

from dmdod.planning.gesture_chunk_planner_v08 import GestureChunkPlannerV08
from audit_dmdogo_v09_h04_beam_decisions import _chunk_record
from audit_dmdogo_v09_h08_protected_quota import (
    QUOTAS, BEAM_WIDTH, trace_quota,
)
from audit_dmdogo_v09_h09_quota_mechanism import (
    compare_exact_h08, trace_window_mechanism,
)
from audit_dmdogo_v09_h011_adaptive_release import (
    OBSERVE_CHUNKS, trace_adaptive,
)
from audit_dmdogo_v09_h012_synthetic_counterexamples import (
    FORMAT as H012_FORMAT, BASE_SEED, GENERATOR_VERSION, MODES,
    audit_one, classify, generate, original_precise_cost,
    select_alternative, summarize,
)

FORMAT = "dmdogo-v09-h013-late-reversal-independent-seeds-v1"
DISCOVERY_START = 4096
EVALUATION_START = 8192
DEFAULT_DISCOVERY = 32
DEFAULT_EVALUATION = 32
QUOTA = 8
EPS = 1e-7


def check_h012_source(source: dict) -> list[dict]:
    """Require complete original 32-trial census with consistent metadata."""
    if (source.get("format") != H012_FORMAT
            or source.get("generator_version") != GENERATOR_VERSION
            or source.get("base_seed") != BASE_SEED
            or source.get("tested_trial_count") != 32
            or source.get("generator_modes") != list(MODES)
            or source.get("beam_width") != BEAM_WIDTH
            or source.get("quota") != QUOTA
            or source.get("grace_chunks") != OBSERVE_CHUNKS
            or source.get("uses_saved_train_charts") is not False
            or source.get("physical_motor_executed") is not False):
        raise ValueError("H0.12 source protocol/identity mismatch")
    trials = source.get("trials", [])
    if [row["trial_index"] for row in trials] != list(range(32)):
        raise ValueError("H0.12 historical trial census is incomplete")
    for row in trials:
        _, spec = generate(row["trial_index"])
        if row["generator"] != spec:
            raise ValueError("H0.12 input generator hashes/seeds differ")
        if len(row["targets"]) + len(row["skipped_targets"]) not in (0, 1, 2):
            raise ValueError("H0.12 per-trial target census implausible")
    if source["aggregate"] != summarize(trials):
        raise ValueError("H0.12 saved aggregate not reproducible from trials")
    return trials


def _historical_late(trials: list[dict]) -> list[dict]:
    cases = []
    for trial in trials:
        for row in trial["targets"]:
            if not row["counterexample"]["released_then_late_rank1"]:
                continue
            cases.append({
                "trial_index": trial["trial_index"],
                "window_index": row["window_index"],
                "saved": row,
            })
    return cases


def _segment(records: list[dict], selected: int, *, radius: int = 2):
    return [
        {
            "offset_chunks": i,
            "window_index": item["window_index"],
            "event_range": item["event_range"],
            "expanded_rank": item["best_tagged_expanded_rank"],
            "natural_tagged_count": item["natural_tagged_count"],
            "kept_tagged_count": item["retained_tagged_count"],
            "tagged_minus_untagged_cost": item[
                "best_tagged_minus_best_untagged"],
            "forced_slot_replacements": item["replaced_slots"],
        }
        for i, item in enumerate(records)
        if abs(i - selected) <= radius
    ]


def autopsy(case: dict, recomputed_trial: dict) -> dict:
    """Reproduce exact saved historical case, then expose 5-chunk and late rank."""
    index = case["trial_index"]
    if recomputed_trial != case["recomputed_reference"]:
        raise ValueError("H0.12 historical trial failed deterministic reproduction")
    historical = case["saved"]
    events, spec = generate(index)
    plan = GestureChunkPlannerV08().plan(events)
    if not plan.feasible:
        raise ValueError("historically feasible original source is now infeasible")
    planner = GestureChunkPlannerV08()
    windows = planner._windows(events)
    chunks = [_chunk_record(chunk) for chunk in plan.chunks]
    original_cost = original_precise_cost(events, plan, windows, chunks)
    join = historical["window_index"]
    target = select_alternative(events, chunks, windows, join)
    if target != historical["target"]:
        raise ValueError("historical target changed on recomputation")
    common = {
        "expected_cost": plan.metrics["beam_cost"],
        "expected_evaluated": plan.candidates_evaluated,
        "reference_final_cost": original_cost,
    }
    full = trace_quota(events, chunks, target, quota=QUOTA, **common)
    diag = trace_window_mechanism(events, chunks, target, quota=QUOTA)
    compare_exact_h08(diag, full)
    adaptive = trace_adaptive(events, chunks, target, **common)
    if classify(full, adaptive, diag) != historical["counterexample"]:
        raise ValueError("historical late-rank1 diagnosis changed")
    records = diag["windows"]
    first_late = next(
        (i for i, row in enumerate(records)
         if row["best_tagged_expanded_rank"] == 1), None)
    if (first_late is None or first_late < OBSERVE_CHUNKS
            or not adaptive["quota_released"]):
        raise ValueError("not a release-then-late-rank1 case")
    return {
        "trial_index": index,
        "generator": spec,
        "window_index": join,
        "target": target,
        "first_rank1_offset": first_late,
        "release_decision_offset": OBSERVE_CHUNKS-1,
        "release_decision_window": join+OBSERVE_CHUNKS-1,
        "release_to_rank1_delay_chunks": first_late-OBSERVE_CHUNKS+1,
        "at_release": _segment(records, OBSERVE_CHUNKS-1, radius=2),
        "at_late_rank1": _segment(records, first_late, radius=3),
        "quota8_final_cost": full["beam_winner"]["final_cost"],
        "adaptive_final_cost": adaptive["beam_winner"]["final_cost"],
        "baseline_v08_final_cost": round(original_cost, 9),
        "full_protected_winner": full["beam_winner"]["is_from_protected_lineage"],
        "adaptive_protected_winner": adaptive[
            "beam_winner"]["is_from_protected_lineage"],
        "full_tagged_terminal_rank": full["best_protected_terminal_rank"],
        "adaptive_tagged_terminal_rank": adaptive[
            "best_protected_terminal_rank"],
        "regret_vs_full_quota8": round(
            adaptive["beam_winner"]["final_cost"] -
            full["beam_winner"]["final_cost"], 9),
        "full_rescue_windows": full["slot_replacement_count"],
        "adaptive_rescue_windows": adaptive["rescue_window_count"],
        "physical_motor_executed": False,
    }


def _cohort(trial_ids: list[int], *, name: str) -> dict:
    records = [audit_one(idx) for idx in trial_ids]
    if [r["trial_index"] for r in records] != trial_ids:
        raise ValueError(f"{name}: attempted synthetic IDs differ")
    for row in records:
        _, spec = generate(row["trial_index"])
        if row["generator"] != spec:
            raise ValueError(f"{name}: generator provenance mismatch")
        if not row["frozen_plan_feasible"]:
            continue
        # A run with insufficient windows or extinct tagged branches is
        # retained as skipped in the source H0.12 report, not erased.
        for candidate in row["targets"]:
            c = candidate["counterexample"]
            if c["quota_released"] is False and c["rule_regret"]:
                raise ValueError("unreleased full quota8 must have zero regret")
    return {
        "name": name,
        "trial_indices": trial_ids,
        "aggregate": summarize(records),
        "trials": records,
    }


def evaluate_counterexamples(cohort: dict) -> dict:
    """Categorize all apparent rule failures without re-tuning a threshold."""
    targets = [
        (trial, candidate)
        for trial in cohort["trials"]
        for candidate in trial["targets"]
    ]
    regrets = [{
        "trial_index": trial["trial_index"],
        "generator": trial["generator"],
        "window_index": row["window_index"],
        "candidate_expanded_rank": row["target"][
            "nonquick_join_expanded_rank"],
        "result": row["counterexample"],
        "costs": row["baseline"],
        "first_five": row["diagnostic"]["first_five_chunks"],
        "last_five": row["diagnostic"]["last_five_chunks"],
    } for trial, row in targets if row["counterexample"]["rule_regret"]]
    late = [{
        "trial_index": trial["trial_index"],
        "window_index": row["window_index"],
        "first_rank1_offset": row["counterexample"]["first_rank1_offset"],
        "regret": row["counterexample"]["rule_regret"],
        "full_quota8_protected_winner": row[
            "counterexample"]["full_quota8_winner_protected"],
    } for trial, row in targets
        if row["counterexample"]["released_then_late_rank1"]]
    return {
        "counterexamples": regrets,
        "late_rank1_cases": late,
        "regret_count": len(regrets),
        "late_rank1_count": len(late),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h012-report", type=Path, required=True)
    p.add_argument("--discovery-trials", type=int, default=DEFAULT_DISCOVERY)
    p.add_argument("--evaluation-trials", type=int, default=DEFAULT_EVALUATION)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (not 1 <= a.discovery_trials <= 64
            or not 1 <= a.evaluation_trials <= 64):
        p.error("discovery/evaluation trials must each be 1..64")
    if a.output.exists():
        p.error("refusing to overwrite earlier H0.13 findings")
    if (GestureChunkPlannerV08().beam_width != 24 or QUOTA != 8
            or 8 not in QUOTAS or OBSERVE_CHUNKS != 5):
        raise ValueError("frozen search or original H0.11 rule changed")
    h012 = json.loads(a.h012_report.read_text(encoding="utf-8"))
    originals = check_h012_source(h012)
    historical = _historical_late(originals)
    if len(historical) != h012["aggregate"]["released_then_late_rank1_count"]:
        raise ValueError("H0.12 historical late reversal count mismatched")
    cache = {}
    old_diagnostics = []
    for item in historical:
        index = item["trial_index"]
        if index not in cache:
            cache[index] = audit_one(index)
            if cache[index] != originals[index]:
                raise ValueError("recomputed H0.12 late-reversal trial differs")
        old_diagnostics.append(autopsy(
            {**item, "recomputed_reference": originals[index]}, cache[index]))
    discovery_ids = list(range(
        DISCOVERY_START, DISCOVERY_START+a.discovery_trials))
    evaluation_ids = list(range(
        EVALUATION_START, EVALUATION_START+a.evaluation_trials))
    if (set(discovery_ids) & set(evaluation_ids)
            or set(discovery_ids) & set(range(32))
            or set(evaluation_ids) & set(range(32))):
        raise ValueError("synthetic discovery/evaluation sources overlap")
    discovery = _cohort(discovery_ids, name="discovery")
    evaluation = _cohort(evaluation_ids, name="evaluation")
    out_d = evaluate_counterexamples(discovery)
    out_e = evaluate_counterexamples(evaluation)
    if (out_d["regret_count"] != discovery["aggregate"]["rule_regret_count"]
            or out_e["regret_count"] != evaluation["aggregate"]["rule_regret_count"]
            or out_d["late_rank1_count"] != discovery[
                "aggregate"]["released_then_late_rank1_count"]
            or out_e["late_rank1_count"] != evaluation[
                "aggregate"]["released_then_late_rank1_count"]):
        raise ValueError("cohort summary/detail disagreement")
    report = {
        "format": FORMAT,
        "source_h012_sha256": hashlib.sha256(
            a.h012_report.read_bytes()).hexdigest(),
        "base_seed": BASE_SEED,
        "generator_version": GENERATOR_VERSION,
        "beam_width": BEAM_WIDTH,
        "quota": QUOTA,
        "release_grace_chunks": OBSERVE_CHUNKS,
        "generator_protocol": (
            "Unchanged H0.12 generator; historical IDs 0..31, "
            "discovery 4096.., evaluation 8192.., disjoint fixed IDs."
        ),
        "evaluation_protocol": (
            "No threshold tuning, candidate retargeting, stopping "
            "adaptation, or outcome-conditioned case omission."
        ),
        "historical_late_reversal_count": len(old_diagnostics),
        "historical_late_reversal_autopsies": old_diagnostics,
        "discovery": discovery,
        "evaluation": evaluation,
        "discovery_falsification": out_d,
        "evaluation_falsification": out_e,
        "physical_motor_executed": False,
        "v08_frozen_changed": False,
        "human_likeness_verified": False,
        "limitations": [
            "Historical H0.12 cases are descriptive, not independent new evidence.",
            "Discovery/evaluation have independent deterministic seeds, but both use the same synthetic generator and candidate preselection.",
            "Fixed five-chunk release was chosen after H0.9 and is not proven safe.",
            "An actual regret is higher final frozen-heuristic cost under release, not a physical Motor failure.",
            "A late rank reversal in full quota8 does not prove release loses a final improvement.",
            "Reported displaced slots do not measure wall-clock runtime improvements.",
            "Individual candidate trials must not be combined as a single playable chart.",
            "No real Train chart, Validation20, FINAL, learned model, or physical Motor was run.",
        ],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({
        "historical_late_reversal_count": len(old_diagnostics),
        "discovery": discovery["aggregate"],
        "evaluation": evaluation["aggregate"],
        "counterexample_summary": {
            "discovery": out_d["regret_count"],
            "evaluation": out_e["regret_count"],
            "evaluation_late_rank1": out_e["late_rank1_count"],
        },
        "saved": str(a.output),
        "physical_motor_executed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
