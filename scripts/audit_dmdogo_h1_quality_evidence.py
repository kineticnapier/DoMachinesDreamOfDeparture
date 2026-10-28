from __future__ import annotations

"""DMDOGO H1.0: frozen objective versus physical/human evidence boundary.

Create a complete, provenance-pinned registry of ALL H0.6 Train
counterfactuals. This is a quality-validation *readiness* audit:
no demonstrated human preference, no physical alternate replay, no
blind A/B labels and no new planning/Motor/Timing/FINAL execution.
"""

import argparse
import hashlib
import json
from math import isfinite
from pathlib import Path
import re

from dmdogo_v08_provenance import FROZEN_HEAD

FORMAT = "dmdogo-h1-fingering-quality-evidence-register-v1"
H06_FORMAT = "dmdogo-v09-h06-improved-suffix-attribution-v1"
H06_BLOB_SHA = "67d8f63086f59c086c3724bc3284fe89178f85f2"
EXPECTED_CHARTS = 3
EXPECTED_EVENTS = 10231
EXPECTED_TARGETS = 22
EXPECTED_IMPROVEMENTS = 7
EPS = 1e-7
SUM_TOL = 2e-6
TERMS = (
    "gesture_preference",
    "hand_exposure",
    "axis_reuse_margin",
    "intra_gesture_hand_changes",
    "intra_gesture_depth_distance",
    "chunk_join_transition",
)
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
AXES = {f"{hand}_{depth}" for hand in ("left", "right")
        for depth in range(1, 5)}


def git_blob_sha(path: Path) -> str:
    raw = path.read_bytes()
    # The source is committed with LF; normalize only source newline
    # conversion on Windows, not JSON content or numeric values.
    raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha1(
        b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def number(value, field: str) -> float:
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def nonnegative_integer(value, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def score_terms(value, name: str) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(TERMS):
        raise ValueError(f"{name} does not contain exactly six frozen terms")
    return {k: number(value[k], f"{name}.{k}") for k in TERMS}


def valid_axes(value, field: str) -> None:
    if (not isinstance(value, list) or not value
            or any(not isinstance(row, list) or not row
                   or len(set(row)) != len(row)
                   or any(axis not in AXES for axis in row)
                   for row in value)):
        raise ValueError(f"invalid alternate frozen-axis grammar: {field}")


def check_flag_changes(flag: dict, label: str) -> None:
    if not isinstance(flag, dict):
        raise ValueError(f"missing {label} flag comparison")
    orig = nonnegative_integer(flag["original_count"], f"{label}.original")
    alt = nonnegative_integer(flag["alternate_count"], f"{label}.alternate")
    added = nonnegative_integer(flag["added_count"], f"{label}.added")
    removed = nonnegative_integer(flag["removed_count"], f"{label}.removed")
    if (added != len(flag["added"]) or removed != len(flag["removed"])
            or orig - removed + added != alt):
        raise ValueError(f"inconsistent {label} added/removed flags")


def _record(case: dict, item: dict) -> dict:
    chart_sha = case["chart_sha256"]
    a, b = item["flag_pair"]
    n = case["event_count"]
    if (type(a) is not int or type(b) is not int
            or a < 0 or b != a+1 or b >= n):
        raise ValueError("flag pair must be a valid adjacent event seam")
    wi = nonnegative_integer(item["window_index"], "window_index")
    valid_axes(item["nonquick_axes"], f"target:{chart_sha}:{a}:{b}")
    costs = item["cost"]
    orig = number(costs["original_v08_final"], "original final cost")
    alt = number(costs["forced_final"], "forced final cost")
    delta = number(costs["final_delta"], "final delta")
    improvement_amount = number(costs["improvement_amount"], "improvement amount")
    if abs((alt-orig)-delta) > SUM_TOL or abs(improvement_amount+delta) > SUM_TOL:
        raise ValueError("H0.6 reported final costs do not reconcile")
    o_terms = score_terms(costs["original_components"], "original components")
    f_terms = score_terms(costs["forced_components"], "forced components")
    deltas = score_terms(costs["six_component_deltas"], "component deltas")
    if (abs(sum(o_terms.values())-orig) > SUM_TOL
            or abs(sum(f_terms.values())-alt) > SUM_TOL
            or abs(sum(deltas.values())-delta) > SUM_TOL
            or any(abs((f_terms[k]-o_terms[k])-deltas[k]) > SUM_TOL
                   for k in TERMS)):
        raise ValueError("H0.6 six-term cost decomposition inconsistent")
    improved = delta < -EPS
    if item["improved_final_cost"] is not improved:
        raise ValueError("H0.6 improvement flag contradicts signed delta")

    effects = item["assignment_effects"]
    changed_chunks = nonnegative_integer(
        effects["changed_chunk_count"], "changed_chunk_count")
    changed_events = nonnegative_integer(
        effects["changed_event_assignment_count"], "changed_event_count")
    if (changed_chunks < 1 or changed_events < 1
            or changed_events > n
            or len(effects["changed_chunk_indices"]) != changed_chunks
            or len(set(effects["changed_chunk_indices"])) != changed_chunks
            or effects["first_changed_chunk"] != wi
            or min(effects["changed_chunk_indices"]) != wi):
        raise ValueError("H0.6 changed path ancestry is inconsistent")
    line = item["beam_lineage"]
    rank = nonnegative_integer(line["join_expanded_rank"], "expanded rank")
    if rank < 1 or line["loss_reason"] not in (
            "beam_pruning", "initial_beam_pruning"):
        raise ValueError("H0.6 beam lineage missing or invalid")
    if line["join_survived_width_24"] is not (rank <= 24):
        raise ValueError("H0.6 frozen width-24 admission mismatch")
    if (line["loss_reason"] == "initial_beam_pruning") is not (rank > 24):
        raise ValueError("H0.6 initial pruning explanation inconsistent")

    metrics = {}
    for label in ("original", "forced"):
        raw = item[f"{label}_assignment_metrics"]
        reuse_violations = nonnegative_integer(
            raw["abstract_reuse_under_80ms_count"],
            f"{label}.abstract_reuse_under_80ms_count")
        metrics[label] = {
            "hand_axis_uses": {
                hand: nonnegative_integer(
                    raw["hand_axis_uses"][hand], f"{label}.{hand}_axes")
                for hand in ("left", "right")},
            "minimum_axis_reuse_ms": (
                None if raw["minimum_axis_reuse_ms"] is None
                else number(raw["minimum_axis_reuse_ms"],
                            f"{label}.minimum_axis_reuse_ms")),
            "abstract_reuse_under_80ms_count": reuse_violations,
            "axis_reuse_under_100ms_count":
                nonnegative_integer(raw["axis_reuse_under_100ms_count"],
                                    f"{label}.reuse_under100"),
            "axis_reuse_under_150ms_count":
                nonnegative_integer(raw["axis_reuse_under_150ms_count"],
                                    f"{label}.reuse_under150"),
            "pure_hand_event_switches":
                nonnegative_integer(raw["pure_hand_event_switches"],
                                    f"{label}.pure_hand_event_switches"),
            "longest_single_hand_event_run":
                nonnegative_integer(raw["longest_single_hand_event_run"],
                                    f"{label}.longest_single_hand_run"),
        }
        if reuse_violations:
            raise ValueError("H0.6 abstract path violates 80ms frozen gate")
    for name, expected in item["assignment_metric_deltas"].items():
        if name in metrics["original"]:
            if (metrics["forced"][name] - metrics["original"][name]
                    != expected):
                raise ValueError(f"H0.6 {name} changed without corresponding delta")

    quick = item["flag_effects"]["quick"]
    rapid = item["flag_effects"]["rapid"]
    check_flag_changes(quick, "quick")
    check_flag_changes(rapid, "rapid")
    if item["flag_effects"]["flag_movement_is_identity_claim"] is not False:
        raise ValueError("a flag movement cannot be a human motion identity claim")

    # All candidates, positive and negative, are included. Never
    # present a local forced suffix as the actually executed alternate.
    return {
        "candidate_id": f"{chart_sha[:12]}-{a}-{b}",
        "chart": case["chart"],
        "chart_sha256": chart_sha,
        "event_pair": [a, b],
        "window_index": wi,
        "alternate_join_gesture_family": item["nonquick_gesture_family"],
        "alternate_join_axes": item["nonquick_axes"],
        "abstract_score": {
            "original_v08_final": orig,
            "forced_suffix_final": alt,
            "forced_minus_original": delta,
            "six_term_deltas": deltas,
            "lower_frozen_heuristic_cost": improved,
        },
        "assignment_difference": {
            "changed_event_count": changed_events,
            "changed_chunk_count": changed_chunks,
            "permanent_rejoin_chunk": effects["permanent_rejoin_chunk"],
        },
        "abstract_exploratory_proxies": {
            "original": metrics["original"],
            "forced_suffix": metrics["forced"],
            "quick_flags_added": quick["added_count"],
            "quick_flags_removed": quick["removed_count"],
            "rapid_flags_added": rapid["added_count"],
            "rapid_flags_removed": rapid["removed_count"],
            "target_quick_removed": item["flag_effects"][
                "target_quick_flag_removed"],
        },
        "beam_join_rank": rank,
        "beam_loss_reason": line["loss_reason"],
        "evidence": {
            "untrained_heuristic_comparison": "measured_in_saved_abstract_replay",
            "alternate_motor_feasibility": "not_tested",
            "human_finger_identity": "not_observed",
            "human_preference": "not_measured",
            "hud_only_observability": "not_tested",
            "blind_review_ready": False,
            "needs_full_ordered_both_candidate_assignments": True,
        },
    }


def compile_register(source: dict, *, input_blob_sha: str) -> dict:
    if (source.get("format") != H06_FORMAT
            or source.get("source_v08_commit") != FROZEN_HEAD
            or source.get("frozen_code_digest")
                != "066eb0c15786331130b46b29c112760dd781df77dcb89284e5031de7e59f9a43"
            or source.get("physical_motor_executed") is not False
            or source.get("human_likeness_verified") is not False
            or source.get("causal_effect_established") is not False
            or source.get("planner_motor_timing_runner_changed") is not False
            or source.get("hyperparameters_changed") is not False):
        raise ValueError("unrecognized or repurposed H0.6 source evidence")
    cases = source.get("cases")
    if (not isinstance(cases, list)
            or len(cases) != EXPECTED_CHARTS
            or source.get("case_count") != len(cases)):
        raise ValueError("H0.6 chart selection census changed")
    ids = set()
    chart_shas = set()
    candidates = []
    chart_counts = []
    event_total = 0
    for case in cases:
        chart_sha = case.get("chart_sha256")
        if (not isinstance(chart_sha, str) or not SHA_RE.fullmatch(chart_sha)
                or chart_sha in chart_shas
                or not isinstance(case.get("chart"), str)
                or not case["chart"]):
            raise ValueError("H0.6 chart identity duplicate/invalid")
        chart_shas.add(chart_sha)
        n = nonnegative_integer(case["event_count"], "event_count")
        event_total += n
        entries = case.get("evidence")
        if (not isinstance(entries, list)
                or case["targeted_count"] != len(entries)
                or case["improved_count"] != sum(
                    item["improved_final_cost"] is True for item in entries)):
            raise ValueError("H0.6 per-chart target/improvement census mismatch")
        for item in entries:
            record = _record(case, item)
            if record["candidate_id"] in ids:
                raise ValueError("repeated comparison candidate ID")
            ids.add(record["candidate_id"])
            candidates.append(record)
        chart_counts.append({
            "chart": case["chart"],
            "chart_sha256": chart_sha,
            "event_count": n,
            "targets": len(entries),
            "lower_cost_counterfactuals": case["improved_count"],
        })
    improvements = [
        x for x in candidates if x["abstract_score"]["lower_frozen_heuristic_cost"]]
    if (event_total != EXPECTED_EVENTS
            or source.get("event_count") != event_total
            or len(candidates) != EXPECTED_TARGETS
            or len(improvements) != EXPECTED_IMPROVEMENTS
            or source["aggregate"]["targeted_count"] != len(candidates)
            or source["aggregate"]["improved_final_cost_count"] != len(improvements)):
        raise ValueError("H0.6 saved total/positive target census changed")

    actual_term_sums = {
        term: sum(x["abstract_score"]["six_term_deltas"][term]
                  for x in improvements)
        for term in TERMS}
    for term in TERMS:
        if abs(actual_term_sums[term] -
               source["aggregate"]["improved_component_delta_sums"][term]) > SUM_TOL:
            raise ValueError("H0.6 aggregate cost term sums have changed")
    if sum(x["abstract_exploratory_proxies"]["target_quick_removed"]
           for x in improvements) != source["aggregate"][
               "improved_target_quick_removed_count"]:
        raise ValueError("H0.6 original target quick-flag aggregate changed")

    return {
        "format": FORMAT,
        "source_format": H06_FORMAT,
        "source_git_blob_sha1": input_blob_sha,
        "source_v08_frozen_commit": FROZEN_HEAD,
        "source_frozen_code_digest": source["frozen_code_digest"],
        "cohort": {
            "type": "previously_selected_three_train_charts_not_holdout",
            "charts": chart_counts,
            "chart_count": len(cases),
            "total_events": event_total,
            "independent_counterfactual_targets": len(candidates),
            "lower_frozen_heuristic_cost": len(improvements),
            "no_lower_frozen_heuristic_cost": len(candidates)-len(improvements),
        },
        "candidates": candidates,
        "quality_evidence_gate": {
            "low_objective_cost": "observed_for_seven_counterfactual_suffixes",
            "abstract_80ms_axis_reuse": "checked_in_saved_abstract_replays",
            "original_v08_physical": (
                "external_separately_verified_train_evidence_"
                "not_revalidated_by_this_h06_only_audit"),
            "alternate_physical_motor": "not_observed",
            "human_fingering_preference": "not_observed",
            "human_demonstration_alignment": "not_observed",
            "blind_human_ab": "not_review_ready_full_paths_not_present_in_h06",
            "overall_human_quality_verified": False,
            "motor_alternate_success_verified": False,
        },
        "next_evidence_requirements": [
            "Reconstruct exact full ordered frozen and counterfactual assignments under pinned chart/source checks, without merging independent interventions.",
            "Create unbiased A/B material (the whole changed portion and enough neighboring context); hide objective labels during any independent human review.",
            "Only independently aligned consenting human observations can support human-preference claims; old key/finger mappings alone cannot.",
            "Motor comparisons require separately verified alternate physical replays with exact chart/assignment/run provenance and explicit approval; original v0.8 success cannot transfer to alternates.",
            "Keep Validation20 and FINAL sealed, and preserve v0.8 Planner, Timing, Motor and runner.",
        ],
        "motor_executed": False,
        "human_demonstrations_used": False,
        "new_candidate_search": False,
        "training_performed": False,
        "validation20_or_final_accessed": False,
        "planner_changed": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h06-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("refusing to overwrite existing H1.0 evidence")
    blob = git_blob_sha(args.h06_report)
    if blob != H06_BLOB_SHA:
        raise ValueError("H0.6 source report differs from pinned original Git blob")
    original = json.loads(args.h06_report.read_text(encoding="utf-8-sig"))
    report = compile_register(original, input_blob_sha=blob)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2, ensure_ascii=False)
        out.write("\n")
    print(json.dumps({
        "cohort": report["cohort"],
        "quality_evidence_gate": report["quality_evidence_gate"],
        "saved": str(args.output),
        "motor_executed": False,
        "human_likeness_verified": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
