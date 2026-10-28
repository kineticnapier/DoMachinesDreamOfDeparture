from __future__ import annotations

"""H0.17: state-aware admissible 1-/2-chunk lookahead, audit only.

For EVERY currently retained tagged state, enumerate all feasible
continuations for one/two chunks without beam pruning. Combine exact
prefix costs with the H0.16 globally admissible cold suffix. Take
minimum over all tagged states, then max with the H0.16 bound.
This only certifies no tagged descendant can beat the completed,
offline v0.8 incumbent. It does NOT validate a changed beam search
or human fingering. No original Train, Validation20, FINAL or Motor.
"""

import argparse
import json
from pathlib import Path

from dmdod.planning.gesture_chunk_planner_v08 import (
    GestureChunkPlannerV08, _candidate_gestures,
)
from audit_dmdogo_v09_h04_beam_decisions import _chunk_record
from audit_dmdogo_v09_h05_nonquick_suffix import _next, _seed_state
from audit_dmdogo_v09_h07_shadow_lane import (
    BeamPath, Node, Proposal,
    _chunk_record_from,
)
from audit_dmdogo_v09_h08_protected_quota import (
    BEAM_WIDTH, TARGET_TAG, choose_quota,
)
from audit_dmdogo_v09_h014_regret_autopsy import (
    TARGET_IDS, Tracked, replay_with_snapshots, verify_h013,
)
from audit_dmdogo_v09_h015_admissible_suffix_bound import (
    verify_frozen_cost_source, certificate, EPS,
)
from audit_dmdogo_v09_h016_cold_block_lower_bounds import (
    FORMAT as H016_FORMAT, QUOTA,
    cold_suffix_bounds,
)
from audit_dmdogo_v09_h012_synthetic_counterexamples import (
    audit_one, generate, original_precise_cost, select_alternative,
)
from dmdogo_v08_provenance import sha256_file

FORMAT = "dmdogo-v09-h017-state-aware-lower-bound-v1"
LOOKAHEAD_MAX = 2
CHECK_CHUNKS = 5
TOL = 1e-7


def tagged_grace_states(events, frozen_chunks: list[dict], target: dict
                        ) -> list[dict]:
    """Independently reproduce original quota8 and save full tagged states.

    Short replay ends after fifth join-inclusive window; comparison
    against the independently obtained H0.14 full-trajectory snapshots
    is required by the audit before any report is written.
    """
    windows = GestureChunkPlannerV08()._windows(events)
    join = target["window_index"]
    if (not 0 < join < len(windows)
            or len(frozen_chunks) != len(windows)
            or join + CHECK_CHUNKS > len(windows)
            or target["flag_pair"] != [
                windows[join-1].events[-1].event_index,
                windows[join].events[0].event_index]):
        raise ValueError("invalid/missing H0.17 target seam or grace period")
    states = [Tracked(BeamPath(_seed_state(events[0].due_s), None), ())]
    collected = []
    for wi in range(join + CHECK_CHUNKS):
        window = windows[wi]
        prev_due = windows[wi-1].events[-1].due_s if wi else None
        axes = tuple(tuple(x) for x in frozen_chunks[wi]["axes"])
        family = frozen_chunks[wi]["gesture_family"]
        expanded = []
        keys = {}
        for parent_rank, parent in enumerate(states, 1):
            for ci, candidate in enumerate(_candidate_gestures(window)):
                nxt = _next(
                    parent.state.state, window, candidate, prev_due,
                    frozen_candidate=(
                        candidate.axes == axes and candidate.family == family))
                if nxt is None:
                    continue
                if (wi == join and parent.state.state.frozen_prefix
                        and ci == target["nonquick_candidate_index"]):
                    nxt.tags = frozenset({TARGET_TAG})
                proposal = Proposal(
                    nxt, parent.state.node,
                    _chunk_record_from(window, candidate),
                    parent_rank, ci)
                expanded.append(proposal)
                keys[id(proposal)] = parent.key + (ci,)
        if not expanded:
            raise ValueError("empty frozen/target beam expansion")
        expanded.sort(key=lambda item: item.state.score)
        if wi < join:
            keep = expanded[:BEAM_WIDTH]
            if sum(p.state.frozen_prefix for p in keep) != 1:
                raise ValueError("frozen prefix lost before H0.17 join")
        else:
            if wi == join:
                found = [(i, p) for i, p in enumerate(expanded, 1)
                         if p.protected]
                if len(found) != 1:
                    raise ValueError("tagged candidate not unique at join")
                rank, picked = found[0]
                if (rank != target["nonquick_join_expanded_rank"]
                        or picked.candidate_index
                        != target["nonquick_candidate_index"]
                        or picked.chunk["axes"] != target["nonquick_axes"]):
                    raise ValueError("H0.17 target admission mismatch")
            keep, _ = choose_quota(expanded, quota=QUOTA)
            tagged = [p.state for p in keep if p.protected]
            collected.append({
                "window_index": wi,
                "tagged": tagged,
                "tagged_count": len(tagged),
                "best_tagged_score": (
                    round(min(t.score for t in tagged), 9) if tagged else None),
                "retained_keys": [list(keys[id(p)]) for p in keep],
            })
        states = [Tracked(
            BeamPath(p.state, Node(p.parent, p.chunk)),
            keys[id(p)]) for p in keep]
    return collected


def exact_lookahead_bound(tagged_states, windows, *, next_window: int,
                          lookahead: int,
                          cold_suffix: tuple[float, ...]) -> dict:
    """All tagged parents, all feasible 1-/2-window suffixes. No beam cut.

    Each exact continuation begins with the ACTUAL state: both hand
    exposure accumulators, last axis, and all axis-use timestamps.
    After k windows, all remaining history is relaxed using the
    global, state-independent admissible H0.16 suffix.
    """
    if (lookahead not in (1, 2)
            or not 0 <= next_window <= len(windows)
            or len(cold_suffix) != len(windows)+1):
        raise ValueError("invalid state-aware horizon/suffix")
    if not tagged_states:
        return {
            "applicable": False, "feasible_continuations": 0,
            "optimistic_total_cost": None,
            "candidate_evaluations": 0,
            "horizon_chunks": min(lookahead, len(windows)-next_window),
        }
    steps = min(lookahead, len(windows)-next_window)
    if steps == 0:
        return {
            "applicable": True,
            "feasible_continuations": len(tagged_states),
            "optimistic_total_cost": min(s.score for s in tagged_states),
            "candidate_evaluations": 0,
            "horizon_chunks": 0,
        }
    states = list(tagged_states)
    evaluations = 0
    for wi in range(next_window, next_window + steps):
        window = windows[wi]
        previous_due = windows[wi-1].events[-1].due_s if wi else None
        candidates = _candidate_gestures(window)
        expanded = []
        for parent in states:
            for candidate in candidates:
                evaluations += 1
                child = _next(parent, window, candidate, previous_due,
                              frozen_candidate=False)
                if child is not None:
                    expanded.append(child)
        if not expanded:
            # This is a *proof* of no feasible successor under the
            # frozen grammar/reuse, but keep it separate from the
            # finite numerical cost certificate.
            return {
                "applicable": False, "feasible_continuations": 0,
                "optimistic_total_cost": None,
                "candidate_evaluations": evaluations,
                "horizon_chunks": steps,
            }
        states = expanded
    floor = cold_suffix[next_window + steps]
    if floor < 0:
        raise ValueError("negative frozen suffix lower bound")
    return {
        "applicable": True,
        "feasible_continuations": len(states),
        "optimistic_total_cost": min(s.score for s in states) + floor,
        "candidate_evaluations": evaluations,
        "horizon_chunks": steps,
    }


def compare_early(sources: list[dict], snapshots: list[dict],
                  windows, *, saved_maximum: list[dict],
                  incumbent: float, tagged_terminal: float | None) -> dict:
    """Certify only on five preserved quota8 early populations."""
    if (len(sources) != CHECK_CHUNKS
            or [s["window_index"] for s in sources]
            != [s["window_index"] for s in snapshots[:CHECK_CHUNKS]]):
        raise ValueError("H0.17 grace population/window census mismatch")
    floors = cold_suffix_bounds(windows)
    old = saved_maximum[:CHECK_CHUNKS]
    if len(old) != CHECK_CHUNKS:
        raise ValueError("missing saved H0.16 early certificate windows")
    output = []
    for si, saved, previous in zip(sources, snapshots, old):
        wi = si["window_index"]
        if (si["tagged_count"] != saved["tagged_retained_count"]
                or si["retained_keys"] != saved["retained_keys"]
                or si["best_tagged_score"] != (
                    saved["retained_best_tagged"]["score"]
                    if saved["retained_best_tagged"] is not None else None)
                or previous["window_index"] != wi):
            raise ValueError("independently replayed tagged frontier differs")
        best_score = (min(x.score for x in si["tagged"])
                      if si["tagged"] else None)
        # H0.16 used the H0.14 nine-decimal snapshot, so enforce
        # byte-for-byte parity using the same rounded prefix. Exact
        # unrounded Branch states remain available for lookahead.
        baseline = certificate(
            round(best_score, 9) if best_score is not None else None,
            floors["maximum"][wi+1], incumbent)
        if any(baseline[key] != previous[key] for key in (
                "applicable", "certified", "lower_bound",
                "incumbent_cost", "margin_lb_minus_incumbent", "slack")):
            raise ValueError("H0.16 saved maximum certificate differs")
        one = exact_lookahead_bound(
            si["tagged"], windows,
            next_window=wi+1, lookahead=1,
            cold_suffix=floors["maximum"])
        two = exact_lookahead_bound(
            si["tagged"], windows,
            next_window=wi+1, lookahead=2,
            cold_suffix=floors["maximum"])
        # Bounds for different horizons are individually admissible,
        # even though cold-block partitions may differ. Max preserves
        # all earlier certificates; never weaken them.
        options = [floors["maximum"][wi+1] + best_score
                   ] if best_score is not None else []
        for variant in (one, two):
            if variant["applicable"]:
                options.append(variant["optimistic_total_cost"])
        best = max(options) if options else None
        strengthened = certificate(best, 0.0, incumbent)
        if baseline["certified"] and not strengthened["certified"]:
            raise ValueError("H0.17 revoked prior valid certificate")
        if (tagged_terminal is not None and tagged_terminal < incumbent-EPS
                and strengthened["certified"]):
            raise ValueError("H0.17 falsely discarded known improving lineage")
        output.append({
            "window_index": wi,
            "offset_from_join": wi-sources[0]["window_index"],
            "tagged_count": si["tagged_count"],
            "original_h016_certificate": baseline,
            "state_aware_one": {
                **{k:v for k,v in one.items()
                   if k != "optimistic_total_cost"},
                "optimistic_total_cost": (
                    round(one["optimistic_total_cost"], 9)
                    if one["optimistic_total_cost"] is not None else None)},
            "state_aware_two": {
                **{k:v for k,v in two.items()
                   if k != "optimistic_total_cost"},
                "optimistic_total_cost": (
                    round(two["optimistic_total_cost"], 9)
                    if two["optimistic_total_cost"] is not None else None)},
            "combined_lower_bound_total": (
                round(best, 9) if best is not None else None),
            "combined_certificate": strengthened,
            "new_certificate_vs_h016": (
                strengthened["certified"] and not baseline["certified"]),
        })
    return {
        "checked_windows": len(output),
        "early_certificates_h016": sum(
            r["original_h016_certificate"]["certified"] for r in output),
        "early_certificates_h017": sum(
            r["combined_certificate"]["certified"] for r in output),
        "new_certificates": sum(r["new_certificate_vs_h016"] for r in output),
        "first_new_certificate_offset": next((
            r["offset_from_join"] for r in output
            if r["new_certificate_vs_h016"]), None),
        "two_step_candidate_evaluations": sum(
            r["state_aware_two"]["candidate_evaluations"] for r in output),
        "records": output,
    }


def _check_h016(h012: dict, h013: dict, h014: dict, h015: dict,
                h016: dict, args, source_blobs: dict) -> None:
    if (h016.get("format") != H016_FORMAT
            or h016.get("source_h012_sha256") != sha256_file(args.h012_report)
            or h016.get("source_h013_sha256") != sha256_file(args.h013_report)
            or h016.get("source_h014_sha256") != sha256_file(args.h014_report)
            or h016.get("source_h015_sha256") != sha256_file(args.h015_report)
            or h016.get("frozen_source_blobs") != source_blobs
            or h016.get("beam_width") != BEAM_WIDTH
            or h016.get("quota") != QUOTA
            or h016.get("five_chunk_rule_unchanged") is not True
            or h016.get("no_new_pruning_policy") is not True
            or h016.get("physical_motor_executed") is not False):
        raise ValueError("H0.16 source report/provenance mismatched")
    original = verify_h013(h013, h012, h012_path=args.h012_report)
    if list(original) != list(TARGET_IDS):
        raise ValueError("H0.13 counterexample identities changed")
    from audit_dmdogo_v09_h014_regret_autopsy import FORMAT as H014_FORMAT
    from audit_dmdogo_v09_h015_admissible_suffix_bound import FORMAT as H015_FORMAT
    if (h014.get("format") != H014_FORMAT
            or h014.get("source_h012_sha256") != sha256_file(args.h012_report)
            or h014.get("source_h013_sha256") != sha256_file(args.h013_report)
            or h015.get("format") != H015_FORMAT
            or h015.get("source_h014_sha256") != sha256_file(args.h014_report)
            or h014.get("target_trial_indices") != list(TARGET_IDS)
            or h014.get("physical_motor_executed") is not False):
        raise ValueError("H0.14/H0.15 historical source mismatch")


def _cohort(name: str, h016: dict) -> dict[int, dict]:
    block = h016["groups"][name]
    original_ids = (
        list(TARGET_IDS) if name == "known_regret_cases" else
        list(range(4096, 4104)) if name == "discovery_fixed" else
        list(range(8192, 8200)))
    if (block["trial_indices"] != original_ids
            or len(block["trials"]) != len(original_ids)):
        raise ValueError("H0.16 cohort/index census mismatch")
    result = {r["trial_index"]: r for r in block["trials"]}
    if len(result) != len(original_ids) or list(result) != original_ids:
        raise ValueError("duplicate/out-of-order H0.16 cohort trial")
    return result


def inspect_trial(trial_id: int, saved: dict) -> dict:
    sample = audit_one(trial_id)
    events, spec = generate(trial_id)
    if saved["generator"] != spec or sample["generator"] != spec:
        raise ValueError("H0.17 generator provenance changed")
    planner = GestureChunkPlannerV08()
    frozen = planner.plan(events)
    if not frozen.feasible:
        raise ValueError("frozen synthetic planning became infeasible")
    windows = planner._windows(events)
    chunks = [_chunk_record(x) for x in frozen.chunks]
    cost = original_precise_cost(events, frozen, windows, chunks)
    original_targets = {row["window_index"]: row for row in sample["targets"]}
    saved_targets = {row["window_index"]: row for row in saved["targets"]}
    if (len(saved_targets) != len(saved["targets"])
            or len(original_targets) != len(sample["targets"])
            or not set(saved_targets).issubset(original_targets)
            or saved["target_count"] != len(saved_targets)):
        raise ValueError("H0.17 targeted subset differs from frozen original")
    rows = []
    for join, prior in saved_targets.items():
        target = select_alternative(events, chunks, windows, join)
        source = original_targets[join]
        if (target != source["target"]
                or prior["flag_pair"] != target["flag_pair"]
                or prior["five_chunk_rule_regret"]
                != source["counterexample"]["rule_regret"]
                or prior["original_v08_cost"] != round(cost, 9)):
            raise ValueError("H0.17 target identity/cost mismatched")
        ref = replay_with_snapshots(
            events, chunks, target, policy="full_quota8")
        if (ref["terminal_winner"]["final_cost"]
                != source["baseline"]["full_quota8_cost"]
                or ref["terminal_best_tagged"] is None
                or ref["terminal_best_tagged"]["final_cost"]
                != prior["tagged_final_cost"]):
            raise ValueError("H0.17 full quota8 source differs")
        traces = tagged_grace_states(events, chunks, target)
        evaluated = compare_early(
            traces, ref["snapshots"], windows,
            saved_maximum=prior["methods"]["maximum"]["window_records"],
            incumbent=cost,
            tagged_terminal=prior["tagged_final_cost"])
        if (evaluated["early_certificates_h016"]
                != prior["methods"]["maximum"]["early_certificate"]):
            # Counts are integers while H0.16 stored a boolean. The
            # previous in-sample results had zero early certificates;
            # reject unusual mismatches instead of inferring.
            if bool(evaluated["early_certificates_h016"]) != prior[
                    "methods"]["maximum"]["early_certificate"]:
                raise ValueError("H0.16 early certificate result differs")
        rows.append({
            "window_index": join,
            "flag_pair": target["flag_pair"],
            "known_regret": source["counterexample"]["rule_regret"],
            "tagged_beats_v08": (
                prior["tagged_final_cost"] < cost-EPS),
            "new_certificates": evaluated["new_certificates"],
            "first_new_certificate_offset":
                evaluated["first_new_certificate_offset"],
            "early": evaluated,
        })
    return {
        "trial_index": trial_id,
        "generator": spec,
        "target_count": len(rows),
        "targets": rows,
    }


def summarize(rows) -> dict:
    all_targets = [x for row in rows for x in row["targets"]]
    return {
        "trial_count": len(rows),
        "target_count": len(all_targets),
        "known_regret_target_count": sum(
            t["known_regret"] for t in all_targets),
        "tagged_beats_v08_count": sum(
            t["tagged_beats_v08"] for t in all_targets),
        "early_certified_h016": sum(
            bool(t["early"]["early_certificates_h016"]) for t in all_targets),
        "early_certified_h017": sum(
            bool(t["early"]["early_certificates_h017"]) for t in all_targets),
        "new_certified_targets": sum(
            t["new_certificates"] > 0 for t in all_targets),
        "new_certified_windows": sum(
            t["new_certificates"] for t in all_targets),
        "two_step_candidate_evaluations": sum(
            t["early"]["two_step_candidate_evaluations"]
            for t in all_targets),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("h012-report", "h013-report", "h014-report",
                  "h015-report", "h016-report", "output"):
        p.add_argument("--"+field, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("refusing to overwrite H0.17 saved output")
    root = Path(__file__).resolve().parents[1]
    source_blobs = verify_frozen_cost_source(root)
    h012, h013, h014, h015, h016 = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (
            args.h012_report, args.h013_report, args.h014_report,
            args.h015_report, args.h016_report)]
    _check_h016(h012,h013,h014,h015,h016,args,source_blobs)
    groups = {}
    for name in ("known_regret_cases", "discovery_fixed", "evaluation_fixed"):
        recorded = _cohort(name, h016)
        outputs = [
            inspect_trial(idx, row) for idx, row in recorded.items()]
        groups[name] = {
            "trial_indices": list(recorded),
            "summary": summarize(outputs),
            "trials": outputs,
        }
        expected = h016["groups"][name]["aggregate"]
        result = groups[name]["summary"]
        if (result["target_count"] != expected["target_count"]
                or result["tagged_beats_v08_count"]
                != expected["tagged_final_beats_incumbent"]
                or result["early_certified_h016"] !=
                expected["methods"]["maximum"]["early_certifiable_targets"]):
            raise ValueError("H0.16 cohort aggregate differs")
    report = {
        "format": FORMAT,
        "source_h012_sha256": sha256_file(args.h012_report),
        "source_h013_sha256": sha256_file(args.h013_report),
        "source_h014_sha256": sha256_file(args.h014_report),
        "source_h015_sha256": sha256_file(args.h015_report),
        "source_h016_sha256": sha256_file(args.h016_report),
        "pinned_frozen_source_blobs": source_blobs,
        "beam_width": BEAM_WIDTH, "quota": QUOTA,
        "checked_grace_chunks": CHECK_CHUNKS,
        "lookahead_max_chunks": LOOKAHEAD_MAX,
        "method": (
            "Enumerate every unpruned feasible one/two-chunk successor "
            "from EVERY current retained tagged full-quota8 state; "
            "append H0.16's state-independent cold suffix, and "
            "max with H0.16's prior valid bound."
        ),
        "groups": groups,
        "production_pruning_changed": False,
        "physical_motor_executed": False,
        "frozen_v08_changed": False,
        "human_likeness_verified": False,
        "limitations": [
            "All trial IDs were examined by prior research; this is bound-strength audit, not an independent holdout.",
            "Two-chunk exact lookahead can increase abstract search work substantially and is not a demonstrated runtime speedup.",
            "A certified tagged frontier only cannot beat an offline completed v0.8 incumbent; it does not ensure unchanged global beam winner under different pruning.",
            "A two-chunk non-continuation is recorded separately rather than promoted to numerical certificate.",
            "No physical Motor, Timing, Validation20, FINAL, or real chart validation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({
        "groups": {name: data["summary"] for name, data in groups.items()},
        "saved": str(args.output),
        "physical_motor_executed": False,
        "production_pruning_changed": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
