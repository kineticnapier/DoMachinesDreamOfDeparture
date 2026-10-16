from __future__ import annotations

"""Read-only counterfactual of the current-floor-change Geometry abstention.

Estimator inputs: at most three successive, causal visible 263D HUD frames.
The frozen original Geometry and near-only GRU stay unmodified. Oracle labels
and true event IDs are used only AFTER counterfactual estimates for scoring.
Neither candidate is adopted for inference or used for Motor training.
"""

from collections import Counter
from dataclasses import dataclass
from math import atan2, isfinite, pi, sin
import json
from pathlib import Path

from . import prestige1 as p
from . import prestige1_autoplay_countdown as cd
from . import prestige1_autoplay_diagnostics as distance
from . import prestige1_autoplay_hybrid_audit as hybrid
from . import prestige1_autoplay_near_ablation as ablation
from . import prestige1_autoplay_near_countdown_audit as near_countdown
from . import prestige1_autoplay_switch_hud_audit as switch_hud
from . import prestige1_autoplay_readout as stage_t
from . import prestige1_transfer as transfer

FORMAT = "prestige1-stage-t-first-frame-floor-change-counterfactual-train20-v1"
REPORT = "stage_timing_autoplay_first_frame_geometry_train20.json"
CANDIDATES = ("current_crossing_step", "carried_previous_step")
EXAMPLES = 8


def _angle(hud: tuple[float, ...]) -> float:
    return atan2(hud[cd.MOTOR_FEATURES], hud[cd.MOTOR_FEATURES+1])


def _motion_step(previous: tuple[float, ...], current: tuple[float, ...]) -> float:
    return cd._wrap(_angle(current)-_angle(previous))


def _synthesized_previous(
    previous: tuple[float, ...], current: tuple[float, ...],
    *, carry_step: float | None,
) -> tuple[float, ...]:
    """Remove ONLY previous/current floor mismatch from the frozen check.

    For carry_step, replace the preceding orbit direction with the angular
    displacement observed at k-2 -> k-1. The current HUD and floor remain
    untouched. This synthetic input is a COUNTERFACTUAL guard test, not an
    assertion that the planet actually followed that trajectory.
    """
    data = list(previous)
    start = cd.SLOT_START + cd.CURRENT_SLOT * cd.FLOOR_DIM
    data[start:start+cd.FLOOR_DIM] = current[start:start+cd.FLOOR_DIM]
    if carry_step is not None:
        angle = _angle(current)-carry_step
        data[cd.MOTOR_FEATURES] = sin(angle)
        data[cd.MOTOR_FEATURES+1] = cos(angle)
    return tuple(data)


@dataclass(frozen=True)
class Counterfactual:
    estimate_s: float | None
    status: str


def first_frame_candidates(
    visible: tuple[tuple[float, ...], ...],
) -> tuple[dict[str, tuple[Counterfactual, ...]], tuple[str, ...]]:
    """Causal predictions for EVERY frame; no labels, event IDs or masks.

    Candidate proposals are only made where ORIGINAL geometry_one abstained
    specifically on a changed current-floor slot. Every OTHER original
    guard (valid floors, speed consistency, events etc.) is left intact.
    """
    if not visible or any(len(frame) != p.INPUT_DIM for frame in visible):
        raise ValueError("Expect nonempty [T,263] HUD frames")
    if any(not all(isfinite(v) for v in frame) for frame in visible):
        raise ValueError("Nonfinite HUD")
    results = {name: [] for name in CANDIDATES}
    original_statuses = []
    for k, frame in enumerate(visible):
        if k == 0:
            status = "no_previous_hud_frame"
        else:
            _, status = cd.geometry_one(visible[k-1], frame)
        original_statuses.append(status)
        options = {name: Counterfactual(None, "not_floor_changed") for name in CANDIDATES}
        if status == "floor_changed":
            previous = visible[k-1]
            raw_previous = _synthesized_previous(previous, frame, carry_step=None)
            value, reason = cd.geometry_one(raw_previous, frame)
            options["current_crossing_step"] = Counterfactual(
                value, "eligible" if value is not None else reason)
            if k >= 2:
                carried = _motion_step(visible[k-2], previous)
                carried_previous = _synthesized_previous(previous, frame,
                                                         carry_step=carried)
                value, reason = cd.geometry_one(carried_previous, frame)
                options["carried_previous_step"] = Counterfactual(
                    value, "eligible" if value is not None else reason)
            else:
                options["carried_previous_step"] = Counterfactual(
                    None, "insufficient_history")
        for name in CANDIDATES:
            results[name].append(options[name])
    return {
        name: tuple(values) for name, values in results.items()
    }, tuple(original_statuses)


def _finite_summary(values: list[float]) -> dict:
    # Signed errors and angular deviations both reported using the existing
    # frozen summary; units are explicitly declared at the call sites.
    return cd._summary(values)


def _score_split(
    group: list[stage_t.TimingSequence],
    traces: dict[int, hybrid.HybridTrace],
) -> dict:
    groups = {
        name: {
            "error": [], "near_error": [],
            "same_frame_gru_error": [], "same_frame_hybrid_error": [],
            "eligible_true_switch": 0,
        }
        for name in CANDIDATES
    }
    original_statuses = Counter()
    crossing_statuses = {name: Counter() for name in CANDIDATES}
    switches = 0
    current_floor_guards = 0
    angular_prior_diff_deg = []
    angular_step_ratio = []
    direction_flip = 0
    usable_prior = 0
    next_frame_comparisons = {name: [] for name in CANDIDATES}
    next_frame_comparison_counts = Counter()
    nonrecovery_reasons = Counter()
    nonrecovery_first_frame = Counter()
    first_frame_errors = []
    per_anchor = []

    for seq in group:
        trace = traces[seq.anchor]
        hud = tuple(tuple(float(v) for v in row)
                    for row in seq.visible.detach().cpu().tolist())
        candidates, statuses = first_frame_candidates(hud)
        anchor_switch = anchor_guard = 0
        n = len(hud)
        for k in range(1, n):
            if not (bool(seq.mask[k-1]) and bool(seq.mask[k])):
                continue
            a, b = seq.event_indices[k-1], seq.event_indices[k]
            if a < 0 or b < 0 or a == b:
                continue
            switches += 1
            anchor_switch += 1
            original_statuses[statuses[k]] += 1
            if statuses[k] == "floor_changed":
                current_floor_guards += 1
                anchor_guard += 1
            actual_s = float(seq.remaining_s[k])
            old_gru = trace.fallback_s[k]
            old_hybrid = trace.chosen_s[k]
            if k >= 2:
                old_step = _motion_step(hud[k-2], hud[k-1])
                cross_step = _motion_step(hud[k-1], hud[k])
                if abs(old_step) > 1e-5:
                    usable_prior += 1
                    angular_prior_diff_deg.append(
                        cd._wrap(cross_step-old_step) * 180. / pi)
                    angular_step_ratio.append(abs(cross_step)/abs(old_step))
                    direction_flip += int(old_step*cross_step < 0)
            # A real note event is used ONLY here, after candidate predictions.
            end = k
            while (end < n and bool(seq.mask[end])
                   and seq.event_indices[end] == b):
                end += 1
            first_stable = next(
                (j for j in range(k, end) if trace.mode[j] == "geometry"), None)
            if first_stable is None:
                nonrecovery_first_frame[statuses[k]] += 1
                nonrecovery_reasons.update(statuses[k:end])
            for name in CANDIDATES:
                candidate = candidates[name][k]
                crossing_statuses[name][candidate.status] += 1
                if candidate.estimate_s is None:
                    continue
                block = groups[name]
                delta_ms = (candidate.estimate_s-actual_s)*1000.
                block["error"].append(delta_ms)
                block["same_frame_gru_error"].append((old_gru-actual_s)*1000.)
                block["same_frame_hybrid_error"].append((old_hybrid-actual_s)*1000.)
                block["eligible_true_switch"] += 1
                if bool(seq.near[k]):
                    block["near_error"].append(delta_ms)
                if k+1 < end and trace.geometry_s[k+1] is not None:
                    # Verify candidate-to-real-geometry countdown change without
                    # resetting any estimator to a privileged truth timestamp.
                    actual_delta = (
                        float(seq.remaining_s[k+1])-actual_s)*1000.
                    predicted_delta = (
                        trace.geometry_s[k+1]-candidate.estimate_s)*1000.
                    next_frame_comparisons[name].append(
                        predicted_delta-actual_delta)
                    next_frame_comparison_counts[name] += 1
                first_frame_errors.append({
                    "anchor": seq.anchor, "frame": k,
                    "candidate": name, "signed_error_ms": delta_ms,
                    "oracle_remaining_ms": actual_s*1000.,
                    "estimate_ms": candidate.estimate_s*1000.,
                    "next_geometry_delay_frames": (
                        first_stable-k if first_stable is not None else None),
                })
        per_anchor.append({
            "anchor": seq.anchor, "chart_sha256": seq.chart_sha256,
            "true_switches": anchor_switch,
            "original_floor_changed_abstentions": anchor_guard,
        })

    if not switches:
        raise RuntimeError("No true note switches for first-frame diagnostic")
    scored = {}
    for name, b in groups.items():
        scored[name] = {
            "eligible_switches": b["eligible_true_switch"],
            "absolute_time_error": _finite_summary(b["error"]),
            "near_time_error": _finite_summary(b["near_error"]),
            "same_frame_frozen_gru": _finite_summary(b["same_frame_gru_error"]),
            "same_frame_existing_hybrid": _finite_summary(
                b["same_frame_hybrid_error"]),
            "next_frame_actual_geometry_countdown_residual": _finite_summary(
                next_frame_comparisons[name]),
            "next_frame_pair_count": next_frame_comparison_counts[name],
            "statuses_at_true_switch": dict(sorted(crossing_statuses[name].items())),
        }
    first_frame_errors.sort(
        key=lambda x: abs(x["signed_error_ms"]), reverse=True)
    return {
        "oracle_true_switches": switches,
        "original_floor_changed_at_switch": current_floor_guards,
        "original_geometry_abstention_at_switch": dict(sorted(original_statuses.items())),
        "crossing_orbit_motion": {
            "with_valid_preceding_step": usable_prior,
            "crossing_step_minus_previous_step_deg": _finite_summary(
                angular_prior_diff_deg),
            "crossing_abs_step_to_previous_abs_step_ratio": {
                "count": len(angular_step_ratio),
                "mean": sum(angular_step_ratio)/len(angular_step_ratio)
                if angular_step_ratio else None,
                "max": max(angular_step_ratio) if angular_step_ratio else None,
            },
            "direction_flips": direction_flip,
        },
        "candidates_on_same_oracle_switch_frames": scored,
        "unrecovered_before_next_oracle_event": {
            "first_abstention_reason": dict(sorted(nonrecovery_first_frame.items())),
            "all_frame_abstention_reasons": dict(sorted(nonrecovery_reasons.items())),
        },
        "largest_counterfactual_errors": first_frame_errors[:EXAMPLES],
        "per_anchor": per_anchor,
    }


def _verify_prior_report(
    source: distance.FrozenTimingInputs, config: p.PrestigeConfig,
    expected_source: dict, hybrid_scores: dict, hybrid_switches: dict,
) -> str:
    root = source.root
    path = root / switch_hud.REPORT
    if not path.is_file():
        raise FileNotFoundError("Run previous HUD switch audit first")
    prior_report = json.loads(path.read_text(encoding="utf-8"))
    prior_source = {**expected_source, "stateless_hybrid": p.sha256(root / hybrid.REPORT)}
    if (
        prior_report.get("format") != switch_hud.FORMAT
        or prior_report.get("no_training") is not True
        or prior_report.get("validation_evaluated") is not False
        or prior_report.get("final_evaluated") is not False
        or prior_report.get("original_stage_a_gate_overridden") is not False
        or prior_report.get("motor_student_training_permitted") is not False
        or prior_report.get("config_sha256") != config.signature()
        or prior_report.get("train_partition") != source.partition
        or prior_report.get("source_sha256") != prior_source
    ):
        raise RuntimeError("Frozen prior HUD switch diagnostic provenance mismatch")
    for split, group in (
        ("train_fit", source.fit), ("train_heldout", source.heldout)
    ):
        previous = prior_report["results"][split]
        pair = switch_hud._pair_summary(group)
        if previous["visible_cue_differences"]["pair_counts"] != pair["pair_counts"]:
            raise RuntimeError(f"HUD event change counts disagree: {split}")
        if (pair["pair_counts"]["true_switch"]
                != hybrid_switches[split]["event_switches_observed"]):
            raise RuntimeError("HUD switch and Hybrid event labels disagreed")
        for key in ("near_frames", "visible_frames", "geometry_near_frames"):
            if hybrid_scores[split][key] <= 0 and key != "geometry_near_frames":
                raise RuntimeError(f"Invalid Hybrid source: {split}/{key}")
    return p.sha256(path)


def run_first_frame_audit(config: p.PrestigeConfig) -> dict:
    root = transfer.output_dir(config)
    output_path = root / REPORT
    source = distance.prepare(config, output_paths=(output_path,))
    frozen_model, near_report, near_fingerprints = near_countdown._restore_near_only(
        config, source)
    near_countdown_sha = hybrid._validate_prior(
        root, source, near_report, near_fingerprints, frozen_model)
    existing_sources = {
        **near_fingerprints,
        "near_only_countdown": near_countdown_sha,
    }
    scores, switches, results = {}, {}, {}
    for split, group in (
        ("train_fit", source.fit), ("train_heldout", source.heldout)
    ):
        traces = {
            seq.anchor: hybrid.infer_hybrid(seq.visible, frozen_model)
            for seq in group
        }
        scores[split] = hybrid._aggregate_scores(group, traces, source.reference)
        switches[split] = hybrid._switching_scores(group, traces)
        results[split] = _score_split(group, traces)
        if (results[split]["oracle_true_switches"]
                != switches[split]["event_switches_observed"]):
            raise RuntimeError("First-frame switch and frozen Hybrid mismatch")
    hybrid_sha = switch_hud._check_saved_hybrid(
        root / hybrid.REPORT, config, source, existing_sources,
        scores, switches)
    switch_hud_sha = _verify_prior_report(
        source, config, existing_sources, scores, switches)
    final_sources = {
        **source.source_sha256,
        "near_only_report": p.sha256(root / ablation.REPORT),
        "near_only_checkpoint": p.sha256(root / ablation.CHECKPOINT),
        "distance_audit": p.sha256(root / distance.REPORT),
        "old_countdown": p.sha256(root / cd.REPORT),
        "near_only_countdown": p.sha256(root / near_countdown.REPORT),
        "stateless_hybrid": p.sha256(root / hybrid.REPORT),
        "hud_switch_audit": p.sha256(root / switch_hud.REPORT),
    }
    if (
        final_sources != {
            **existing_sources, "stateless_hybrid": hybrid_sha,
            "hud_switch_audit": switch_hud_sha,
        }
        or p._protect_old_best(config) != source.original_best_sha
    ):
        raise RuntimeError("Frozen Timing inputs changed during first-frame audit")
    output = {
        "format": FORMAT,
        "role": "Train20 fit/chart-disjoint heldout only",
        "purpose": "Test floor-change Geometry guard with causal orbit step and carried step",
        "oracle_input_to_candidates": False,
        "evaluator_only_event_and_timestamp_labels": True,
        "new_training": False,
        "no_training": True,
        "validation_evaluated": False,
        "final_evaluated": False,
        "original_stage_a_gate_overridden": False,
        "motor_student_training_permitted": False,
        "automatically_adopted": False,
        "first_frame_counterfactual_only": True,
        "config_sha256": config.signature(),
        "source_sha256": final_sources,
        "train_partition": source.partition,
        "results": results,
        "limitations": [
            "Synthesized prior HUD floor and/or prior orbit are only a diagnostic",
            "floor_changed is the ONLY Geometry guard lifted",
            "All other original speed, missing-floor and special-event safeguards remain",
            "A candidate accuracy measured after an oracle switch cannot claim detector accuracy",
            "Crossing orbit coordinates may reset on floor transitions; no continuity assumed",
            "Only frozen Train20 expert HUD; no Validation/FINAL, Motor/Fingering, or training",
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    p._no_replace_json(output_path, output)
    return output
