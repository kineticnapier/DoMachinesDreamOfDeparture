from __future__ import annotations

"""Immutable, HUD-only framewise Hybrid Timing baseline and switching audit.

Inference uses causal [T,263] HUD observations and a frozen near-only GRU.
Exact Autoplay timestamps/event IDs are used ONLY AFTER predictions for
evaluation, NEVER for choosing Geometry vs GRU, resets, or predictions.
This is deliberately STATELESS between frames: a separate stateful event
tracker must be independently designed and audited before Motor adoption.
"""

from collections import Counter
from dataclasses import dataclass
from html import escape
import json
import math
from pathlib import Path

import torch

from . import prestige1 as p
from . import prestige1_autoplay_countdown as cd
from . import prestige1_autoplay_diagnostics as distance
from . import prestige1_autoplay_near_ablation as ablation
from . import prestige1_autoplay_near_countdown_audit as prior
from . import prestige1_autoplay_readout as stage_t
from . import prestige1_transfer as transfer

FORMAT = "prestige1-stage-t-stateless-hud-hybrid-switch-audit-train20-v1"
REPORT = "stage_timing_autoplay_hybrid_switch_train20.json"
PLOT_PREFIX = "stage_timing_autoplay_hybrid_switch_holdout"
MODES = ("geometry", "gru")
TRANSITIONS = ("geometry->gru", "gru->geometry")
JUMP_WARNING_MS = 20.0  # descriptive fixed threshold, not a selected parameter


@dataclass(frozen=True)
class HybridTrace:
    """Inference-only outputs, each produced from past/current HUD and model."""
    chosen_s: tuple[float, ...]
    fallback_s: tuple[float, ...]
    geometry_s: tuple[float | None, ...]
    mode: tuple[str, ...]
    abstain_reasons: tuple[str, ...]


@torch.no_grad()
def infer_hybrid(
    visible: torch.Tensor, frozen_gru: stage_t.TimingReadout,
) -> HybridTrace:
    """CRITICAL: no oracle clock, mask, event ID, or future target argument."""
    if visible.ndim != 2 or visible.shape[1] != p.INPUT_DIM or len(visible) < 1:
        raise ValueError("Hybrid inference requires causal [T,263] HUD only")
    if not bool(torch.isfinite(visible).all()):
        raise ValueError("Nonfinite HUD")
    frozen_gru.eval()
    fallback_t, _ = stage_t.decode(frozen_gru(visible))
    fallback = tuple(float(t) for t in fallback_t.detach().cpu().tolist())
    hud = [tuple(float(x) for x in frame) for frame in visible.detach().cpu().tolist()]
    predicted, geometric, modes, reasons = [], [], [], []
    for index, frame in enumerate(hud):
        estimate, status = (
            cd.geometry_one(hud[index-1], frame)
            if index else (None, "no_previous_hud_frame")
        )
        if estimate is not None:
            if not math.isfinite(estimate) or estimate < 0.:
                raise RuntimeError("Nonfinite or negative Geometry estimate")
            predicted.append(float(estimate))
            modes.append("geometry")
            reasons.append("eligible_simple_floor")
        else:
            predicted.append(fallback[index])
            modes.append("gru")
            reasons.append(status)
        geometric.append(estimate)
    if not all(math.isfinite(x) and x >= 0. for x in predicted):
        raise RuntimeError("Invalid Hybrid timing prediction")
    return HybridTrace(tuple(predicted), fallback, tuple(geometric),
                       tuple(modes), tuple(reasons))


def _base_scores(seq: stage_t.TimingSequence, trace: HybridTrace) -> dict:
    """Oracle evaluation only. Do not feed any result back to infer_hybrid."""
    groups = {
        name: {"all_visible": [], "near": [], "far": []}
        for name in ("hybrid", "near_only_gru", "geometry", "constant")
    }
    per_mode = {name: {"all_visible": [], "near": []} for name in MODES}
    reference_time = None
    return {"groups": groups, "per_mode": per_mode}


def _aggregate_scores(group: list[stage_t.TimingSequence],
                      traces: dict[int, HybridTrace],
                      reference: dict) -> dict:
    buckets = {
        name: {"all_visible": [], "near": [], "far": []}
        for name in ("hybrid", "near_only_gru", "geometry", "constant")
    }
    mode_buckets = {
        name: {"all_visible": [], "near": []} for name in MODES
    }
    choices = Counter()
    reasons = Counter()
    per_anchor = []
    for seq in group:
        trace = traces[seq.anchor]
        if len(trace.chosen_s) != len(seq.event_indices):
            raise RuntimeError("Hybrid trace/HUD length mismatch")
        per_errs = {name: [] for name in buckets}
        near_errs = {name: [] for name in buckets}
        valid_count = near_count = geometry_count = geometry_near_count = 0
        for k in range(len(trace.chosen_s)):
            choices[trace.mode[k]] += 1
            reasons[trace.abstain_reasons[k]] += 1
            if not bool(seq.mask[k]):
                continue
            valid_count += 1
            is_near = bool(seq.near[k])
            near_count += int(is_near)
            geo = trace.geometry_s[k]
            geometry_count += int(geo is not None)
            geometry_near_count += int(geo is not None and is_near)
            truth = float(seq.remaining_s[k])
            estimates = {
                "hybrid": trace.chosen_s[k],
                "near_only_gru": trace.fallback_s[k],
                "geometry": geo,
                "constant": float(reference["time_s"]),
            }
            for name, estimate in estimates.items():
                if estimate is None:
                    continue
                delta = (estimate-truth)*1000.
                buckets[name]["all_visible"].append(delta)
                buckets[name]["near" if is_near else "far"].append(delta)
                per_errs[name].append(delta)
                if is_near:
                    near_errs[name].append(delta)
            mode_buckets[trace.mode[k]]["all_visible"].append(
                (trace.chosen_s[k]-truth)*1000.)
            if is_near:
                mode_buckets[trace.mode[k]]["near"].append(
                    (trace.chosen_s[k]-truth)*1000.)
        per_anchor.append({
            "anchor": seq.anchor, "chart_sha256": seq.chart_sha256,
            "visible_frames": valid_count, "near_frames": near_count,
            "geometry_visible_frames": geometry_count,
            "geometry_near_frames": geometry_near_count,
            "hybrid": {
                "all_visible": cd._summary(per_errs["hybrid"]),
                "near": cd._summary(near_errs["hybrid"]),
            },
            "near_only_gru": {
                "all_visible": cd._summary(per_errs["near_only_gru"]),
                "near": cd._summary(near_errs["near_only_gru"]),
            },
        })
    all_valid = len(buckets["hybrid"]["all_visible"])
    all_near = len(buckets["hybrid"]["near"])
    geometry_valid = len(buckets["geometry"]["all_visible"])
    geometry_near = len(buckets["geometry"]["near"])
    if not all_valid or not all_near:
        raise RuntimeError("Hybrid audit had no scored near/visible frames")
    return {
        "all_hud_mode_counts": dict(choices),
        "hud_eligibility_reasons": dict(sorted(reasons.items())),
        "visible_frames": all_valid, "near_frames": all_near,
        "geometry_visible_frames": geometry_valid,
        "geometry_near_frames": geometry_near,
        "geometry_visible_coverage": geometry_valid / all_valid,
        "geometry_near_coverage": geometry_near / all_near,
        "matched_baselines": {
            name: {label: cd._summary(values) for label, values in b.items()}
            for name, b in buckets.items()
        },
        "hybrid_by_mode": {
            name: {label: cd._summary(values) for label, values in b.items()}
            for name, b in mode_buckets.items()
        },
        "per_anchor": per_anchor,
    }


def _switching_scores(group: list[stage_t.TimingSequence],
                      traces: dict[int, HybridTrace]) -> dict:
    # All pair categories are diagnostic only. They use oracle labels ONLY
    # to evaluate whether observed changes correspond to a true event jump.
    by_class = {
        state: {
            label: {
                "hybrid_residual": [], "gru_residual": [],
                "prediction_delta": [], "oracle_delta": [],
                "error_before": [], "error_after": [],
                "near_hybrid_residual": [],
            }
            for label in ("same_event", "event_changed")
        }
        for state in ("geometry->geometry", "geometry->gru",
                      "gru->geometry", "gru->gru")
    }
    all_pair_counts = Counter()
    observable_counts = Counter()
    false_rises = {"same_event": 0, "same_event_pairs": 0}
    positive_switches = correct_rises = 0
    total_switches = 0
    per_anchor = []
    for seq in group:
        trace = traces[seq.anchor]
        local_transitions = Counter()
        for k in range(1, len(trace.chosen_s)):
            state = f"{trace.mode[k-1]}->{trace.mode[k]}"
            all_pair_counts[state] += 1
            local_transitions[state] += 1
            if not (bool(seq.mask[k-1]) and bool(seq.mask[k])):
                continue
            if seq.event_indices[k-1] < 0 or seq.event_indices[k] < 0:
                continue
            observable_counts[state] += 1
            same = seq.event_indices[k] == seq.event_indices[k-1]
            label = "same_event" if same else "event_changed"
            now_true = float(seq.remaining_s[k])
            prev_true = float(seq.remaining_s[k-1])
            true_delta = (now_true-prev_true)*1000.
            hybrid_delta = (trace.chosen_s[k]-trace.chosen_s[k-1])*1000.
            gru_delta = (trace.fallback_s[k]-trace.fallback_s[k-1])*1000.
            if same and abs(true_delta+1000*p.CONTROL_DT) > .2:
                raise RuntimeError("Oracle same-event countdown drift")
            row = by_class[state][label]
            row["hybrid_residual"].append(hybrid_delta-true_delta)
            row["gru_residual"].append(gru_delta-true_delta)
            row["prediction_delta"].append(hybrid_delta)
            row["oracle_delta"].append(true_delta)
            row["error_before"].append(
                (trace.chosen_s[k-1]-prev_true)*1000.)
            row["error_after"].append((trace.chosen_s[k]-now_true)*1000.)
            if bool(seq.near[k-1]) and bool(seq.near[k]):
                row["near_hybrid_residual"].append(hybrid_delta-true_delta)
            if same:
                false_rises["same_event_pairs"] += 1
                false_rises["same_event"] += int(hybrid_delta > 0.)
            else:
                total_switches += 1
                if true_delta > 0.:
                    positive_switches += 1
                    correct_rises += int(hybrid_delta > 0.)
        per_anchor.append({
            "anchor": seq.anchor, "chart_sha256": seq.chart_sha256,
            "mode_pairs_all_hud": dict(local_transitions),
        })
    result = {}
    for state, classes in by_class.items():
        result[state] = {}
        for label, row in classes.items():
            after = cd._summary(row["error_after"])
            residual = cd._summary(row["hybrid_residual"])
            result[state][label] = {
                "pairs": len(row["hybrid_residual"]),
                "hybrid_countdown_or_jump_residual": residual,
                "gru_only_counterfactual_residual": cd._summary(row["gru_residual"]),
                "hybrid_predicted_delta_ms": cd._summary(row["prediction_delta"]),
                "oracle_delta_ms": cd._summary(row["oracle_delta"]),
                "before_hybrid_absolute_error": cd._summary(row["error_before"]),
                "after_hybrid_absolute_error": after,
                "near_pair_residual": cd._summary(row["near_hybrid_residual"]),
                "spikes_over_20ms": sum(
                    abs(v)>JUMP_WARNING_MS for v in row["hybrid_residual"]),
                "spike_rate_over_20ms": (
                    sum(abs(v)>JUMP_WARNING_MS for v in row["hybrid_residual"])
                    / len(row["hybrid_residual"])
                    if row["hybrid_residual"] else None
                ),
            }
    return {
        "all_hud_mode_pair_counts": dict(all_pair_counts),
        "observable_mode_pair_counts": dict(observable_counts),
        "by_mode_transition_and_oracle_event": result,
        "event_switches_observed": total_switches,
        "positive_oracle_event_jumps": positive_switches,
        "hybrid_positive_jump_detected": correct_rises,
        "positive_jump_direction_recall": (
            correct_rises/positive_switches if positive_switches else None),
        "same_oracle_event_pairs": false_rises["same_event_pairs"],
        "incorrect_upward_steps_while_same_event": false_rises["same_event"],
        "incorrect_upward_step_rate": (
            false_rises["same_event"]/false_rises["same_event_pairs"]
            if false_rises["same_event_pairs"] else None
        ),
        "per_anchor": per_anchor,
        "warning": (
            "True event identity is used AFTER inference solely to segment metrics. "
            "Positive-jump direction recall is not an autonomous event-ID tracker. "
            "Stateless framewise hybrid cannot measure wrong-target tracking duration "
            "or genuine stateful re-synchronization latency."
        ),
    }


def _plot(seq: stage_t.TimingSequence, trace: HybridTrace) -> str:
    start, end = cd._representative_window(seq)
    width, height = 960, 380
    left, right, top, bottom = 80, 20, 55, 58
    ceiling = .350  # plotted values only
    def x(i: int) -> float:
        return left+(i-start)/max(1, end-start-1)*(width-left-right)
    def y(t: float) -> float:
        return top+(1-max(0., min(ceiling, t))/ceiling)*(height-top-bottom)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="25" font-size="17">Train heldout #{seq.anchor:02d} — framewise HUD Hybrid (view capped at 350ms)</text>',
    ]
    for ms in (0, 100, 200, 300):
        yy=y(ms/1000)
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="10" y="{yy+4:.1f}" font-size="13">{ms}ms</text>')
    for k in range(start+1, end):
        xx=x(k)
        if seq.event_indices[k] != seq.event_indices[k-1]:
            parts.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{top}" y2="{height-bottom}" stroke="#aaa" stroke-dasharray="3,4"/>')
        if trace.mode[k] != trace.mode[k-1]:
            col="#dd7635" if trace.mode[k]=="gru" else "#198a5e"
            parts.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{top}" y2="{height-bottom}" stroke="{col}" opacity=".4" stroke-dasharray="2,6"/>')
    series = (
        ("Auto", [float(v) for v in seq.remaining_s.detach().cpu().tolist()], "#171717"),
        ("Near-only GRU", trace.fallback_s, "#3565bb"),
        ("Framewise Hybrid", trace.chosen_s, "#a02d91"),
    )
    for label, values, color in series:
        points=" ".join(f"{x(k):.1f},{y(values[k]):.1f}" for k in range(start,end))
        parts.append(f'<polyline fill="none" points="{points}" stroke="{color}" stroke-width="2"/>')
    # Geometry-only candidate is intentionally discontinuous where ineligible.
    for k in range(start, end):
        if trace.geometry_s[k] is not None:
            parts.append(f'<circle cx="{x(k):.1f}" cy="{y(trace.geometry_s[k]):.1f}" r="1.7" fill="#14834c"/>')
    legends=(
        ("Auto","#171717"), ("Near-only GRU","#3565bb"),
        ("Framewise Hybrid","#a02d91"), ("Geometry eligible","#14834c"),
        ("G→GRU","#dd7635"), ("GRU→G","#198a5e"),
    )
    for k,(label,color) in enumerate(legends):
        xx=left+(k%3)*280
        yy=height-31+(k//3)*18
        parts.append(f'<line x1="{xx}" x2="{xx+21}" y1="{yy}" y2="{yy}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{xx+26}" y="{yy+4}" font-size="12">{escape(label)}</text>')
    parts.append('</svg>')
    return "\n".join(parts)+"\n"


def _validate_prior(
    root: Path, source: distance.FrozenTimingInputs,
    old_report: dict, fingerprints: dict, model: stage_t.TimingReadout,
) -> str:
    path = root / prior.REPORT
    if not path.is_file():
        raise FileNotFoundError("Run the saved Near-only Countdown comparison first")
    report = json.loads(path.read_text(encoding="utf-8"))
    expected_source = {
        **source.source_sha256,
        "near_only_report": p.sha256(root / ablation.REPORT),
        "near_only_checkpoint": p.sha256(root / ablation.CHECKPOINT),
        "distance_audit": p.sha256(root / distance.REPORT),
        "old_countdown": p.sha256(root / cd.REPORT),
    }
    if (
        report.get("format") != prior.FORMAT
        or report.get("role") != "Train20 fit / chart-disjoint heldout ONLY"
        or report.get("no_training") is not True
        or report.get("validation_evaluated") is not False
        or report.get("final_evaluated") is not False
        or report.get("old_stage_a_gate_overridden") is not False
        or report.get("motor_student_training_permitted") is not False
        or report.get("source_sha256") != expected_source
        or report.get("train_partition") != source.partition
        or fingerprints != expected_source
    ):
        raise RuntimeError("Frozen near-only Countdown provenance mismatch")
    for split, group in (("train_fit", source.fit), ("train_heldout", source.heldout)):
        saved = report["metrics"][split]["near_only_gru"]
        current = stage_t.evaluate(model, group, source.reference)
        prior._check_scores(current, {"aggregate": saved["prediction"],
                                      "per_anchor": old_report["near_only_gru"][split]["per_anchor"]},
                            label=f"hybrid input {split}")
        measured = cd.dynamics(model, group, source.reference)
        previous = saved["countdown"]
        observed = prior._near_slope_metrics(measured)
        for name in (
            "near_countdown_mae_ms", "mean_predicted_near_step_ms",
            "switch_mae_ms", "switch_rise_detected", "near_pair_count",
        ):
            if isinstance(observed[name], float):
                if abs(observed[name]-previous[name]) > .05:
                    raise RuntimeError(f"Near-only Countdown replay changed: {split}/{name}")
            elif observed[name] != previous[name]:
                raise RuntimeError(f"Near-only Countdown counts changed: {split}/{name}")
    return p.sha256(path)


def run_hybrid_audit(config: p.PrestigeConfig) -> dict:
    root = transfer.output_dir(config)
    report_path = root / REPORT
    paths = {i: root / f"{PLOT_PREFIX}_{i:02d}.svg" for i in cd.PLOT_ANCHORS}
    source = distance.prepare(config, output_paths=(report_path, *paths.values()))
    model, frozen_near, fingerprints = prior._restore_near_only(config, source)
    near_countdown_sha = _validate_prior(root, source, frozen_near, fingerprints, model)
    snapshots = {
        **fingerprints,
        "near_only_countdown": near_countdown_sha,
    }
    scores, switching, plots = {}, {}, {}
    for split, group in (("train_fit", source.fit), ("train_heldout", source.heldout)):
        traces = {seq.anchor: infer_hybrid(seq.visible, model) for seq in group}
        scores[split] = _aggregate_scores(group, traces, source.reference)
        switching[split] = _switching_scores(group, traces)
        old_score = frozen_near["near_only_gru"][split]["aggregate"]
        current_score = scores[split]["matched_baselines"]["near_only_gru"]
        for scope, key, expected in (
            ("near", "mae_ms", old_score["near_mae_ms"]),
            ("all_visible", "mae_ms", old_score["all_visible_mae_ms"]),
        ):
            if abs(current_score[scope][key]-expected) > .05:
                raise RuntimeError(f"Saved near-only score disagrees with Hybrid replay: {split}/{scope}")
        prior_geometry = prior._geometry_near_subset(
            group, source.models["gru"], model, source.reference)
        geo_score = scores[split]["matched_baselines"]["geometry"]["near"]
        if (
            geo_score["count"] != prior_geometry["eligible_near_frames"]
            or abs(geo_score["mae_ms"] -
                   prior_geometry["error"]["geometry"]["mae_ms"]) > .05
        ):
            raise RuntimeError("HUD geometry eligibility/accuracy changed")
        if split == "train_heldout":
            for seq in group:
                if seq.anchor not in paths:
                    raise RuntimeError("Unexpected chart-disjoint heldout anchor")
                plots[paths[seq.anchor]] = _plot(seq, traces[seq.anchor])
    if set(plots) != set(paths):
        raise RuntimeError("Missing Train-heldout Hybrid SVG overlay")
    expected_snapshots = {
        **source.source_sha256,
        "near_only_report": p.sha256(root / ablation.REPORT),
        "near_only_checkpoint": p.sha256(root / ablation.CHECKPOINT),
        "distance_audit": p.sha256(root / distance.REPORT),
        "old_countdown": p.sha256(root / cd.REPORT),
        "near_only_countdown": p.sha256(root / prior.REPORT),
    }
    if snapshots != expected_snapshots or p._protect_old_best(config) != source.original_best_sha:
        raise RuntimeError("Protected Timing source changed during Hybrid audit")
    report = {
        "format": FORMAT, "role": "Train20 fit / chart-disjoint heldout only",
        "algorithm": "stateless per-frame HUD geometry if eligible, else frozen near-only GRU",
        "prediction_inputs": "causal 263D HUD and frozen GRU weights only",
        "oracle_used_only_for_evaluation": True,
        "no_training": True, "validation_evaluated": False, "final_evaluated": False,
        "original_stage_a_gate_overridden": False,
        "motor_student_training_permitted": False, "automatically_adopted": False,
        "true_event_tracker_implemented": False,
        "config_sha256": config.signature(),
        "train_partition": source.partition,
        "source_sha256": snapshots,
        "scores": scores,
        "switching": switching,
        "plots": [path.name for path in paths.values()],
        "limitations": [
            "A stateless frame-by-frame selector, NOT a stateful count-down event tracker.",
            "Geometry does not cover Midspin, Pause, SetSpeed, Twirl and other special event states.",
            "The mode selector never sees oracle event identities, clock or future chart truths.",
            "Evaluator event IDs are used solely for stratified switch diagnostics.",
            "A mode transition may have a legitimate note transition; same-event and event-change are separated.",
            "No tracking-duration or stateful re-sync latency claims can be made with this baseline.",
            "Expert-driven HUD stream only, no closed-loop policy rollout.",
            "No training, hyperparameter sweep, Motor/Fingering training, gate override, Validation or FINAL.",
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    for path, contents in plots.items():
        with path.open("x", encoding="utf-8") as target:
            target.write(contents)
    p._no_replace_json(report_path, report)
    return report
