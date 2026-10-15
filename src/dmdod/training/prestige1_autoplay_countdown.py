from __future__ import annotations

"""Read-only Stage T countdown / event-switch / visible-geometry diagnosis.

The models see ONLY the 263D HUD. The Auto event ordinals and true times
are used strictly after inference to classify and score pairs, never as
inputs to a model or a geometry-based predictor.
"""

from collections import Counter
from html import escape
from math import atan2, cos, isfinite, pi
from pathlib import Path
import json
import math
import statistics

import torch

from dmdod.features.real_chart import (
    DEFAULT_REAL_CHART_FEATURE_CONFIG as FLOORS,
    REAL_CHART_FLOOR_FEATURE_DIM,
)
from dmdod.features.hud_features import HUD_BPM_REFERENCE
from . import prestige1 as p
from . import prestige1_autoplay_diagnostics as base
from . import prestige1_autoplay_readout as stage_t
from . import prestige1_transfer as transfer

FORMAT = "prestige1-stage-t-countdown-and-hud-geometry-train20-v1"
REPORT = "stage_timing_autoplay_countdown_train20.json"
PLOT_ANCHORS = (2, 7, 13, 14, 20)
PLOT_PREFIX = "stage_timing_autoplay_countdown_holdout"
CONTROL_S = p.CONTROL_DT
MOTOR_FEATURES = 24
ORBIT_FEATURES = 2
SLOT_START = MOTOR_FEATURES + ORBIT_FEATURES
FLOOR_DIM = REAL_CHART_FLOOR_FEATURE_DIM
HUD_OFFSET = SLOT_START + FLOORS.floor_slots * FLOOR_DIM
NEXT_SLOT = FLOORS.behind_floors + 1
CURRENT_SLOT = FLOORS.behind_floors
SKIP_MARKER_FROM = 8
SKIP_MARKER_TO = 14


def _summary(numbers: list[float]) -> dict:
    if not numbers:
        return {"count": 0, "mae_ms": None, "bias_ms": None,
                "median_abs_ms": None, "p95_abs_ms": None,
                "within_2ms": None, "within_5ms": None}
    absolute = sorted(abs(v) for v in numbers)
    index = max(0, math.ceil(.95 * len(absolute)) - 1)
    return {
        "count": len(numbers),
        "mae_ms": statistics.mean(absolute),
        "bias_ms": statistics.mean(numbers),
        "median_abs_ms": statistics.median(absolute),
        "p95_abs_ms": absolute[index],
        "within_2ms": sum(abs(v) <= 2. for v in numbers) / len(numbers),
        "within_5ms": sum(abs(v) <= 5. for v in numbers) / len(numbers),
    }


def _predicted_seconds(seq: stage_t.TimingSequence,
                       model: stage_t.TimingReadout | None, reference: dict) -> list[float]:
    if model is None:
        return [float(reference["time_s"])] * len(seq.event_indices)
    model.eval()
    with torch.no_grad():
        time_s, _ = stage_t.decode(model(seq.visible))
        return [float(x) for x in time_s.cpu().tolist()]


def _visible_slot(hud: tuple[float, ...], slot: int) -> tuple[float, ...]:
    lo = SLOT_START + slot * FLOOR_DIM
    return hud[lo:lo + FLOOR_DIM]


def _wrap(angle: float) -> float:
    return (angle + pi) % (2 * pi) - pi


def geometry_one(previous: tuple[float, ...],
                 current: tuple[float, ...]) -> tuple[float | None, str]:
    """HUD-only angular extrapolation for simple next-floor transitions.

    This is a conditional baseline, not a general ADOFAI timing model:
    no pause/twirl/set-speed/multiplanet/midspin, no chart truth, no clock
    or target identity. The direction is inferred from observed orbit motion.
    It deliberately abstains where this simplistic rule is ambiguous.
    """
    if len(previous) != p.INPUT_DIM or len(current) != p.INPUT_DIM:
        raise ValueError("Geometry estimator requires only two 263D HUD frames")
    if not all(isfinite(x) for x in previous + current):
        return None, "nonfinite_hud"
    prev_floor = _visible_slot(previous, CURRENT_SLOT)
    floor = _visible_slot(current, CURRENT_SLOT)
    behind = _visible_slot(current, CURRENT_SLOT - 1)
    ahead = _visible_slot(current, NEXT_SLOT)
    if (
        behind[-1] < .5 or floor[-1] < .5
        or prev_floor[-1] < .5 or ahead[-1] < .5
    ):
        return None, "unavailable_ordinary_floor"
    if any(abs(a-b) > 1e-6 for a, b in zip(prev_floor, floor)):
        return None, "floor_changed"
    if (
        floor[6] > .5 or floor[7] > .5 or
        ahead[6] > .5 or ahead[7] > .5
    ):
        return None, "midspin_or_multiplanet"
    if any(
        v > .5
        for tile in (floor, ahead)
        for v in tile[SKIP_MARKER_FROM:SKIP_MARKER_TO]
    ):
        return None, "special_event"
    # A zero floor arc modulo 2pi is ambiguous: could be 0 or 360 degrees.
    entry = atan2(floor[2], floor[3])
    exit_angle = atan2(floor[4], floor[5])
    if abs(_wrap(exit_angle - entry)) < 1e-4:
        return None, "ambiguous_full_turn"
    if max(abs(v) for v in current[MOTOR_FEATURES:MOTOR_FEATURES+ORBIT_FEATURES]) < .5:
        return None, "invalid_orbit"
    prev_angle = atan2(previous[MOTOR_FEATURES], previous[MOTOR_FEATURES+1])
    angle = atan2(current[MOTOR_FEATURES], current[MOTOR_FEATURES+1])
    angular_step = _wrap(angle - prev_angle)
    if abs(angular_step) <= 1e-5:
        return None, "stopped_orbit"
    bpm = HUD_BPM_REFERENCE * (2. ** float(current[HUD_OFFSET]))
    if not isfinite(bpm) or bpm <= 0:
        return None, "invalid_bpm"
    angular_rate = pi * bpm / 60.
    # In the current renderer, a simple floor rotates by pi*tile_bpm/60
    # (pitch already in HUD Tile BPM). Real BPM is *not* angular velocity.
    if not .5 <= abs(angular_step)/(angular_rate * CONTROL_S) <= 1.5:
        return None, "inconsistent_orbit_motion"
    sign = 1. if angular_step > 0 else -1.
    # A full revolution would be indistinguishable from 0 with 263D alone.
    distance = (sign * (exit_angle - angle)) % (2 * pi)
    seconds = distance / angular_rate
    if not isfinite(seconds):
        return None, "nonfinite_result"
    return seconds, "eligible_simple_floor"


def _pair_dynamics(seq: stage_t.TimingSequence, values: list[float]) -> dict:
    same_residuals: list[float] = []
    same_near_residuals: list[float] = []
    predicted_deltas: list[float] = []
    actual_deltas: list[float] = []
    switch_residuals: list[float] = []
    switch_truth_positive = 0
    switch_predicted_positive = 0
    switch_true_negative = 0
    for k in range(len(seq.event_indices)-1):
        if not (bool(seq.mask[k]) and bool(seq.mask[k+1])):
            continue
        previous_index, next_index = seq.event_indices[k:k+2]
        if previous_index < 0 or next_index < 0:
            continue
        prev_true, next_true = float(seq.remaining_s[k]), float(seq.remaining_s[k+1])
        delta_true_ms = 1000. * (next_true - prev_true)
        delta_pred_ms = 1000. * (values[k+1] - values[k])
        if previous_index == next_index:
            # For an unchanged target, the reference delta should equal -10ms.
            if abs(delta_true_ms + 1000 * CONTROL_S) > .2:
                raise RuntimeError("Same-event oracle countdown clock drifted")
            residual = delta_pred_ms - delta_true_ms
            same_residuals.append(residual)
            predicted_deltas.append(delta_pred_ms)
            actual_deltas.append(delta_true_ms)
            if bool(seq.near[k] and seq.near[k+1]):
                same_near_residuals.append(residual)
        else:
            switch_residuals.append(delta_pred_ms - delta_true_ms)
            if delta_true_ms > 0:
                switch_truth_positive += 1
                switch_predicted_positive += int(delta_pred_ms > 0)
            else:
                switch_true_negative += 1
    return {
        "same_residuals": same_residuals,
        "same_near_residuals": same_near_residuals,
        "same_prediction_deltas": predicted_deltas,
        "same_truth_deltas": actual_deltas,
        "switch_residuals": switch_residuals,
        "switch_truth_positive": switch_truth_positive,
        "switch_predicted_positive": switch_predicted_positive,
        "switch_truth_nonpositive": switch_true_negative,
    }


def dynamics(model, group: list[stage_t.TimingSequence], reference: dict) -> dict:
    combined = {key: [] for key in (
        "same_residuals", "same_near_residuals", "same_prediction_deltas",
        "same_truth_deltas", "switch_residuals"
    )}
    positive_switches = detected = nonpositive = 0
    rows = []
    for seq in group:
        scores = _pair_dynamics(seq, _predicted_seconds(seq, model, reference))
        for key in combined:
            combined[key].extend(scores[key])
        positive_switches += scores["switch_truth_positive"]
        detected += scores["switch_predicted_positive"]
        nonpositive += scores["switch_truth_nonpositive"]
        rows.append({
            "anchor": seq.anchor, "chart_sha256": seq.chart_sha256,
            "same_event_countdown": _summary(scores["same_residuals"]),
            "same_event_near_countdown": _summary(scores["same_near_residuals"]),
            "event_switch_residual": _summary(scores["switch_residuals"]),
            "positive_switches": scores["switch_truth_positive"],
            "positive_switches_predicted_up": scores["switch_predicted_positive"],
        })
    return {
        "aggregate": {
            "same_event_countdown": _summary(combined["same_residuals"]),
            "same_event_near_countdown": _summary(combined["same_near_residuals"]),
            "mean_predicted_step_ms": (
                statistics.mean(combined["same_prediction_deltas"])
                if combined["same_prediction_deltas"] else None),
            "mean_oracle_step_ms": (
                statistics.mean(combined["same_truth_deltas"])
                if combined["same_truth_deltas"] else None),
            "event_switch_residual": _summary(combined["switch_residuals"]),
            "true_positive_delta_switches": positive_switches,
            "correctly_upward_switches": detected,
            "upward_switch_recall": (
                detected / positive_switches if positive_switches else None),
            "nonpositive_delta_switches": nonpositive,
        },
        "per_anchor": rows,
    }


def geometry_comparison(group: list[stage_t.TimingSequence], models: dict,
                        reference: dict) -> dict:
    errors = {"geometry": [], "constant": [], "mlp": [], "gru": []}
    near_errors = {key: [] for key in errors}
    exclusions = Counter()
    per_anchor = []
    n_visible = n_near = 0
    for seq in group:
        inputs = [tuple(row) for row in seq.visible.detach().cpu().tolist()]
        pred = {
            name: _predicted_seconds(seq, model, reference)
            for name, model in (("constant", None), *models.items())
        }
        eligible = near_eligible = 0
        for i in range(1, len(inputs)):
            if not bool(seq.mask[i]):
                continue
            n_visible += 1
            if bool(seq.near[i]):
                n_near += 1
            value, status = geometry_one(inputs[i-1], inputs[i])
            if value is None:
                exclusions[status] += 1
                continue
            eligible += 1
            if bool(seq.near[i]):
                near_eligible += 1
            truth = float(seq.remaining_s[i])
            for name, candidate in (("geometry", value), *[(n, p[i]) for n, p in pred.items()]):
                signed_error = (candidate - truth) * 1000.
                errors[name].append(signed_error)
                if bool(seq.near[i]):
                    near_errors[name].append(signed_error)
        per_anchor.append({
            "anchor": seq.anchor, "chart_sha256": seq.chart_sha256,
            "eligible_frames": eligible, "near_eligible_frames": near_eligible,
        })
    return {
        "warning": (
            "Conditional 1-step angular HUD heuristic, with inference-only eligibility. "
            "No claims about unhandled full turns, Twirl, Pause, Midspin, SetSpeed, "
            "multi-planet, segment countdown, or arbitrary multi-event lookahead. "
            "Compare NN on identical eligible frames; no oracle-gated prediction."
        ),
        "visible_candidate_frames": n_visible,
        "near_candidate_frames": n_near,
        "eligible_frames": len(errors["geometry"]),
        "near_eligible_frames": len(near_errors["geometry"]),
        "eligibility_fraction": len(errors["geometry"])/n_visible if n_visible else None,
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "same_eligible_frame_comparison": {
            name: {
                "all_visible": _summary(err),
                "near_200ms": _summary(near_errors[name]),
            }
            for name, err in errors.items()
        },
        "per_anchor": per_anchor,
    }


def loss_contributions(model: stage_t.TimingReadout | None,
                       group: list[stage_t.TimingSequence], reference: dict) -> dict:
    """Decompose the ORIGINAL Stage T objective by true distance bin.

    This is an evaluator-side aggregate loss accounting, NOT the exact SGD
    chunk-mean history. It uses the original smooth-L1 scaling and fixed
    near/far weights (1.0/0.25) plus per-valid-frame near-class BCE.
    """
    sums = {
        name: {"frames": 0, "weight": 0., "time_weighted_sum": 0., "bce_sum": 0.}
        for name in base.BINS
    }
    for seq in group:
        if model is None:
            predicted = [float(reference["time_s"])] * len(seq.event_indices)
            probabilities = [float(reference["near_probability"])] * len(seq.event_indices)
        else:
            model.eval()
            with torch.no_grad():
                times, chance = stage_t.decode(model(seq.visible))
            predicted = [float(x) for x in times.cpu().tolist()]
            probabilities = [float(x) for x in chance.cpu().tolist()]
        masks = base._bin_masks(seq)
        for name, mask in masks.items():
            indices = torch.where(mask)[0].cpu().tolist()
            for i in indices:
                truth = float(seq.remaining_s[i])
                delta = abs(predicted[i] - truth)/stage_t.TIME_UNIT_S
                smooth = .5 * delta * delta if delta < 1. else delta - .5
                weight = 1. if bool(seq.near[i]) else stage_t.FAR_TIME_WEIGHT
                probability = max(1e-7, min(1-1e-7, probabilities[i]))
                bce = -math.log(probability) if bool(seq.near[i]) else -math.log1p(-probability)
                entry = sums[name]
                entry["frames"] += 1
                entry["weight"] += weight
                entry["time_weighted_sum"] += weight*smooth
                entry["bce_sum"] += bce
    denom_weight = sum(x["weight"] for x in sums.values())
    denom_frames = sum(x["frames"] for x in sums.values())
    all_time = sum(x["time_weighted_sum"] for x in sums.values())
    all_bce = sum(x["bce_sum"] for x in sums.values())
    if denom_weight <= 0 or denom_frames <= 0:
        raise RuntimeError("No observable frames for loss decomposition")
    result = {}
    for name, row in sums.items():
        result[name] = {
            "frames": row["frames"],
            "time_term_normalized_contribution": row["time_weighted_sum"]/denom_weight,
            "time_term_fraction": row["time_weighted_sum"]/all_time if all_time else None,
            "near_event_bce_normalized_contribution": row["bce_sum"]/denom_frames,
            "bce_fraction": row["bce_sum"]/all_bce if all_bce else None,
        }
    return {
        "normalization": (
            "Original per-frame smooth-L1 normalized by global sum of 1/.25 "
            "weights and event BCE by global visible frame count; NOT the exact "
            "per-SGD-chunk-mean objective"
        ),
        "time_term_total": all_time/denom_weight,
        "event_bce_total": all_bce/denom_frames,
        "by_distance": result,
    }


def _representative_window(seq: stage_t.TimingSequence, width: int = 61):
    valid_switches = [
        i for i in range(1, len(seq.event_indices))
        if seq.mask[i] and seq.mask[i-1]
        and seq.event_indices[i] != seq.event_indices[i-1]
    ]
    center = valid_switches[0] if valid_switches else min(len(seq.event_indices)//2, 35)
    start = max(0, center - width // 2)
    end = min(len(seq.event_indices), start + width)
    return start, end


def svg_timing_trace(seq: stage_t.TimingSequence,
                     predictions: dict[str, list[float]]) -> str:
    start, end = _representative_window(seq)
    n = end - start
    W, H = 920, 360
    LEFT, RIGHT, TOP, BOTTOM = 70, 24, 50, 55
    high_s = .350  # visualization ceiling ONLY; labels/predictions are unmodified
    def X(k):
        return LEFT + (k - start)/(max(1, n-1))*(W-LEFT-RIGHT)
    def Y(seconds):
        return TOP + (1 - min(high_s, max(0., seconds))/high_s)*(H-TOP-BOTTOM)
    rows = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
            '<rect width="100%" height="100%" fill="#fff"/>',
            f'<text x="{LEFT}" y="24" font-size="17" fill="#111">Train heldout #{seq.anchor:02d} — Timing countdown (display capped at 350ms)</text>']
    for ms in (0, 100, 200, 300):
        y = Y(ms / 1000)
        rows.extend([
            f'<line x1="{LEFT}" x2="{W-RIGHT}" y1="{y:.1f}" y2="{y:.1f}" stroke="#ddd"/>',
            f'<text x="9" y="{y+4:.1f}" font-size="13" fill="#555">{ms}ms</text>',
        ])
    for k in range(start+1, end):
        if seq.event_indices[k] != seq.event_indices[k-1]:
            x = X(k)
            rows.append(
                f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{TOP}" y2="{H-BOTTOM}" '
                'stroke="#aaa" stroke-dasharray="4,4"/>'
            )
    series = {
        "Auto truth": ([float(v) for v in seq.remaining_s.detach().cpu().tolist()], "#111"),
        "Constant": (predictions["constant"], "#a2a2a2"),
        "MLP": (predictions["mlp"], "#e17e24"),
        "GRU": (predictions["gru"], "#2466bc"),
    }
    for name, (values, color) in series.items():
        points = " ".join(
            f"{X(k):.1f},{Y(values[k]):.1f}" for k in range(start, end)
        )
        rows.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
    for i, (name, (_, color)) in enumerate(series.items()):
        x = LEFT + i * 165
        y = H - 20
        rows.append(
            f'<line x1="{x}" x2="{x+24}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="3"/>'
            f'<text x="{x+30}" y="{y+4}" font-size="13" fill="#333">{escape(name)}</text>'
        )
    rows.append('</svg>')
    return "\n".join(rows)+"\n"


def run_countdown_audit(config: p.PrestigeConfig) -> dict:
    root = transfer.output_dir(config)
    report_path = root / REPORT
    plot_paths = {
        anchor: root / f"{PLOT_PREFIX}_{anchor:02d}.svg"
        for anchor in PLOT_ANCHORS
    }
    source = base.prepare(config, output_paths=(report_path, *plot_paths.values()))
    groups = (("train_fit", source.fit), ("train_heldout", source.heldout))
    result = {
        "format": FORMAT,
        "role": "Train20 chart-disjoint, heldout selected without validation",
        "no_training": True, "read_only_models": True,
        "stage_a_overridden": False,
        "motor_student_training_permitted": False,
        "validation_evaluated": False, "final_evaluated": False,
        "student_inference_inputs": "causal 263D HUD only",
        "oracle_event_index_use": "evaluator-only pair segmentation, never network or geometry input",
        "source_sha256": source.source_sha256,
        "train_partition": source.partition,
        "timing_dynamics": {
            split: {
                name: dynamics(model, group, source.reference)
                for name, model in (("constant", None), *source.models.items())
            }
            for split, group in groups
        },
        "loss_contributions": {
            split: {
                name: loss_contributions(model, group, source.reference)
                for name, model in (("constant", None), *source.models.items())
            }
            for split, group in groups
        },
        "geometry_baseline": {
            split: geometry_comparison(group, source.models, source.reference)
            for split, group in groups
        },
        "plots": [str(p.name) for p in plot_paths.values()],
        "limitations": [
            "The experiment uses expert-generated visible HUD trajectories, not a closed-loop Student.",
            "A correct countdown slope does not establish a correct absolute event time.",
            "The geometry heuristic is only defined on a restricted observational subset.",
            "No Stage A gate, Student training, Validation or FINAL is affected.",
        ],
    }
    heldout = {seq.anchor: seq for seq in source.heldout}
    if set(heldout) != set(plot_paths):
        raise RuntimeError("Frozen heldout anchors changed")
    pictures = {}
    for anchor, output in plot_paths.items():
        seq = heldout[anchor]
        predicted = {
            name: _predicted_seconds(seq, model, source.reference)
            for name, model in (("constant", None), *source.models.items())
        }
        pictures[output] = svg_timing_trace(seq, predicted)
    if (
        source.source_sha256["stage_t_readout"] != p.sha256(root / stage_t.REPORT)
        or p._protect_old_best(config) != source.original_best_sha
    ):
        raise RuntimeError("Frozen Stage T source changed during countdown replay")
    # Write-once. The caller should preserve any outputs if interrupted.
    root.mkdir(parents=True, exist_ok=True)
    for path, svg in pictures.items():
        with path.open("x", encoding="utf-8") as f:
            f.write(svg)
    p._no_replace_json(report_path, result)
    return result
