from __future__ import annotations

"""Posthoc countdown diagnostics for causal Timing forecasts.

Inference functions in this module accept HUD observations only. Exact event
times enter only :func:`evaluate_countdown`, after every forecast and causal
tracker snapshot has already been produced.
"""

from dataclasses import dataclass
from math import isfinite
from statistics import median

import torch

from dmdod.evaluation.autoplay_sequence_v1 import CausalForecastReconciler
from dmdod.planning import hud_timing_tracker_transition_v14 as timing_v14
from dmdod.training import prestige1_autoplay_sequence_student as student


WINDOW_S = 0.080
TRACKER_ASSOCIATION_S = 0.020


@dataclass(frozen=True, slots=True)
class CausalFrameForecast:
    now_s: float
    raw_offset_s: float | None
    confidence: float | None
    present: bool
    accepted_offsets_s: tuple[float, ...]
    pending: tuple[tuple[int, float], ...]


def infer_neural(
    visible: torch.Tensor,
    frame_times_s: tuple[float, ...],
    model: student.MultiEventTimingStudent,
    *,
    threshold: float,
) -> tuple[CausalFrameForecast, ...]:
    """Run v3 using only the causal 263D HUD sequence."""

    if visible.shape != (len(frame_times_s), 263):
        raise ValueError("neural countdown inference requires [T,263] HUD")
    with torch.no_grad():
        offsets, logits = model(visible)
        probabilities = torch.sigmoid(logits)
    tracker = CausalForecastReconciler()
    rows = []
    for index, now_s in enumerate(frame_times_s):
        decoded = student.decode_forecast(
            offsets[index], logits[index], confidence_threshold=threshold
        )
        tracker.observe(
            now_s, decoded.offsets_s, decoded.confidences,
            eligible=not decoded.abstained,
        )
        rows.append(CausalFrameForecast(
            now_s=float(now_s),
            raw_offset_s=float(offsets[index, 0].detach().cpu()),
            confidence=float(probabilities[index, 0].detach().cpu()),
            present=not decoded.abstained,
            accepted_offsets_s=decoded.offsets_s,
            pending=tuple((event.local_id, event.due_s) for event in tracker.pending),
        ))
    return tuple(rows)


def infer_v14(
    hud_rows: tuple[tuple[float, ...], ...],
    frame_times_s: tuple[float, ...],
) -> tuple[CausalFrameForecast, ...]:
    """Run Timing v1.4 on the exact same HUD observations as v3."""

    if len(hud_rows) != len(frame_times_s) or not hud_rows:
        raise ValueError("invalid v1.4 countdown HUD sequence")
    tracker = CausalForecastReconciler()
    previous = None
    rows = []
    for hud, now_s in zip(hud_rows, frame_times_s):
        forecast = timing_v14.forecast_visible_window(
            hud, horizon_s=WINDOW_S, previous_hud=previous
        )
        previous = hud
        confidence = 1.0 if forecast.times_s else None
        tracker.observe(
            now_s, forecast.times_s,
            tuple(1.0 for _ in forecast.times_s),
            eligible=forecast.eligible,
            covered_until_s=forecast.covered_until_s,
        )
        rows.append(CausalFrameForecast(
            now_s=float(now_s),
            raw_offset_s=float(forecast.times_s[0]) if forecast.times_s else None,
            confidence=confidence,
            present=bool(forecast.times_s),
            accepted_offsets_s=tuple(float(value) for value in forecast.times_s),
            pending=tuple((event.local_id, event.due_s) for event in tracker.pending),
        ))
    return tuple(rows)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, int(0.95 * len(ordered) + 0.999999) - 1)]


def _event_row(event_index: int, event_s: float, frames: list[CausalFrameForecast]) -> dict:
    samples = []
    for frame in frames:
        true_remaining_ms = (event_s - frame.now_s) * 1000.0
        raw_ms = None if frame.raw_offset_s is None else frame.raw_offset_s * 1000.0
        tracker_candidates = [
            (abs(due_s - event_s), local_id) for local_id, due_s in frame.pending
            if abs(due_s - event_s) <= TRACKER_ASSOCIATION_S + 1e-9
        ]
        local_id = min(tracker_candidates)[1] if tracker_candidates else None
        samples.append({
            "time_s": frame.now_s,
            "true_remaining_ms": true_remaining_ms,
            "predicted_remaining_ms": raw_ms,
            "absolute_due_error_ms": (
                abs(raw_ms - true_remaining_ms) if raw_ms is not None else None
            ),
            "confidence": frame.confidence,
            "present": frame.present,
            "accepted_count": len(frame.accepted_offsets_s),
            "tracker_local_id_within_20ms": local_id,
        })

    raw_errors = [row["absolute_due_error_ms"] for row in samples
                  if row["absolute_due_error_ms"] is not None]
    gated_errors = [row["absolute_due_error_ms"] for row in samples
                    if row["present"] and row["absolute_due_error_ms"] is not None]
    raw_steps = []
    gated_steps = []
    monotonic_violations = 0
    for before, after in zip(samples, samples[1:]):
        a = before["predicted_remaining_ms"]
        b = after["predicted_remaining_ms"]
        elapsed_ms = (after["time_s"] - before["time_s"]) * 1000.0
        if a is not None and b is not None:
            residual = (b - a) + elapsed_ms
            raw_steps.append(abs(residual))
            monotonic_violations += int(b >= a - 1e-6)
            if before["present"] and after["present"]:
                gated_steps.append(abs(residual))

    states = [row["present"] for row in samples]
    disappearances = sum(a and not b for a, b in zip(states, states[1:]))
    # An initial acquisition is not a reappearance. Count only a new present
    # run after this event was previously present and subsequently absent.
    reappearances = 0
    seen_present = False
    disappeared = False
    for state in states:
        if state:
            reappearances += int(disappeared)
            seen_present = True
            disappeared = False
        elif seen_present:
            disappeared = True
    ids = [row["tracker_local_id_within_20ms"] for row in samples
           if row["tracker_local_id_within_20ms"] is not None]
    switches = sum(a != b for a, b in zip(ids, ids[1:]))
    confidences = [row["confidence"] for row in samples if row["confidence"] is not None]
    return {
        "event_index": event_index,
        "event_time_s": event_s,
        "frames": len(samples),
        "presence_coverage": sum(states) / len(states) if states else None,
        "disappearances": disappearances,
        "reappearances": reappearances,
        "raw_due_mae_ms": _mean(raw_errors),
        "raw_due_p95_ms": _p95(raw_errors),
        "gated_due_mae_ms": _mean(gated_errors),
        "raw_countdown_step_mae_ms": _mean(raw_steps),
        "gated_countdown_step_mae_ms": _mean(gated_steps),
        "monotonic_violations": monotonic_violations,
        "tracker_id_switches_within_20ms": switches,
        "distinct_tracker_ids_within_20ms": len(set(ids)),
        "mean_confidence": _mean(confidences),
        "samples": samples,
    }


def evaluate_countdown(
    forecasts: tuple[CausalFrameForecast, ...],
    exact_event_times_s: tuple[float, ...],
) -> dict:
    """Associate forecasts to exact events posthoc and diagnose failure layers."""

    if not forecasts or any(not isfinite(value) for value in exact_event_times_s):
        raise ValueError("invalid countdown evaluation inputs")
    if any(b <= a for a, b in zip(exact_event_times_s, exact_event_times_s[1:])):
        raise ValueError("event truth must be strictly ordered")
    events = []
    for event_index, event_s in enumerate(exact_event_times_s):
        if event_s <= 1e-9:
            continue
        previous_s = exact_event_times_s[event_index - 1] if event_index else -1.0
        window = [frame for frame in forecasts
                  if previous_s <= frame.now_s < event_s
                  and 0.0 < event_s - frame.now_s <= WINDOW_S + 1e-9]
        if len(window) >= 2:
            events.append(_event_row(event_index, event_s, window))
    if not events:
        raise ValueError("no event has a measurable countdown window")

    def values(name: str) -> list[float]:
        return [float(row[name]) for row in events if row[name] is not None]

    total_frames = sum(row["frames"] for row in events)
    total_present = sum(row["presence_coverage"] * row["frames"] for row in events)
    aggregate = {
        "events": len(events),
        "frames": total_frames,
        "presence_coverage": total_present / total_frames,
        "events_with_disappearance": sum(row["disappearances"] > 0 for row in events),
        "events_with_reappearance": sum(row["reappearances"] > 0 for row in events),
        "disappearances": sum(row["disappearances"] for row in events),
        "reappearances": sum(row["reappearances"] for row in events),
        "raw_due_mae_ms": _mean(values("raw_due_mae_ms")),
        "raw_due_p95_ms": _p95([
            sample["absolute_due_error_ms"] for row in events for sample in row["samples"]
            if sample["absolute_due_error_ms"] is not None
        ]),
        "raw_countdown_step_mae_ms": _mean(values("raw_countdown_step_mae_ms")),
        "gated_countdown_step_mae_ms": _mean(values("gated_countdown_step_mae_ms")),
        "monotonic_violations": sum(row["monotonic_violations"] for row in events),
        "tracker_id_switches_within_20ms": sum(
            row["tracker_id_switches_within_20ms"] for row in events
        ),
        "events_with_multiple_tracker_ids_within_20ms": sum(
            row["distinct_tracker_ids_within_20ms"] > 1 for row in events
        ),
        "median_event_mean_confidence": (
            median(values("mean_confidence")) if values("mean_confidence") else None
        ),
    }
    existence_failure = aggregate["presence_coverage"] < 0.80
    time_failure = (
        aggregate["raw_countdown_step_mae_ms"] is None
        or aggregate["raw_countdown_step_mae_ms"] > 2.0
        or aggregate["raw_due_mae_ms"] is None
        or aggregate["raw_due_mae_ms"] > 10.0
    )
    reassociation_observed = aggregate["events_with_multiple_tracker_ids_within_20ms"] > 0
    aggregate["failure_attribution"] = {
        "event_existence": existence_failure,
        "time_regression": time_failure,
        "forecast_reassociation_observed": reassociation_observed,
        "forecast_reassociation_independent": reassociation_observed and not time_failure,
        "criteria": {
            "existence": "presence coverage < 0.80",
            "time_regression": "raw step MAE > 2ms or raw due MAE > 10ms",
            "reassociation": "at least one truth event maps to >1 causal local ID within 20ms",
            "independent_reassociation": "reassociation observed without raw time-regression failure",
        },
    }
    return {"aggregate": aggregate, "per_event": events}
