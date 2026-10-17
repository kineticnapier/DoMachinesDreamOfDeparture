from __future__ import annotations

"""Matched, de-duplicated evaluation of causal Timing event forecasts."""

from dataclasses import dataclass, replace
from math import isfinite
from statistics import median

import torch

from dmdod.planning import hud_timing_tracker_transition_v14 as timing_v14
from dmdod.training import prestige1_autoplay_readout as old_readout
from dmdod.training import prestige1_autoplay_sequence_student as student


MATCH_TOLERANCES_MS = (1.0, 5.0, 10.0, 20.0)
PRACTICAL_MIN_LEAD_MS = 10.0
PRACTICAL_MAX_LEAD_MS = 80.0
IDENTITY_TOLERANCE_S = 0.005


@dataclass(frozen=True, slots=True)
class ForecastEvent:
    local_id: int
    due_s: float
    first_seen_s: float
    confidence: float | None
    confirmations: int

    @property
    def lead_ms(self) -> float:
        return (self.due_s - self.first_seen_s) * 1000.0


def _ordered_matches(left: list[float], right: list[float], tolerance_s: float):
    """Order-preserving DP: maximize matches, then minimize total error."""

    n, m = len(left), len(right)
    dp = [[(0, 0.0, ()) for _ in range(m + 1)] for _ in range(n + 1)]

    def better(a, b):
        return a if (a[0], -a[1]) >= (b[0], -b[1]) else b

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            value = better(dp[i - 1][j], dp[i][j - 1])
            error = abs(left[i - 1] - right[j - 1])
            if error <= tolerance_s:
                previous = dp[i - 1][j - 1]
                paired = (previous[0] + 1, previous[1] + error,
                          previous[2] + ((i - 1, j - 1, error),))
                value = better(value, paired)
            dp[i][j] = value
    return dp[n][m][2]


class CausalForecastReconciler:
    """Reconcile overlapping forecasts without access to target identity."""

    def __init__(self, *, horizon_s: float = student.HORIZON_S):
        self.horizon_s = horizon_s
        self.pending: list[ForecastEvent] = []
        self.reported: list[ForecastEvent] = []
        self.next_id = 0

    def observe(
        self,
        now_s: float,
        offsets_s: tuple[float, ...],
        confidences: tuple[float, ...],
        *,
        eligible: bool,
        covered_until_s: float | None = None,
    ) -> None:
        if not isfinite(now_s) or now_s < 0 or len(offsets_s) != len(confidences):
            raise ValueError("invalid causal forecast update")
        if any(not 0 < value <= self.horizon_s + 1e-7 for value in offsets_s):
            raise ValueError("forecast outside fixed horizon")
        if any(b <= a for a, b in zip(offsets_s, offsets_s[1:])):
            raise ValueError("forecast event times must be ordered")
        elapsed = [event for event in self.pending if event.due_s <= now_s + 1e-7]
        self.reported.extend(elapsed)
        survivors = [event for event in self.pending if event.due_s > now_s + 1e-7]
        if not eligible:
            self.pending = survivors
            return
        due = [now_s + value for value in offsets_s]
        old_due = [event.due_s for event in survivors]
        pairs = _ordered_matches(old_due, due, IDENTITY_TOLERANCE_S)
        by_new = {new: old for old, new, _ in pairs}
        reused = {old for old, _, _ in pairs}
        updated: list[ForecastEvent] = []
        for index, (value, confidence) in enumerate(zip(due, confidences)):
            if not isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("invalid forecast confidence")
            if index in by_new:
                old = survivors[by_new[index]]
                updated.append(replace(
                    old, due_s=value, confidence=confidence,
                    confirmations=old.confirmations + 1,
                ))
            else:
                updated.append(ForecastEvent(
                    self.next_id, value, now_s, confidence, 1
                ))
                self.next_id += 1
        covered = self.horizon_s if covered_until_s is None else covered_until_s
        if not 0 <= covered <= self.horizon_s + 1e-7:
            raise ValueError("invalid covered forecast window")
        # Unmatched events outside the asserted coverage survive. Events within
        # it are contradicted and canceled, not scored as predictions.
        updated.extend(
            event for index, event in enumerate(survivors)
            if index not in reused and event.due_s > now_s + covered + 1e-7
        )
        self.pending = sorted(updated, key=lambda event: (event.due_s, event.local_id))

    def finish(self, *, cutoff_s: float) -> tuple[ForecastEvent, ...]:
        self.reported.extend(event for event in self.pending if event.due_s <= cutoff_s)
        self.pending = [event for event in self.pending if event.due_s > cutoff_s]
        unique = {event.local_id: event for event in self.reported}
        return tuple(sorted(unique.values(), key=lambda event: (event.due_s, event.local_id)))


def _model_frame_forecasts(model, sequence, *, kind: str, threshold: float):
    with torch.no_grad():
        if kind == "new":
            offsets, logits = model(sequence.visible)
            rows = []
            for row in range(len(sequence.visible)):
                decoded = student.decode_forecast(
                    offsets[row], logits[row], confidence_threshold=threshold
                )
                rows.append((decoded.offsets_s, decoded.confidences, not decoded.abstained, None))
            return rows
        raw = model(sequence.visible)
        offsets, probability = old_readout.decode(raw)
        rows = []
        for offset, confidence in zip(offsets.tolist(), probability.tolist()):
            eligible = confidence >= 0.5 and 0 < offset <= student.HORIZON_S
            rows.append(((float(offset),) if eligible else (),
                         (float(confidence),) if eligible else (), eligible, None))
        return rows


def _v14_frame_forecasts(sequence):
    rows = []
    previous = None
    for tensor in sequence.visible:
        hud = tuple(float(value) for value in tensor.tolist())
        forecast = timing_v14.forecast_visible_window(
            hud, horizon_s=student.HORIZON_S, previous_hud=previous
        )
        previous = hud
        rows.append((
            forecast.times_s,
            tuple(1.0 for _ in forecast.times_s),
            forecast.eligible,
            forecast.covered_until_s,
        ))
    return rows


def produce_predictions(sequence, predictor, *, kind: str, threshold: float = 0.5):
    rows = (_v14_frame_forecasts(sequence) if kind == "v1.4" else
            _model_frame_forecasts(predictor, sequence, kind=kind, threshold=threshold))
    tracker = CausalForecastReconciler()
    abstained = 0
    raw_counts = []
    for now_s, (offsets, confidences, eligible, covered) in zip(sequence.frame_times_s, rows):
        abstained += int(not eligible)
        raw_counts.append(len(offsets))
        tracker.observe(now_s, tuple(offsets), tuple(confidences), eligible=eligible,
                        covered_until_s=covered)
    cutoff = sequence.exact_event_times_s[-1] + 0.020
    predictions = tuple(
        event for event in tracker.finish(cutoff_s=cutoff)
        if -1e-7 <= event.due_s <= cutoff
    )
    actual_counts = sequence.present.sum(dim=1).cpu().tolist()
    count_mae = sum(abs(a - b) for a, b in zip(raw_counts, actual_counts)) / len(raw_counts)
    return predictions, {
        "frames": len(rows), "abstained_frames": abstained,
        "abstention_rate": abstained / len(rows),
        "raw_80ms_count_mae": count_mae,
    }


def _confidence_calibration(predictions, matched_prediction_ids):
    usable = [event for event in predictions if event.confidence is not None]
    bins = []
    weighted_gap = 0.0
    for lower in (0.0, 0.2, 0.4, 0.6, 0.8):
        upper = lower + 0.2
        members = [event for event in usable if lower <= event.confidence < upper
                   or (upper == 1.0 and event.confidence == 1.0)]
        if not members:
            continue
        confidence = sum(event.confidence for event in members) / len(members)
        accuracy = sum(event.local_id in matched_prediction_ids for event in members) / len(members)
        weighted_gap += len(members) * abs(confidence - accuracy)
        bins.append({"lower": lower, "upper": upper, "count": len(members),
                     "mean_confidence": confidence, "accuracy": accuracy})
    return {"ece": weighted_gap / len(usable) if usable else None, "bins": bins}


def score_predictions(sequence, predictions: tuple[ForecastEvent, ...]) -> dict:
    zero_time = sum(time <= 1e-9 for time in sequence.exact_event_times_s)
    targets = [time for time in sequence.exact_event_times_s if time > 1e-9]
    results = {}
    for tolerance_ms in MATCH_TOLERANCES_MS:
        pairs = _ordered_matches(
            [event.due_s for event in predictions], targets, tolerance_ms / 1000.0
        )
        errors = [error * 1000 for _, _, error in pairs]
        tp = len(pairs)
        precision = tp / len(predictions) if predictions else 0.0
        recall = tp / len(targets) if targets else 0.0
        results[str(int(tolerance_ms))] = {
            "matches": tp, "targets": len(targets), "predictions": len(predictions),
            "false_predictions": len(predictions) - tp,
            "tracking_rate": recall, "precision": precision,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "mae_ms": sum(errors) / len(errors) if errors else None,
            "p95_ms": (sorted(errors)[max(0, ceil_index(len(errors), 0.95))]
                         if errors else None),
            "matched_errors_ms": errors,
        }
    practical = tuple(
        event for event in predictions
        if PRACTICAL_MIN_LEAD_MS <= event.lead_ms <= PRACTICAL_MAX_LEAD_MS + 1e-6
    )
    practical_pairs = _ordered_matches(
        [event.due_s for event in practical], targets, 0.010
    )
    errors = [error * 1000 for _, _, error in practical_pairs]
    matched_ids = {practical[prediction].local_id for prediction, _, _ in practical_pairs}
    matched_target_indices = {target for _, target, _ in practical_pairs}
    precision = len(practical_pairs) / len(practical) if practical else 0.0
    recall = len(practical_pairs) / len(targets) if targets else 0.0
    leads = [practical[prediction].lead_ms for prediction, _, _ in practical_pairs]
    lead_bins = {
        "10_20": sum(10 <= value < 20 for value in leads),
        "20_40": sum(20 <= value < 40 for value in leads),
        "40_80": sum(40 <= value <= 80 + 1e-6 for value in leads),
    }
    cohorts = {}
    original_indices = [i for i, value in enumerate(sequence.exact_event_times_s) if value > 1e-9]
    labels = [("All",) + sequence.special_event_tags[index] for index in original_indices]
    for index, time in enumerate(targets):
        if ((index and time - targets[index - 1] <= 0.010 + 1e-9) or
                (index + 1 < len(targets) and targets[index + 1] - time <= 0.010 + 1e-9)):
            labels[index] += ("HighDensity10ms",)
    for tag in sorted({tag for row in labels for tag in row}):
        members = {index for index, row in enumerate(labels) if tag in row}
        cohorts[tag] = {
            "targets": len(members),
            "matches": len(members & matched_target_indices),
            "tracking_rate": len(members & matched_target_indices) / len(members) if members else None,
        }
    return {
        "unforecastable_zero_time_targets": zero_time,
        "all_leads": results,
        "practical_10ms_tolerance_10_80ms_lead": {
            "matches": len(practical_pairs), "targets": len(targets),
            "predictions": len(practical), "false_predictions": len(practical) - len(practical_pairs),
            "tracking_rate": recall, "precision": precision,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "mae_ms": sum(errors) / len(errors) if errors else None,
            "p95_ms": sorted(errors)[max(0, ceil_index(len(errors), 0.95))] if errors else None,
            "median_matched_lead_ms": median(leads) if leads else None,
            "lead_bins": lead_bins,
            "matched_errors_ms": errors,
            "matched_leads_ms": leads,
            "cohorts": cohorts,
            "confidence_calibration": _confidence_calibration(practical, matched_ids),
        },
    }


def aggregate_method_rows(rows: list[dict]) -> dict:
    """Aggregate chart-local matching without permitting cross-chart matches."""

    if not rows:
        raise ValueError("no evaluation rows")

    def combine(parts: list[dict], *, practical: bool) -> dict:
        matches = sum(part["matches"] for part in parts)
        targets = sum(part["targets"] for part in parts)
        predictions = sum(part["predictions"] for part in parts)
        errors = [value for part in parts for value in part.get("matched_errors_ms", [])]
        precision = matches / predictions if predictions else 0.0
        recall = matches / targets if targets else 0.0
        result = {
            "matches": matches, "targets": targets, "predictions": predictions,
            "false_predictions": predictions - matches,
            "false_predictions_per_target": (predictions - matches) / targets if targets else None,
            "tracking_rate": recall, "precision": precision,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "mae_ms": sum(errors) / len(errors) if errors else None,
            "p95_ms": sorted(errors)[ceil_index(len(errors), .95)] if errors else None,
        }
        if practical:
            leads = [value for part in parts for value in part.get("matched_leads_ms", [])]
            result["median_matched_lead_ms"] = median(leads) if leads else None
            result["lead_bins"] = {
                key: sum(part["lead_bins"][key] for part in parts)
                for key in ("10_20", "20_40", "40_80")
            }
            tags = sorted({tag for part in parts for tag in part["cohorts"]})
            result["cohorts"] = {}
            for tag in tags:
                count = sum(part["cohorts"].get(tag, {}).get("targets", 0) for part in parts)
                hit = sum(part["cohorts"].get(tag, {}).get("matches", 0) for part in parts)
                result["cohorts"][tag] = {
                    "targets": count, "matches": hit,
                    "tracking_rate": hit / count if count else None,
                }
            bins = {}
            for part in parts:
                for item in part["confidence_calibration"]["bins"]:
                    key = (item["lower"], item["upper"])
                    entry = bins.setdefault(key, {"count": 0, "confidence_sum": 0.0, "correct": 0.0})
                    entry["count"] += item["count"]
                    entry["confidence_sum"] += item["mean_confidence"] * item["count"]
                    entry["correct"] += item["accuracy"] * item["count"]
            output_bins = []
            weighted_gap = 0.0
            total = sum(value["count"] for value in bins.values())
            for (lower, upper), value in sorted(bins.items()):
                confidence = value["confidence_sum"] / value["count"]
                accuracy = value["correct"] / value["count"]
                weighted_gap += value["count"] * abs(confidence - accuracy)
                output_bins.append({"lower": lower, "upper": upper,
                                    "count": value["count"],
                                    "mean_confidence": confidence, "accuracy": accuracy})
            result["confidence_calibration"] = {
                "ece": weighted_gap / total if total else None, "bins": output_bins,
            }
        return result

    scored = [row["score"] for row in rows]
    return {
        "charts": len(rows),
        "frames": sum(row["frame_metrics"]["frames"] for row in rows),
        "abstained_frames": sum(row["frame_metrics"]["abstained_frames"] for row in rows),
        "abstention_rate": (
            sum(row["frame_metrics"]["abstained_frames"] for row in rows) /
            sum(row["frame_metrics"]["frames"] for row in rows)
        ),
        "raw_80ms_count_mae": (
            sum(row["frame_metrics"]["raw_80ms_count_mae"] * row["frame_metrics"]["frames"]
                for row in rows) / sum(row["frame_metrics"]["frames"] for row in rows)
        ),
        "unforecastable_zero_time_targets": sum(
            value["unforecastable_zero_time_targets"] for value in scored
        ),
        "all_leads": {
            tolerance: combine([value["all_leads"][tolerance] for value in scored], practical=False)
            for tolerance in ("1", "5", "10", "20")
        },
        "practical_10ms_tolerance_10_80ms_lead": combine(
            [value["practical_10ms_tolerance_10_80ms_lead"] for value in scored],
            practical=True,
        ),
    }


def ceil_index(length: int, quantile: float) -> int:
    if length < 1:
        raise ValueError("empty quantile")
    from math import ceil
    return min(length - 1, ceil(quantile * length) - 1)


def load_models(new_checkpoint, old_mlp, old_gru, *, device: torch.device):
    new_payload = torch.load(new_checkpoint, map_location=device, weights_only=True)
    if new_payload.get("format") != student.FORMAT:
        raise RuntimeError("new Timing checkpoint format mismatch")
    new_model = student.MultiEventTimingStudent(hidden=int(new_payload["hidden"])).to(device)
    new_model.load_state_dict(new_payload["model_state_dict"], strict=True)
    result = {"new": new_model.eval()}
    for name, path, recurrent in (("mlp", old_mlp, False), ("gru", old_gru, True)):
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload.get("format") != old_readout.FORMAT or payload.get("model_name") != name:
            raise RuntimeError("old Timing checkpoint format mismatch")
        model = old_readout.TimingReadout(int(new_payload["hidden"]), recurrent=recurrent).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        result[name] = model.eval()
    return result, float(new_payload["confidence_threshold"]), new_payload
