from __future__ import annotations

"""Posthoc Phase 17 audit through a visible-counter boundary.

The Phase 16 scorer trace is privileged renderer input.  ``render_counters``
immediately discards target identity and exposes only cumulative display
labels.  ``JudgementObserver`` receives those frames; target truth is joined
again only after observer output exists.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import ceil
from pathlib import Path

from dmdod.features.judgement_observer_v01 import (
    JudgementCounterFrame,
    JudgementDeltaStatus,
    JudgementObserver,
)


FORMAT = "dmdod-phase17-visible-judgement-audit-v0.1"
CANDIDATE = "v15_midspin_twirl_checkpoint_on"


def next_capture_s(event_time_s: float, control_dt_s: float) -> float:
    """Ideal zero-extra-latency capture grid used only by this simulator audit."""
    return ceil((float(event_time_s) - 1e-12) / control_dt_s) * control_dt_s


def render_counters(
    judgement_trace: list[dict], labels: tuple[str, ...], capture_times_s: list[float]
) -> list[JudgementCounterFrame]:
    """Render generic cumulative counters, dropping scorer identity fields.

    This fixture models only margins present in Phase 16's exported KeyDown
    trace.  It does not assert the installed Stock HUD's exact labels or Miss
    rendering behavior.
    """
    counts: Counter[str] = Counter()
    ordered = sorted(
        ((float(row["time_s"]), str(row["margin"])) for row in judgement_trace),
        key=lambda item: item[0],
    )
    cursor = 0
    frames: list[JudgementCounterFrame] = []
    for capture_s in capture_times_s:
        while cursor < len(ordered) and ordered[cursor][0] <= capture_s + 1e-12:
            _, margin = ordered[cursor]
            if margin in labels:
                counts[margin] += 1
            cursor += 1
        frames.append(JudgementCounterFrame.from_mapping(
            observed_at_s=capture_s,
            displayed_at_s=capture_s,
            counters={label: counts[label] for label in labels},
        ))
    return frames


def observe_frames(
    frames: list[JudgementCounterFrame], labels: tuple[str, ...]
) -> list[dict]:
    observer = JudgementObserver(labels)
    rows: list[dict] = []
    for frame in frames:
        result = observer.observe(frame)
        rows.append({
            "status": result.status.value,
            "observed_at_s": result.observed_at_s,
            "displayed_at_s": result.displayed_at_s,
            "event_interval_s": result.event_interval_s,
            "deltas": result.delta_map,
            "total_increment": result.total_increment,
            "within_interval_order_known": result.within_interval_order_known,
        })
    return rows


def run(source: Path) -> dict:
    raw = source.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    case = data["cases"][0]
    arm = case["arms"][CANDIDATE]
    window = arm["diagnostic_window_posthoc"]
    trace = window["judgement_trace"]
    first_mismatch = next(
        row for row in trace
        if int(row["target_index"]) == 37 and abs(float(row["time_s"]) - 3.261) < 0.002
    )
    event_time_s = float(first_mismatch["time_s"])
    control_dt_s = float(data["control_dt_s"])
    capture_s = next_capture_s(event_time_s, control_dt_s)
    capture_times = [capture_s - control_dt_s * 2, capture_s - control_dt_s, capture_s]
    labels = tuple(sorted({str(row["margin"]) for row in trace}))
    frames = render_counters(trace, labels, capture_times)
    observed = observe_frames(frames, labels)
    detected = next(
        row for row in observed
        if row["status"] == JudgementDeltaStatus.DELTA.value
        and row["deltas"].get(str(first_mismatch["margin"])) == 1
    )

    # Evaluator-only counterfactual: ID38's physical press relative to its
    # intended due is near-perfect.  This expected label is not Observer input.
    id38 = next(row for row in arm["decisions"] if int(row["local_id"]) == 38)
    intended_error_ms = (event_time_s - float(id38["due_local_s"])) * 1000.0
    expected_if_intended = "perfect" if abs(intended_error_ms) <= 25.0 else "unverified"
    actual_vector = detected["deltas"]
    expected_vector = dict.fromkeys(labels, 0)
    if expected_if_intended in expected_vector:
        expected_vector[expected_if_intended] = 1

    checks = {
        "observer_input_contains_only_visible_counter_fields": (
            set(frames[-1].__dataclass_fields__)
            == {"observed_at_s", "displayed_at_s", "counters"}
        ),
        "id38_press_produces_visible_margin_increment": (
            actual_vector.get("very_late") == 1
        ),
        "ideal_next_10ms_capture_detects_category_difference": (
            detected["observed_at_s"] == capture_s
            and actual_vector != expected_vector
        ),
        "observer_output_has_no_target_identity": all(
            "target_index" not in row and "local_id" not in row for row in observed
        ),
    }
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "CATEGORY_ANOMALY_OBSERVABLE_IDENTITY_NONIDENTIFIABLE",
        "source": {
            "path": str(source),
            "sha256": sha256(raw).hexdigest(),
            "scope": "posthoc_renderer_and_evaluator_only",
        },
        "visible_schema": {
            "labels": labels,
            "claim": "repository Phase16 KeyDown-margin fixture; not verified Stock HUD layout",
            "miss_counter_behavior": "not represented by the exported KeyDown trace; Stock behavior unverified",
        },
        "capture_model": {
            "control_dt_s": control_dt_s,
            "extra_capture_latency_s": 0.0,
            "live_latency_verified": False,
        },
        "visible_observations": observed,
        "id38_posthoc_evaluator": {
            "physical_keydown_s": event_time_s,
            "actual_fifo_target_index": int(first_mismatch["target_index"]),
            "actual_visible_margin": str(first_mismatch["margin"]),
            "intended_timing_id": 38,
            "intended_due_s": float(id38["due_local_s"]),
            "intended_error_ms": intended_error_ms,
            "expected_margin_if_scored_against_intended_due": expected_if_intended,
            "first_ideal_observation_s": detected["observed_at_s"],
            "availability_delay_ms": (detected["observed_at_s"] - event_time_s) * 1000.0,
        },
        "identifiability": {
            "category_anomaly": "observable with an independent causal expected-margin model",
            "fifo_backlog_identity": "nonidentifiable from cumulative deltas alone",
            "reason": (
                "The same very_late:+1 vector can arise from a correctly matched but late press. "
                "Counters expose neither target identity nor within-frame order."
            ),
            "safe_control_correction": "not established and not implemented",
        },
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source)
    if not all(result["checks"].values()):
        result["verdict"] = "AUDIT_INCOMPLETE"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print("verdict:", result["verdict"])
    print("checks:", result["checks"])
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
