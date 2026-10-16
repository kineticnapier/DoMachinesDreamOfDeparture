"""Offline normal-JRP CBPM display precision audit; no Student inference.

Compute display numbers *from* compiled chart as a GUI renderer does,
then compare the next-floor gap reconstructed from text-quantized CBPM with
the true chart gap, purely posthoc. Do not treat this as event tracking.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.timing import compile_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.features.jrp_bpm_v1 import (
    jrp_normal_bpm_for_floor, JRP_SOURCE_REVISION, JRP_OBSERVATION_VERSION,
)


def audit(path: Path, end_s: float, *, decimals: int = 2) -> dict:
    segment = build_playable_segment(compile_adofai(load_adofai(str(path))), end_s=end_s)
    errors_ms: list[float] = []
    unavailable: list[int] = []
    pause_examples: list[dict] = []
    for floor in segment.chart.floors[:-1]:
        if floor.index < 1 or floor.target_time_s > end_s:
            continue
        rendered = jrp_normal_bpm_for_floor(segment, floor.index, decimals=decimals)
        shown = rendered.visible_numeric[1]
        if shown is None or shown <= 0:
            unavailable.append(floor.index)
            continue
        next_floor = segment.chart.floors[floor.index + 1]
        true_gap_s = (next_floor.target_time_s - floor.target_time_s) / segment.pitch_ratio
        gap_from_display_s = 60. / shown
        err = abs(gap_from_display_s - true_gap_s) * 1000
        errors_ms.append(err)
        if floor.pause_s:
            pause_examples.append({"floor": floor.index, "pause_s": floor.pause_s,
                                   "TBPM_CBPM_KPS": rendered.printed,
                                   "true_gap_s": true_gap_s,
                                   "display_gap_s": gap_from_display_s,
                                   "absolute_error_ms": err})
    return {
        "source_rev": JRP_SOURCE_REVISION,
        "feature_version": JRP_OBSERVATION_VERSION,
        "mode": "normal (no Jongyeol pseudo BPM)",
        "chart": path.name,
        "clip_end_s": end_s,
        "decimals": decimals,
        "valid_intervals": len(errors_ms),
        "unavailable_floor_indices": unavailable,
        "mae_ms": statistics.mean(errors_ms) if errors_ms else None,
        "max_abs_error_ms": max(errors_ms, default=None),
        "within_0_1ms": sum(e <= .1 for e in errors_ms),
        "pause_examples": pause_examples,
        "limitation": "Offline formula parity, not HUD-only Timing tracking or JRP live display parity",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("chart", type=Path)
    parser.add_argument("--end-s", type=float, required=True)
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing overwrite")
    result = audit(args.chart, args.end_s, decimals=args.decimals)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
