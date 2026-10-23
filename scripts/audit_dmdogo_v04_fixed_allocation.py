from __future__ import annotations

"""Posthoc timing ablation: replay frozen Free allocations with v0.4 compensation."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_dmdogo_v04_comparison as v04
import run_fingering_continuity_comparison_v03 as v03

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chart", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-timeout-s", type=float, default=600.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    fixed = {int(event["event_index"]): event["decisions"]["free"]["chosen"]
             for event in replay["events"]}
    segment = build_playable_segment(compile_adofai(load_adofai(args.chart)))
    try:
        result = v04.run_research_arm(
            segment, None, policy_name="latency_fixed_free_ablation",
            planning=False, compensation=True, stop_after_s=None,
            wall_timeout_s=args.wall_timeout_s, fixed_choices=fixed)
    except RuntimeError as error:
        if "fixed allocation is not live-eligible" not in str(error):
            raise
        payload = {
            "format": "dmdogo-v04-fixed-free-latency-ablation",
            "verdict": "FIXED_ALLOCATION_CAUSAL_REPLAY_BLOCKED",
            "causal_status": "posthoc allocation-fixed ablation; not an online policy",
            "future_keydown_or_scorer_used": False,
            "source_replay": str(args.replay),
            "reason": str(error),
            "interpretation": (
                "Compensation changes Motor state, so a later frozen Free allocation "
                "is no longer live-eligible. Timing cannot be isolated by replaying "
                "future baseline choices without violating the live eligibility contract."
            ),
        }
    else:
        payload = {
            "format": "dmdogo-v04-fixed-free-latency-ablation",
            "verdict": "FIXED_ALLOCATION_CAUSAL_REPLAY_COMPLETED",
            "causal_status": "posthoc allocation-fixed ablation; not an online policy",
            "future_keydown_or_scorer_used": False,
            "source_replay": str(args.replay),
            "stats": result["stats"],
            "continuity": v03.continuity_metrics(result["decisions"]),
            "compensation": v04._telemetry_summary(result["compensation_telemetry"]),
            "termination_reason": result["termination_reason"],
            "telemetry": result["compensation_telemetry"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
