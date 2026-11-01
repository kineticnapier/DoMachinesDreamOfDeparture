#!/usr/bin/env python3
"""Post-hoc research A001 interval frontier diagnostic (NOT preregistered).

Replays v1's previously selected (duty, lead) controls for EVERY period on
both sides of the single-vs-alternating comparison. The original 95% judgment
remains unchanged; removing startup targets is diagnostic only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from a001_periodic_motor_probe import Trial
from a001_startup_diagnostic import run_diagnostic

# From the frozen v1 selected screens in Actions run 41287352794.
# Each tuple: interval_ms, duty, lead_ms.
SINGLE_LEFT = (
    (60, .55, 20), (80, .70, 20), (100, .55, 170),
    (120, .55, 80), (150, .55, 50), (180, .55, 80),
    (220, .55, 80), (260, .55, 80), (320, .55, 80),
    (400, .40, 170), (500, .40, 170), (650, .70, 80),
    (800, .70, 110),
)
ALTERNATING = (
    (60, .55, 290), (80, .55, 80), (100, .55, 290),
    (120, .55, 320), (150, .55, 80), (180, .55, 290),
    (220, .40, 170), (260, .40, 170), (320, .55, 170),
    (400, .70, 140), (500, .25, 230), (650, .25, 230),
    (800, .25, 230),
)
SETTLE_CUTS = (0, 4, 8)
EVALUATION_START_S = 1.130
EVALUATION_DURATION_S = 7.200


def report_frontier():
    rows = []
    for mode, configs in (("single_left", SINGLE_LEFT), ("alternating", ALTERNATING)):
        for interval, duty, lead in configs:
            r = run_diagnostic(
                Trial(mode, interval, duty, lead,
                      EVALUATION_START_S, EVALUATION_DURATION_S)
            )
            hits = {d["excluded_first_targets"]: d for d in r["start_sensitivity"]}
            rows.append({
                "mode": mode, "interval_ms": interval,
                "duty": duty, "lead_ms": lead,
                "total_hits": r["hits"], "targets": r["target_count"],
                "total_extras": r["extra_keydowns_in_scoring_range"],
                "misses": r["missed_target_indices"],
                "hit_rate_all": hits[0]["hit_rate"],
                "hit_rate_after_first_4": hits[4]["hit_rate"],
                "hit_rate_after_first_8": hits[8]["hit_rate"],
            })
    return {
        "schema": "research-a001-posthoc-interval-frontier-v1",
        "status": "unapproved, post-hoc, not physical limit",
        "source": "v1 frozen screen winners in run 41287352794",
        "target_conditions": {
            "start_s": EVALUATION_START_S,
            "duration_s": EVALUATION_DURATION_S,
            "same_hand": True,
            "all_targets_are_scored_in_primary_result": True,
            "posthoc_transient_cutoffs": SETTLE_CUTS,
        },
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = report_frontier()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": report["schema"],
        "rows": report["rows"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
