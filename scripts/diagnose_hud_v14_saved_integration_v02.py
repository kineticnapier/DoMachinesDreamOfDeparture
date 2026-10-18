from __future__ import annotations

"""Read existing HUD-v1.4 integration summaries without simulating anything.

Shows missed/unsupported forecast coverage and allocation decisions from
the stored reports. Older r1 summaries do not store per-target Hit flags;
this tool explicitly does NOT fabricate missed ordinals from their totals.
It never changes checkpoints, actions, parameters, or artifact files.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from scripts.audit_hud_timing_tracker_transition_v14_matrix import generated
from scripts.run_hud_v14_fingering_motor_integration_v02 import (
    DEFAULT_TRAIN_CHART, synthetic_segment,
)


DEFAULT_REPORTS = (
    Path("artifacts/fingering_network_v02/hud-v14-ordinary-r1/integration-summary.json"),
    Path("artifacts/fingering_network_v02/hud-v14-special-r1/integration-summary.json"),
    Path("artifacts/fingering_network_v02/hud-v14-train-smoke-r1/integration-summary.json"),
)


def segment_for(label: str, *, train_chart: Path, train_end_s: float):
    if label.startswith("ordinary-") and label.endswith("bpm"):
        return synthetic_segment(float(label[len("ordinary-"):-3]))
    if label.startswith("special-"):
        return generated((0., 90., 180., 270.), 496.,
                         label.removeprefix("special-"))
    if label.endswith(".adofai"):
        if label != train_chart.name:
            raise ValueError("nonmatching Train chart: " + label)
        return build_playable_segment(
            compile_adofai(load_adofai(train_chart)), end_s=train_end_s
        )
    raise ValueError("unknown predeclared fixture label: " + label)


def diagnose(report: dict, *, train_chart: Path, train_end_s: float) -> list[dict]:
    if report.get("format") != "dmdod-hud-v14-fingering-motor-integration-v0.2":
        raise ValueError("unexpected integration report format")
    output = []
    for case in report["cases"]:
        segment = segment_for(case["label"],
                              train_chart=train_chart, train_end_s=train_end_s)
        truth = tuple(float(t.episode_time_s) for t in segment.targets)
        for name, arm in case["arms"].items():
            if "error" in arm:
                output.append({"case": case["label"], "arm": name,
                               "error": arm["error"]})
                continue
            forecasts = arm["first_seen_predictions"]
            decisions = arm["decisions"]
            by_id = {d["local_id"]: d for d in decisions}
            unmatched = []
            late = []
            undecided = []
            for ordinal, due in enumerate(truth):
                nearest = min(
                    forecasts,
                    key=lambda f: abs(f["due_local_s"] - due),
                    default=None,
                )
                if nearest is None or abs(nearest["due_local_s"] - due) > .020:
                    unmatched.append(ordinal)
                    continue
                local_id = nearest["local_id"]
                d = by_id.get(local_id)
                if d is None:
                    undecided.append(ordinal)
                elif d["reason"] == "insufficient_observable_lead":
                    late.append(ordinal)
            missing = arm.get("miss_diagnostics_posthoc", None)
            if missing is None:
                actual_missed_ordinals = "unknown_in_r1_summary"
            else:
                actual_missed_ordinals = [
                    {"ordinal": m["target_ordinal"], "category": m["category"],
                     "nearest": m["nearest_local_forecast"],
                     "decision": m["decision"]}
                    for m in missing
                ]
            output.append({
                "case": case["label"],
                "arm": name,
                "hits": arm["stats"]["hits"],
                "targets": arm["stats"]["targets"],
                "misses": arm["stats"]["misses"],
                "timing_without_20ms_forecast_ordinals": unmatched,
                "forecast_with_no_decision_ordinals": undecided,
                "insufficient_lead_ordinals": late,
                "decision_reason_counts": dict(Counter(
                    d["reason"] for d in decisions
                )),
                "actual_missed_ordinals_posthoc": actual_missed_ordinals,
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", type=Path,
                        default=DEFAULT_REPORTS)
    parser.add_argument("--train-chart", type=Path, default=DEFAULT_TRAIN_CHART)
    parser.add_argument("--train-end-s", type=float, default=5.)
    args = parser.parse_args()
    for filename in args.reports:
        if not filename.exists():
            parser.error(f"missing saved summary: {filename}")
        content = json.loads(filename.read_text(encoding="utf-8"))
        rows = diagnose(content, train_chart=args.train_chart,
                        train_end_s=args.train_end_s)
        print(f"=== {filename} ===")
        for row in rows:
            print(json.dumps(row, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
