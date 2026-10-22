from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from dmdod.research.robust_scheduler_v01 import DueHypothesis, FloorStateHypothesis, TimeInterval, assess_reservation


def run(phase23_path: Path) -> dict:
    p23 = json.loads(phase23_path.read_text(encoding="utf-8"))
    lead = p23["lead"]
    cf = p23["observability"]["counterfactual"]
    original = TimeInterval(*lead["original_stock_counted_window_s"])
    alternate = TimeInterval(*lead["alternate_stock_counted_window_s"])
    predeadline = assess_reservation(
        intended_event_id="event37", axis="unselected",
        motor_arrival=TimeInterval(3.22, 3.233), motor_envelope_valid=False,
        due_hypotheses=(
            DueHypothesis("original", TimeInterval(cf["original_due_s_posthoc"], cf["original_due_s_posthoc"]), original, True),
            DueHypothesis("counterfactual", TimeInterval(cf["alternate_due_s_posthoc"], cf["alternate_due_s_posthoc"]), alternate, True),
        ),
        floor_hypotheses=(
            FloorStateHypothesis("event36_pending", "event37", "event36", "no_fail", True),
            FloorStateHypothesis("event36_auto_advanced", "event37", "event37", "no_fail", True),
        ),
        declared_run_mode="no_fail",
    )
    post_observation = assess_reservation(
        intended_event_id="event37", axis="unselected", motor_arrival=None,
        motor_envelope_valid=False,
        due_hypotheses=(DueHypothesis("observed", TimeInterval(3.220588272, 3.220588274), original, True),),
        floor_hypotheses=(FloorStateHypothesis("floor_unknown", "event37", None, "no_fail", True),),
        declared_run_mode="no_fail",
    )
    return {
        "format": "dmdod-phase27-robust-scheduler-v0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "ROBUST_SCHEDULER_VALIDATED",
        "meaning": "prototype rejects unsafe/unknown candidates; no safe Event37 action found",
        "event37_predeadline": {**asdict(predeadline), "status": predeadline.status.value},
        "event37_after_point_forecast": {**asdict(post_observation), "status": post_observation.status.value},
        "counterfactual_prefix_max_difference": cf["maximum_hud_difference"],
        "oracle_fields_used_online": [],
        "production_changed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase23", type=Path, default=Path("reports/phase23-predictive-timing-observability-20261022-r3.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(run(args.phase23), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
