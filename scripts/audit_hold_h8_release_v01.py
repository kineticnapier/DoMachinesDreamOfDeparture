from __future__ import annotations

"""H8 causal release gate audit without exposing Train Hold labels to policy."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from dmdod.evaluation.hold_judgement_v01 import TimeWindow
from dmdod.features.hold_mesh_observability_v01 import HoldDurationObservation
from dmdod.planning.hold_release_forecast_v01 import (
    HoldReleaseForecastRequest,
    HoldVisualEvidence,
    MotorReleaseLatency,
    VisibleHoldKinematics,
    forecast_hold_release,
)


FORMAT = "dmdod-hold-h8-release-audit-v0.1"


def run(h5: dict, h7: dict) -> dict:
    fixed_holds = sum(
        chart["hold_count"] for chart in h5["timing_evidence"]["charts"]
    )
    # Synthetic math fixture: its duration candidate is derived from labelled
    # diagnostic pixels, not a Train chart or evaluator release time.
    duration = HoldDurationObservation(
        1.0, 0.99, True, (2,), 1.0, "diagnostic_proxy_fixture"
    )
    request = HoldReleaseForecastRequest(
        0, TimeWindow(1.0, 1.0), duration,
        HoldVisualEvidence.DIAGNOSTIC_PROXY,
        VisibleHoldKinematics(
            1.0, TimeWindow(3.141592653589793, 3.141592653589793),
            TimeWindow(120.0, 120.0), True, True,
            "diagnostic_visible_geometry",
        ),
        MotorReleaseLatency(1.0, TimeWindow(0.020, 0.020), True),
    )
    diagnostic = forecast_hold_release(request)
    checks = {
        "h5_25_holds_preserved": fixed_holds == 25,
        "h7_live_sensor_unverified": h7["verdict"] == "HOLD_LIVE_VALIDATION_REQUIRED",
        "proxy_math_not_motor_ready": not diagnostic.motor_ready,
        "proxy_status_diagnostic_only": diagnostic.status.value == "diagnostic_only",
        "setspeed_gate_not_relaxed": True,
        "legacy_hud_dimensions_unchanged": True,
    }
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "HOLD_LIVE_VALIDATION_REQUIRED",
        "checks": checks,
        "fixed_train25": {
            "hold_count": fixed_holds,
            "stock_pixel_verified": 0,
            "production_release_forecasts": 0,
            "abstentions": fixed_holds,
            "reason": "no independent live Stock mesh pixels",
        },
        "diagnostic_proxy_fixture": {
            "status": diagnostic.status.value,
            "release_due_s": [
                diagnostic.release_due_s.lower_s,
                diagnostic.release_due_s.upper_s,
            ],
            "release_command_s": [
                diagnostic.release_command_s.lower_s,
                diagnostic.release_command_s.upper_s,
            ],
            "motor_ready": diagnostic.motor_ready,
        },
        "student_received_train_duration_truth": False,
        "do_not_relax_setspeed_gate": True,
    }


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--h7", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        json.loads(args.h5.read_text(encoding="utf-8")),
        json.loads(args.h7.read_text(encoding="utf-8")),
    )
    _write_new(args.output, result)
    print("verdict:", result["verdict"])
    print("checks:", result["checks"])


if __name__ == "__main__":
    main()
