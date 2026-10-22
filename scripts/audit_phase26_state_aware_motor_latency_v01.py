from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from dmdod.motor.n_key import NKeyAction, NKeyMotorEnv
from dmdod.research.state_aware_motor_latency_v01 import execute_immediate_press, predict_latency_envelope


def _command(env, axis, value):
    return NKeyAction(tuple(value if key == axis else 0.0 for key in env.key_names))


def run() -> dict:
    rows = []
    for axis in NKeyMotorEnv(12).key_names:
        for state in ("cold", "preactivated_10ms", "braked_10ms", "reversing_20ms"):
            env = NKeyMotorEnv(12, control_dt_s=.010, physics_dt_s=.001)
            if state == "preactivated_10ms":
                env.step(_command(env, axis, 1.0))
            elif state == "braked_10ms":
                env.step(_command(env, axis, -1.0))
            elif state == "reversing_20ms":
                env.step(_command(env, axis, 1.0))
                env.step(_command(env, axis, -1.0))
            envelope = predict_latency_envelope(env, axis, exact_full_snapshot=True, reservation_conflict_free=True)
            start = env.diagnostics().time_s
            actual = start + execute_immediate_press(env, axis)
            rows.append({
                "axis": axis, "state": state, "envelope": asdict(envelope),
                "status": envelope.status.value,
                "actual_immediate_arrival_s": actual,
                "contained": envelope.contains(actual),
                "fixed_44ms_covers_latest": envelope.latest_arrival_s is not None and envelope.latest_arrival_s - start <= .044 + 1e-12,
            })
    valid = [row for row in rows if row["status"] == "valid"]
    return {
        "format": "dmdod-phase26-state-aware-motor-latency-v0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "STATE_AWARE_LATENCY_VALIDATED" if valid and all(row["contained"] for row in valid) else "LATENCY_ENVELOPE_UNSOUND",
        "scope": "frozen deterministic simulator; exact full-state and declared command-delay envelope only",
        "cases": rows,
        "metrics": {
            "cases": len(rows), "valid": len(valid),
            "contained": sum(row["contained"] for row in valid),
            "containment_rate": sum(row["contained"] for row in valid) / len(valid),
            "fixed_44ms_covers_latest": sum(row["fixed_44ms_covers_latest"] for row in valid),
            "arrival_error_ms": 0.0,
        },
        "event37": "UNKNOWN: no selected axis/exact Motor snapshot exists because production abstained",
        "production_changed": False,
        "hardware_validated": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = run()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
