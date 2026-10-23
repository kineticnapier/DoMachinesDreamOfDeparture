from __future__ import annotations

"""Create-once analysis of a completed DMDOGO v0.5 comparison."""

import argparse
import json
from pathlib import Path

from dmdod.evaluation.continuity_compensated_v05 import analyze_continuity_compensation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    telemetry = json.loads(args.telemetry.read_text(encoding="utf-8"))
    floors = {int(row["index"]): float(row["bpm"]) for row in replay["floors"]}
    bpm_by_event = {
        int(row["event_index"]): floors[min(row["floor_indices"]) - 1]
        for row in replay["events"]
    }
    result = {
        "format": "dmdogo-continuity-compensated-cause-audit-v0.5",
        "source_replay": str(args.replay),
        "source_telemetry": str(args.telemetry),
        "chart": replay["chart"],
        "chart_sha256": replay["chart_sha256"],
        "analysis": analyze_continuity_compensation(
            telemetry["continuity"]["decisions"],
            telemetry["continuity_compensated"]["decisions"],
            bpm_by_event=bpm_by_event,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
