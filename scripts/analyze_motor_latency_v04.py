from __future__ import annotations

"""Analyze frozen DMDOGO replay command-to-KeyDown latency without rerunning physics."""

import argparse
import json
from pathlib import Path

from dmdod.evaluation.motor_latency_v04 import analyze_latency_rows, pair_fifo_events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--policy", default="free")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = json.loads(args.replay.read_text(encoding="utf-8"))
    floors = {int(row["index"]): float(row["bpm"]) for row in payload["floors"]}
    policy = payload["policies"][args.policy]
    rows = pair_fifo_events(
        payload["events"], policy["keydowns"], policy=args.policy,
        stats=policy["saved_metrics"], bpm_by_source_floor=floors,
    )
    result = {
        "format": "dmdogo-motor-latency-v0.4",
        "source": str(args.replay),
        "chart": payload["chart"],
        "chart_sha256": payload["chart_sha256"],
        "policy": args.policy,
        "analysis": analyze_latency_rows(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
