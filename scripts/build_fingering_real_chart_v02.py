from __future__ import annotations

"""Build a deterministic Train-only real-chart fingering inventory."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from dmdod.training.fingering_real_chart_v02 import build_inventory


def write_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--max-charts", type=int)
    parser.add_argument("--max-windows-per-chart", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or args.status_file.exists():
        parser.error("refusing to overwrite output")
    started = datetime.now(timezone.utc)
    try:
        report = build_inventory(
            args.dataset, max_charts=args.max_charts,
            max_windows_per_chart=args.max_windows_per_chart,
        )
        report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_once(args.output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        status = {
            "pid": os.getpid(), "started_at_utc": started.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "exit_code": 0, "output": str(args.output),
        }
        write_once(args.status_file, json.dumps(status, indent=2) + "\n")
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        if not args.status_file.exists():
            write_once(args.status_file, json.dumps({
                "pid": os.getpid(), "started_at_utc": started.isoformat(),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "exit_code": 1, "error_type": type(error).__name__, "error": str(error),
            }, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
