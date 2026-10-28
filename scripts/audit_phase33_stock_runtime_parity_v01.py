from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from dmdod.research.stock_runtime_parity_v01 import compare_runtime_to_reference
from dmdod.research.stock_runtime_trace_v02 import load_runtime_jsonl_v02


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("reports/phase20c-frame-stock-state-20261022-r1.json"))
    parser.add_argument("--floor-map", type=Path)
    parser.add_argument(
        "--song-time-offset-s", type=float,
        help="Externally measured live-to-reference song-time offset (seconds)",
    )
    parser.add_argument(
        "--song-time-tolerance-s", type=float,
        help="Externally calibrated maximum residual song-time error (seconds)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.song_time_offset_s is None) != (args.song_time_tolerance_s is None):
        parser.error("--song-time-offset-s and --song-time-tolerance-s must be supplied together")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    floor_map = None
    if args.floor_map:
        floor_map = {int(key): int(value) for key, value in json.loads(args.floor_map.read_text(encoding="utf-8")).items()}
    results = [
        compare_runtime_to_reference(
            session, reference, floor_to_target=floor_map,
            song_time_offset_s=args.song_time_offset_s,
            song_time_tolerance_s=args.song_time_tolerance_s,
        )
        for session in load_runtime_jsonl_v02(args.runtime)
    ]
    verdicts = {row["verdict"] for row in results}
    verdict = "RUNTIME_PARITY_MISMATCH" if "RUNTIME_PARITY_MISMATCH" in verdicts else ("RUNTIME_PARITY_CONFIRMED" if verdicts == {"RUNTIME_PARITY_CONFIRMED"} else "RUNTIME_PARITY_UNVERIFIED")
    output = {
        "format": "dmdod-phase33-stock-runtime-parity-v0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "runtime": str(args.runtime), "reference": str(args.reference),
        "song_time_alignment": {
            "offset_s": args.song_time_offset_s,
            "tolerance_s": args.song_time_tolerance_s,
            "externally_calibrated": args.song_time_offset_s is not None,
        },
        "sessions": results, "production_changed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
