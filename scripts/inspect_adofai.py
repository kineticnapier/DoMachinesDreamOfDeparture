from __future__ import annotations

import argparse
from pathlib import Path

from dmdod.adofai_timing import load_compiled_adofai


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect DMDOD's ExtremeEditor-derived ADOFAI chart compilation."
    )
    parser.add_argument("chart", type=Path)
    parser.add_argument("--start", type=float, default=None, help="chart-time segment start in seconds")
    parser.add_argument("--end", type=float, default=None, help="chart-time segment end in seconds")
    parser.add_argument("--limit", type=int, default=24, help="maximum floor rows to print")
    args = parser.parse_args()

    compiled = load_compiled_adofai(str(args.chart))
    if args.start is None and args.end is None:
        floors = compiled.floors
    else:
        start = 0.0 if args.start is None else args.start
        end = compiled.duration_s if args.end is None else args.end
        floors = compiled.floor_entries_between(start, end)

    print(f"chart: {compiled.source_path}")
    print(
        f"floors={len(compiled.floors)} duration={compiled.duration_s:.6f}s "
        f"bpm={compiled.initial_bpm:g} pitch={compiled.pitch_percent:g}% offset={compiled.offset_ms:g}ms"
    )
    print("floor   target_s    exit_s      bpm   dir planets mid  x        y        events")
    for floor in floors[: max(0, args.limit)]:
        events = ",".join(floor.event_markers) or "-"
        direction = "CCW" if floor.is_ccw else "CW"
        print(
            f"{floor.index:5d} {floor.target_time_s:10.6f} {floor.exit_time_s:10.6f} "
            f"{floor.bpm:8.2f} {direction:>3s} {floor.num_planets:7d} "
            f"{'Y' if floor.midspin else '-':>3s} {floor.x:8.3f} {floor.y:8.3f} {events}"
        )
    if len(floors) > args.limit:
        print(f"... {len(floors) - args.limit} more")


if __name__ == "__main__":
    main()
