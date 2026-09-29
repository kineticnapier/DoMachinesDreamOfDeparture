from __future__ import annotations

import argparse
import json
from pathlib import Path

from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import load_compiled_adofai


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add privileged next-target timing to a motion diagnostic JSON."
    )
    parser.add_argument("input_json")
    parser.add_argument("chart")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_path = Path(args.input_json)
    if not input_path.exists():
        raise SystemExit(f"diagnostic JSON not found: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise SystemExit("diagnostic JSON has no rows list")

    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    segment_meta = summary.get("segment") if isinstance(summary.get("segment"), dict) else {}
    start_s = float(segment_meta.get("start", 0.0))
    end_value = segment_meta.get("end")

    compiled = load_compiled_adofai(args.chart)
    end_s = compiled.duration_s if end_value is None else float(end_value)
    segment = build_playable_segment(compiled, start_s=start_s, end_s=end_s)

    for row in rows:
        if not isinstance(row, dict):
            continue
        hits = int(row.get("hits", 0))
        misses = int(row.get("misses", 0))
        index = hits + misses
        if 0 <= index < len(segment.targets):
            target = segment.targets[index]
            t = float(row.get("t_s", 0.0))
            row["next_target_index"] = index
            row["next_target_floor"] = int(target.floor_index)
            row["next_target_time_s"] = round(float(target.episode_time_s), 6)
            row["time_to_target_ms"] = round((float(target.episode_time_s) - t) * 1000.0, 3)
        else:
            row["next_target_index"] = None
            row["next_target_floor"] = None
            row["next_target_time_s"] = None
            row["time_to_target_ms"] = None

    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    meta["privileged_diagnostics_only"] = [
        "next_target_index",
        "next_target_floor",
        "next_target_time_s",
        "time_to_target_ms",
    ]
    payload["meta"] = meta

    output_path = Path(args.output) if args.output else input_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"enriched={output_path.resolve()} rows={len(rows)}")


if __name__ == "__main__":
    main()
