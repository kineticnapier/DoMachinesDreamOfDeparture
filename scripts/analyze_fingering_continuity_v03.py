from __future__ import annotations

"""Analyze a saved DMDOGO Free/Continuity replay without rerunning physics."""

import argparse
import json
from pathlib import Path

from dmdod.evaluation.fingering_continuity_v03 import analyze_saved_replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    payload = json.loads(args.replay.read_text(encoding="utf-8"))
    result = analyze_saved_replay(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
