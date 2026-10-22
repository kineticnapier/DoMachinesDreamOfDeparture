from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from dmdod.research.stock_runtime_log_v01 import audit_runtime_capture, load_runtime_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = audit_runtime_capture(load_runtime_jsonl(args.input))
    result.update({
        "format": "dmdod-phase25-stock-runtime-parity-v0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "live_mod_executed": False,
        "production_changed": False,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
