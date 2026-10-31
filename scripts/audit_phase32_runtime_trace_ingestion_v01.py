from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from dmdod.research.stock_runtime_trace_v02 import audit_trace_sessions, load_runtime_jsonl_with_sha256_v02


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    sessions, trace_sha256 = load_runtime_jsonl_with_sha256_v02(args.input)
    result = audit_trace_sessions(sessions)
    result["generated_utc"] = datetime.now(timezone.utc).isoformat()
    result["input"] = str(args.input)
    result["input_sha256"] = trace_sha256
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
