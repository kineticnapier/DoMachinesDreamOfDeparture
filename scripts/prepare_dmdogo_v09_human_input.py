from __future__ import annotations

"""Convert existing FingeringRecorder v0.1.1-style text to v0.9 input.

NO automatic claim of verified fingers or event-time alignment: v0.1.1
uses declared key/finger mappings. Converter never records extra sessions,
uses screenshots or captures keyboard input itself.
"""

import argparse
import json
from math import isfinite
from pathlib import Path
import re

from compare_fingering_human_log_v011 import (
    EXPECTED_FINGER, HUMAN_KEY_AXIS, parse_human_log,
)

FORMAT = "dmdogo-v09-human-keypress-input-v1"
SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")
ALLOWED_FINGERS = {
    "LP", "LR", "LM", "LI", "LT", "RP", "RR", "RM", "RI", "RT",
}
ALLOWED_AXES = {
    f"{hand}_{n}" for hand in ("left", "right") for n in range(1, 7)
}


def normalize_legacy(rows: list[dict], *, chart_sha256: str) -> dict:
    if not isinstance(chart_sha256, str) or not SHA256_RE.fullmatch(chart_sha256):
        raise ValueError("chart_sha256 must be the exact chart's 64-character SHA256")
    if not rows:
        raise ValueError("human log must contain at least one KeyDown")
    events = []
    for row in rows:
        events.append({
            "time_ms": row["time_ms"],
            "kind": "keydown",
            "recorder_key": row["key"],
            "axis_proxy": row["axis"],
            "finger_label": row["finger"],
            "finger_label_evidence": "declared_key_mapping_not_observed",
        })
    payload = {
        "format": FORMAT,
        "source_kind": "legacy_fingering_recorder_text_v011",
        "chart_sha256": chart_sha256.lower(),
        "clock_reference": "recorder_relative_ms_unaligned",
        "alignment": {
            "status": "unverified",
            "oracle_time_offset_ms": None,
            "event_correspondence_verified": False,
        },
        "finger_verification": {
            "method": "fixed_declaration_from_legacy_key_map",
            "independent_finger_observation": False,
        },
        "records": events,
        "limitations": [
            "Legacy Kxx labels are recorder layout IDs, not OS scan codes.",
            "Finger labels are assumed from a declared mapping, not visually verified.",
            "Time-zero alignment, missed notes and extra presses are not resolved.",
            "Axes 5/6 exist in v0.1.1, not in frozen eight-axis v0.8 Motor.",
            "Do not use per-note matching to train a policy without independent alignment.",
        ],
    }
    validate_normalized(payload)
    return payload


def validate_normalized(payload: dict) -> None:
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError("invalid human input format")
    if not isinstance(payload.get("chart_sha256"), str) or not SHA256_RE.fullmatch(
            payload["chart_sha256"]):
        raise ValueError("missing or malformed exact chart SHA256")
    if payload.get("clock_reference") != "recorder_relative_ms_unaligned":
        raise ValueError("unsupported clock reference; not yet synchronized")
    alignment = payload.get("alignment")
    if (not isinstance(alignment, dict)
            or alignment.get("status") != "unverified"
            or alignment.get("event_correspondence_verified") is not False
            or alignment.get("oracle_time_offset_ms") is not None):
        raise ValueError("legacy capture must remain unverified")
    verification = payload.get("finger_verification")
    if (not isinstance(verification, dict)
            or verification.get("method") != "fixed_declaration_from_legacy_key_map"
            or verification.get("independent_finger_observation") is not False):
        raise ValueError("legacy mapping is not an observed-finger source")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("human log has no presses")
    previous = float("-inf")
    for i, entry in enumerate(records):
        if not isinstance(entry, dict):
            raise ValueError(f"invalid event {i}")
        time_ms = entry.get("time_ms")
        if type(time_ms) not in (int, float):
            raise ValueError(f"invalid timestamp at {i}")
        try:
            valid = isfinite(time_ms)
        except OverflowError:
            valid = False
        if not valid or time_ms < 0 or time_ms < previous:
            raise ValueError(f"human log is non-finite, negative or unsorted at {i}")
        previous = float(time_ms)
        if (entry.get("kind") != "keydown"
                or not isinstance(entry.get("recorder_key"), str)
                or not re.fullmatch(r"K\d{2}", entry["recorder_key"])
                or entry.get("axis_proxy") not in ALLOWED_AXES
                or entry.get("finger_label") not in ALLOWED_FINGERS
                or entry.get("finger_label_evidence") !=
                "declared_key_mapping_not_observed"
                or entry["recorder_key"] not in HUMAN_KEY_AXIS
                or entry["axis_proxy"] != HUMAN_KEY_AXIS[entry["recorder_key"]]
                or entry["finger_label"] != EXPECTED_FINGER[entry["recorder_key"]]):
            raise ValueError(f"invalid/invented key or finger mapping at {i}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-log", required=True, type=Path)
    parser.add_argument("--chart-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite human recording")
    values = parse_human_log(args.legacy_log)
    payload = normalize_legacy(values, chart_sha256=args.chart_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "status": "normalized_unaligned",
        "records": len(payload["records"]),
        "chart_sha256": payload["chart_sha256"],
        "finger_observation_verified": False,
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
