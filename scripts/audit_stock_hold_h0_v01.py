from __future__ import annotations

"""Inventory Train-only Hold actions and fingerprint local stock-code evidence."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from dmdod.adofai.chart import _normalize_loose_json


FORMAT = "dmdod-stock-hold-h0-audit-v0.1"
DEFAULT_TRAIN_DIR = Path("data/DMDOD-v140-tuf-p11-p20/Train")
EVIDENCE_FILES = (
    "scrPlayer.cs",
    "scrController.cs",
    "scrConductor.cs",
    "scrLevelMaker.cs",
    "scnGame.cs",
    "scrPlanet.cs",
    "scrHoldRenderer.cs",
    "HoldBehavior.cs",
)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load_root(path: Path) -> dict:
    return json.loads(
        _normalize_loose_json(path.read_text(encoding="utf-8-sig")),
        strict=False,
    )


def inventory(train_dir: Path, evidence_dir: Path) -> dict:
    charts = []
    for path in sorted(train_dir.glob("*.adofai")):
        root = _load_root(path)
        actions = tuple(
            action for action in root.get("actions", ())
            if isinstance(action, dict)
        )
        holds = tuple(
            action for action in actions
            if action.get("eventType") == "Hold"
            and action.get("active", True) is not False
        )
        if not holds:
            continue
        by_floor: dict[int, list[str]] = {}
        for action in actions:
            floor = action.get("floor")
            if isinstance(floor, int):
                by_floor.setdefault(floor, []).append(
                    str(action.get("eventType", "<unknown>"))
                )
        settings = root.get("settings") or {}
        charts.append({
            "filename": path.name,
            "sha256": _sha(path),
            "initial_bpm": settings.get("bpm"),
            "pitch_percent": settings.get("pitch", 100),
            "angle_count": len(root.get("angleData") or ()),
            "holds": [{
                "floor": action.get("floor"),
                "duration_raw": action.get("duration"),
                "distance_multiplier_raw": action.get("distanceMultiplier"),
                "landing_animation_raw": action.get("landingAnimation"),
                "same_floor_actions": by_floor.get(action.get("floor"), []),
            } for action in holds],
        })

    evidence = []
    for name in EVIDENCE_FILES:
        path = evidence_dir / name
        evidence.append({
            "filename": name,
            "exists": path.is_file(),
            "sha256": _sha(path) if path.is_file() else None,
            "last_write_time": (
                datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat() if path.is_file() else None
            ),
        })
    durations = [
        hold["duration_raw"] for chart in charts for hold in chart["holds"]
    ]
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Train80 only; no Validation20/Reserve20/FINAL",
        "train_dir": str(train_dir),
        "evidence_dir": str(evidence_dir),
        "evidence_version_caveat": (
            "decompiled source has AssemblyVersion 0.0.0.0; hashes identify "
            "the audited snapshot but do not prove parity with another install"
        ),
        "evidence_files": evidence,
        "hold_chart_count": len(charts),
        "hold_action_count": len(durations),
        "duration_values": sorted(set(durations), key=lambda value: str(value)),
        "charts": charts,
    }


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.train_dir, args.evidence_dir)
    _write_new(args.output, result)
    print("hold charts:", result["hold_chart_count"])
    print("hold actions:", result["hold_action_count"])
    print("durations:", result["duration_values"])


if __name__ == "__main__":
    main()
