from __future__ import annotations

"""Prepare isolated 8x/1x Train-bool packages for a stock icon capture.

The original chart is never modified.  The counterfactual byte stream changes
only the numeric token of Floor 1's unique active SetSpeed bpmMultiplier.
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil


DEFAULT_CHART = Path(
    "data/DMDOD-v140-tuf-p11-p20/Train/04042 - bool - vo[ICE].adofai"
)
EXPECTED_CHART_SHA256 = "04981e92f74288b8aac7286e83420b933036d2caead9c157bc33b706779ad1b1"
FORMAT = "dmdod-stock-setspeed-icon-capture-package-v0.1"


def _load_root(raw: bytes) -> dict:
    text = raw.decode("utf-8-sig")
    root = json.loads(text)
    if not isinstance(root, dict):
        raise ValueError("chart root must be an object")
    return root


def mutate_floor1_multiplier(raw: bytes, value: int) -> bytes:
    """Replace one verified Floor-1 SetSpeed multiplier token in-place."""
    if value not in (1, 8):
        raise ValueError("capture pair is fixed to multipliers 8 and 1")
    root = _load_root(raw)
    candidates = [
        action for action in root.get("actions", [])
        if isinstance(action, dict)
        and action.get("floor") == 1
        and action.get("eventType") == "SetSpeed"
        and action.get("active", True) not in (False, "Disabled")
    ]
    if len(candidates) != 1:
        raise ValueError("expected exactly one active Floor-1 SetSpeed")
    action = candidates[0]
    if action.get("speedType") != "Multiplier" or float(action.get("bpmMultiplier", 0)) != 8.0:
        raise ValueError("expected the fixed Floor-1 8x multiplier")
    if value == 8:
        return raw

    pattern = re.compile(
        rb'("floor"\s*:\s*1\s*,\s*"eventType"\s*:\s*"SetSpeed"'
        rb'(?:(?!\}\s*,?\s*\{).)*?"bpmMultiplier"\s*:\s*)8(?:\.0+)?(?=\s*[,}])'
    )
    changed, count = pattern.subn(rb"\g<1>1", raw)
    if count != 1:
        raise ValueError(f"expected one byte-level multiplier replacement, got {count}")
    _assert_only_multiplier_diff(root, _load_root(changed))
    return changed


def _assert_only_multiplier_diff(original: dict, changed: dict) -> None:
    if original.keys() != changed.keys():
        raise ValueError("counterfactual root keys changed")
    left = json.loads(json.dumps(original))
    right = json.loads(json.dumps(changed))
    differences = []

    def walk(a, b, path=()):
        if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
            for key in a:
                walk(a[key], b[key], path + (key,))
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for index, (x, y) in enumerate(zip(a, b, strict=True)):
                walk(x, y, path + (index,))
        elif a != b:
            differences.append((path, a, b))

    walk(left, right)
    if len(differences) != 1:
        raise ValueError(f"expected one semantic difference, got {differences!r}")
    path, before, after = differences[0]
    if path[-1] != "bpmMultiplier" or float(before) != 8.0 or float(after) != 1.0:
        raise ValueError(f"unexpected counterfactual difference: {differences!r}")
    action = right["actions"][path[-2]]
    if action.get("floor") != 1 or action.get("eventType") != "SetSpeed":
        raise ValueError("changed multiplier is not the Floor-1 SetSpeed")


def prepare(chart_path: Path, output_dir: Path, audio_source: Path | None = None) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    if audio_source is not None and not audio_source.is_file():
        raise FileNotFoundError(audio_source)
    raw = chart_path.read_bytes()
    observed_hash = sha256(raw).hexdigest()
    if observed_hash != EXPECTED_CHART_SHA256:
        raise ValueError(f"Train bool SHA256 mismatch: {observed_hash}")
    root = _load_root(raw)
    song_filename = root.get("settings", {}).get("songFilename")
    if not isinstance(song_filename, str) or not song_filename:
        raise ValueError("chart has no songFilename")

    worlds = {}
    for name, multiplier in (("original-8x", 8), ("counterfactual-1x", 1)):
        world_dir = output_dir / name
        world_dir.mkdir(parents=True)
        chart_output = world_dir / f"Train bool stock icon capture {multiplier}x.adofai"
        payload = mutate_floor1_multiplier(raw, multiplier)
        chart_output.write_bytes(payload)
        audio_output = world_dir / song_filename
        if audio_source is not None:
            shutil.copy2(audio_source, audio_output)
        worlds[name] = {
            "multiplier_evaluator_only": multiplier,
            "chart": str(chart_output.relative_to(output_dir)),
            "chart_sha256": sha256(payload).hexdigest(),
            "audio": str(audio_output.relative_to(output_dir)),
            "audio_copied": audio_output.is_file(),
            "audio_sha256": (
                sha256(audio_output.read_bytes()).hexdigest() if audio_output.is_file() else None
            ),
        }

    manifest = {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_chart": str(chart_path),
        "source_chart_sha256": observed_hash,
        "semantic_difference": "actions[Floor 1 SetSpeed].bpmMultiplier only: 8 -> 1",
        "worlds": worlds,
        "capture_manifest": "capture-manifest.template.json",
    }
    (output_dir / "package.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    template = capture_manifest_template(output_dir)
    (output_dir / "capture-manifest.template.json").write_text(
        json.dumps(template, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def capture_manifest_template(output_dir: Path) -> dict:
    capture = lambda world: {
        "world": world,
        "video_path": None,
        "recording": {
            "recording_utc": None,
            "game_build": None,
            "capture_software": None,
            "capture_mode": None,
            "reported_fps": None,
            "resolution": None,
            "mods_enabled": [],
            "camera_and_display_conditions": None,
        },
        "sync": {
            "anchor_kind": "first_countdown_tick",
            "anchor_video_s": None,
            "anchor_episode_s": 0.0,
            "uncertainty_s": None,
        },
        "capture_latency": {"min_s": None, "max_s": None, "basis": None},
        "annotations": [],
    }
    return {
        "format": "dmdod-stock-setspeed-icon-capture-manifest-v0.1",
        "deadline_episode_s": 3.9761634902576737,
        "crop_xywh": None,
        "allowed_states": [
            "DOUBLE_RABBIT_VISIBLE",
            "TILE_VISIBLE_NO_SPEED_ICON",
            "UNKNOWN_OR_OCCLUDED",
        ],
        "captures": [capture("original-8x"), capture("counterfactual-1x")],
        "notes": [
            "Paths are relative to this manifest unless absolute.",
            "Do not label no-icon unless the tile and expected icon site are both visible.",
            "Do not enter multiplier or compiled icon enum as a sensor observation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chart", type=Path, default=DEFAULT_CHART)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audio-source", type=Path)
    args = parser.parse_args()
    result = prepare(args.chart, args.output_dir, args.audio_source)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
