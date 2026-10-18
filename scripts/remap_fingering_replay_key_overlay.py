from __future__ import annotations

"""Rebuild a saved Fingering Replay HTML with the user-confirmed 16-key labels.

Render-only. Reuses saved Motor frames and KeyDown logs; never loads a checkpoint,
reruns inference, or changes source JSON. Refuses to overwrite an existing viewer.
"""

import argparse
import json
from pathlib import Path

import build_fingering_replay_viewer_v03 as viewer


def regenerate(source_json: Path, output_html: Path) -> dict:
    if source_json.resolve() == output_html.resolve():
        raise ValueError("source and destination must be different")
    if output_html.exists():
        raise FileExistsError(output_html)
    original = json.loads(source_json.read_text(encoding="utf-8"))
    if not isinstance(original, dict) or not isinstance(original.get("key_overlay"), dict):
        raise ValueError("not a Fingering Replay payload with key_overlay")
    if not isinstance(original.get("policies"), dict) or not original["policies"]:
        raise ValueError("missing saved physical policy data")
    if not isinstance(original.get("events"), list):
        raise ValueError("missing replay events")
    payload = dict(original)
    overlay = dict(original["key_overlay"])
    overlay["mapping"] = [
        {"label": label, "contact": contact}
        for label, contact in viewer.KEY_DISPLAY_MAP
    ]
    overlay["mapping_semantics"] = (
        "User-provided key layout mapped to simulator logical contacts; "
        "not a verification of OS keyboard signals."
    )
    payload["key_overlay"] = overlay

    # The policy subtree is preserved by reference; no replay/simulation is performed.
    assert payload["policies"] is original["policies"]
    output_html.parent.mkdir(parents=True, exist_ok=True)
    with output_html.open("x", encoding="utf-8") as handle:
        handle.write(viewer.render_html(payload))
    return {
        "source": str(source_json),
        "output": str(output_html),
        "events": len(payload["events"]),
        "policy_count": len(payload["policies"]),
        "mapping_only": True,
        "inference_run": False,
        "physics_run": False,
        "training_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(regenerate(args.source_json, args.output_html), ensure_ascii=False))


if __name__ == "__main__":
    main()
