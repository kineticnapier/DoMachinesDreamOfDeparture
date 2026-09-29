from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import visualize_real_chart as viz  # noqa: E402


def _sample_data() -> dict:
    return {
        "frames": [[0.0, 0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0]],
        "floors": [[0, 0.0, 0.0, 0, 0]],
        "events": [],
        "segment": {"start": 0.0, "end": 1.0, "duration": 1.0, "targets": 1},
        "result": {
            "hits": 0,
            "misses": 0,
            "xacc": 0.0,
            "pp": 0.0,
            "mae": None,
            "early": 0,
            "overload": False,
            "keydowns": 0,
        },
    }


def test_json_for_script_escapes_script_end_marker():
    encoded = viz._json_for_script({"value": "</script>"})
    assert "</script>" not in encoded
    assert "<\\/script>" in encoded


def test_html_document_embeds_replay_and_source_metadata():
    html = viz._html_document(
        _sample_data(),
        checkpoint="checkpoints/model.pt",
        chart="charts/test.adofai",
    )
    assert "DMDOD Real Chart Replay" in html
    assert "training=DISABLED" in html
    assert "checkpoints/model.pt" in html
    assert "charts/test.adofai" in html
    assert "requestAnimationFrame" in html


def test_finite_replaces_non_finite_values():
    assert viz._finite(float("inf")) == 0.0
    assert viz._finite(float("nan")) == 0.0
