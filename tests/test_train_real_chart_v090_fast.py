from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v080_fast as v080_fast  # noqa: E402
import train_real_chart_v090 as v090  # noqa: E402
import train_real_chart_v090_fast as fast090  # noqa: E402


def test_fast_v090_installs_eval_path_before_press_persistence(monkeypatch) -> None:
    events: list[object] = []

    monkeypatch.setattr(v080_fast, "_install_fast_path", lambda: events.append("fast"))

    def fake_press_persistence(**kwargs) -> None:
        events.append(("press", kwargs))

    monkeypatch.setattr(v090, "_install_press_persistence", fake_press_persistence)

    fast090._install_v090_fast_path(
        coef=6.0,
        lookahead_frames=4,
        commit_threshold=0.30,
        hold_margin=0.20,
    )

    assert events == [
        "fast",
        (
            "press",
            {
                "coef": 6.0,
                "lookahead_frames": 4,
                "commit_threshold": 0.30,
                "hold_margin": 0.20,
            },
        ),
    ]


def test_fast_v090_reuses_resume_compatible_v080_fast_scheduler() -> None:
    assert v080_fast.FAST_EVAL_VERSION.startswith("v080-")
    assert v080_fast.DEFAULT_ANCHOR_BATCH_SIZE > 0
