from __future__ import annotations

from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import diagnose_four_key_teacher_capacity as diag
from dmdod.adofai_chart import parse_adofai_text
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import compile_adofai


def _segment(*, bpm: float = 120.0):
    chart = parse_adofai_text(
        f"""
        {{
          "angleData": [0, 90, 180, 270, 0, 90, 180, 270, 0],
          "settings": {{
            "bpm": {bpm},
            "pitch": 100,
            "countdownTicks": 0,
            "separateCountdownTime": false
          }},
          "actions": []
        }}
        """
    )
    compiled = compile_adofai(chart)
    return build_playable_segment(compiled, start_s=0.0, end_s=compiled.duration_s)


def test_peak_targets_in_window_counts_dense_burst() -> None:
    segment = _segment(bpm=600.0)
    assert diag._peak_targets_in_window(segment, 0.2) >= 3
    assert diag._min_target_gap_ms(segment) is not None


def test_capacity_diagnostic_reports_consistent_teacher_accounting() -> None:
    result = diag.diagnose_four_key_teacher_capacity(
        _segment(bpm=240.0),
        lead_s=0.044,
        control_dt_s=0.010,
    )

    assert result.stats.targets > 0
    assert result.physical_keydowns >= result.stats.hits
    assert 0 <= result.max_reservations <= 4
    assert 0 <= result.max_pressed_keys <= 4
    assert result.missed_targets == result.stats.misses
    assert (
        result.missed_after_capacity_block + result.missed_without_capacity_block
        == result.missed_targets
    )


def test_capacity_diagnostic_formatter_exposes_bottleneck_fields() -> None:
    result = diag.diagnose_four_key_teacher_capacity(
        _segment(bpm=240.0),
        lead_s=0.044,
        control_dt_s=0.010,
    )
    text = diag._format_result("unit", result)

    assert "peak/lead=" in text
    assert "reservation-cap=" in text
    assert "release-wait=" in text
    assert "after-capacity-block=" in text
