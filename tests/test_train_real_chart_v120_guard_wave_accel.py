from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v120_guard_wave_accel as guard_wave


def test_adaptive_wave_size_targets_one_worker_wave() -> None:
    assert guard_wave._adaptive_wave_size(6, 100, workers=12) == 2
    assert guard_wave._adaptive_wave_size(4, 100, workers=12) == 3
    assert guard_wave._adaptive_wave_size(3, 100, workers=12) == 4
    assert guard_wave._adaptive_wave_size(2, 100, workers=12) == 6
    assert guard_wave._adaptive_wave_size(1, 100, workers=12) == 12


def test_adaptive_wave_size_respects_remaining_and_cap() -> None:
    assert guard_wave._adaptive_wave_size(1, 5, workers=12) == 5
    assert guard_wave._adaptive_wave_size(1, 100, workers=12, max_segments=8) == 8
    assert guard_wave._adaptive_wave_size(6, 1, workers=12) == 1
    assert guard_wave._adaptive_wave_size(6, 0, workers=12) == 0


def test_adaptive_wave_size_never_drops_below_one_for_work() -> None:
    assert guard_wave._adaptive_wave_size(99, 10, workers=12) == 1
    assert guard_wave._adaptive_wave_size(1, 10, workers=1) == 1
