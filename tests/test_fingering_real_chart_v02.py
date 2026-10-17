from __future__ import annotations

import json
from pathlib import Path

import pytest

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.training.fingering_real_chart_v02 import (
    excluded_family_reason,
    load_train_manifest,
    select_dataset_windows,
    select_complex_windows,
)
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher


def _chart(interval_bpm: float = 2000.0):
    raw = json.dumps({
        "angleData": [0.0] * 14,
        "settings": {"bpm": interval_bpm, "countdownTicks": 0},
        "actions": [
            {"floor": 5, "eventType": "SetSpeed", "speedType": "Multiplier", "bpmMultiplier": 2},
            {"floor": 8, "eventType": "Twirl"},
        ],
    })
    return compile_adofai(parse_adofai_text(raw))


def test_complex_windows_are_deterministic_non_overlapping_and_sha_bound():
    compiled = _chart()
    events = AutoplayTimingTeacher.from_segment(build_playable_segment(compiled)).events
    first = select_complex_windows(
        chart_sha256="a" * 64, events=events, compiled=compiled,
        window_s=.12, max_windows=3,
    )
    second = select_complex_windows(
        chart_sha256="a" * 64, events=events, compiled=compiled,
        window_s=.12, max_windows=3,
    )
    assert first == second and first
    assert all(row.chart_sha256 == "a" * 64 for row in first)
    assert all(left.end_s <= right.start_s for left, right in zip(first, first[1:]))
    assert any("short_interval" in row.reasons for row in first)
    assert all(row.events >= 1 and row.score < 100 for row in first)


def test_manifest_loader_rejects_non_train_paths(tmp_path: Path):
    (tmp_path / "Train").mkdir()
    (tmp_path / "Final").mkdir()
    outside = tmp_path / "Final" / "sealed.adofai"
    outside.write_text("{}", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "Train": [{"path": "Final/sealed.adofai"}],
        "Final": [{"path": "Final/sealed.adofai"}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="non-Train"):
        load_train_manifest(tmp_path)


def test_inochi_and_nerfed_variants_are_excluded_as_families():
    assert excluded_family_reason("命日.adofai", {"song": "命日"}) == "excluded_family_inochi"
    assert excluded_family_reason("Song (Nerfed).adofai", {"song": "Song"}) == "excluded_nerfed_variant"
    assert excluded_family_reason("Unrelated.adofai", {"song": "Unrelated"}) is None


def test_pilot_selection_is_chart_disjoint_and_keeps_counterexample():
    charts = []
    for index, minimum in enumerate((20.0, 12.0, 5.0, 18.0, 25.0)):
        digest = f"{index + 1:064x}"
        charts.append({
            "name": f"c{index}", "path": f"Train/c{index}.adofai", "sha256": digest,
            "family": f"f{index}", "difficulty": "P20", "g10_or_harder_verified": False,
            "excluded_reason": None,
            "complex_windows": [{
                "chart_sha256": digest, "start_s": index * 2.0, "end_s": index * 2.0 + 1.0,
                "first_event": 1, "last_event": 4, "score": 10.0 + index,
                "reasons": ["sub_control_interval"] if minimum < 10 else ["burst"],
                "events": 4, "min_interval_ms": minimum,
            }],
        })
    selected = select_dataset_windows({"charts": charts}, max_windows=4)
    assert len(selected) == 4
    assert len({row["sha256"] for row in selected}) == 4
    assert any(row["min_interval_ms"] < 10 for row in selected)
