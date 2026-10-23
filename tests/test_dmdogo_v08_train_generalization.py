from __future__ import annotations

import sys
from pathlib import Path
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from select_dmdogo_v08_train_generalization import (
    BOOL_SHA, describe_events, make_manifest, select_cases,
)


def entry(i: int, *, chords: int = 0, gap: float = 50.,
          ratio: float = 1., size: int = 100) -> dict:
    return {
        "chart": f"chart-{i}.adofai",
        "path": f"data/dataset/Train/chart-{i}.adofai",
        "sha256": f"{i:064x}", "event_count": size,
        "duration_s": 30., "chord_events": chords,
        "chord_fraction": chords / size,
        "min_inter_onset_ms": gap, "p10_gap_ms": gap,
        "p90_to_p10_gap_ratio": ratio,
        "median_gap_ms": gap,
    }


def test_chord_density_gap_and_irregularity_are_distinct():
    samples = [
        entry(1, chords=40),
        entry(2, gap=8.),
        entry(3, ratio=8.),
        entry(4, chords=10, gap=10., ratio=2.),
    ]
    cases = select_cases(samples)
    assert [e["sha256"] for e in cases] == [
        f"{i:064x}" for i in (1, 2, 3)]
    assert [e["selection_reason"] for e in cases] == [
        "simultaneous_contacts", "small_inter_onset_gaps", "uneven_timing"]
    assert select_cases(list(reversed(samples))) == cases


def test_dedup_sha_and_exclude_frozen_bool():
    a = entry(1)
    b = {**entry(2), "sha256": BOOL_SHA}
    duplicate = {**a, "chart": "alias.adofai",
                 "path": "data/dataset/Train/alias.adofai"}
    cases = select_cases([a, b, duplicate])
    assert len(cases) == 1
    assert cases[0]["sha256"] == a["sha256"]


def test_manifest_marks_oracle_selection_only_no_physical_scores():
    manifest = make_manifest([entry(1)], [
        {"path": "broken.adofai", "reason": "ValueError"}
    ], train_root="data/dataset/Train")
    assert manifest["execution_status"] == "selection_only; no physical evaluation performed"
    assert manifest["train_charts_screened"] == 2
    assert manifest["train_charts_parsed"] == 1
    assert len(manifest["cases"]) == 1
    assert "physical_result" not in manifest


def test_short_charts_excluded():
    assert select_cases([entry(1, size=4), entry(2, size=11)]) == []


def test_count_parameter_validation():
    with pytest.raises(ValueError):
        select_cases([entry(1)], count=0)


def test_fingerprinting_reports_true_chords_and_no_fabricated_metrics():
    class Event:
        def __init__(self, s, c):
            self.episode_time_s = s
            self.chord_size = c
    result = describe_events(
        (Event(0., 1), Event(.01, 2), Event(.21, 1)),
        name="test.adofai", path="Train/test.adofai", sha256="a" * 64,
    )
    assert result["event_count"] == 3
    assert result["chord_events"] == 1
    assert result["p10_gap_ms"] == 29.
    assert result["p90_to_p10_gap_ratio"] == pytest.approx(181. / 29., abs=1e-6)
    assert "hits" not in result
