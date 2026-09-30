from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import eval_v110_strict_dataset as strict_dataset


def _result(
    *,
    hits: int,
    targets: int,
    misses: int,
    early: int,
    x_points: float,
    x_denominator: int,
    perfects: int,
    margin_count: int,
    mae: float | None,
    keydowns: int,
    overloaded: bool = False,
):
    return SimpleNamespace(
        stats=SimpleNamespace(
            hits=hits,
            targets=targets,
            misses=misses,
            too_early_presses=early,
            x_accuracy_points=x_points,
            x_accuracy_denominator=x_denominator,
            perfects=perfects,
            hit_margin_count=margin_count,
            mean_abs_error_ms=mae,
            overloaded=overloaded,
        ),
        physical_keydowns=keydowns,
    )


def test_aggregate_uses_global_x_pp_and_hit_weighted_mae() -> None:
    results = [
        _result(
            hits=9,
            targets=10,
            misses=1,
            early=2,
            x_points=8.0,
            x_denominator=10,
            perfects=7,
            margin_count=10,
            mae=20.0,
            keydowns=12,
        ),
        _result(
            hits=1,
            targets=2,
            misses=1,
            early=1,
            x_points=0.5,
            x_denominator=2,
            perfects=0,
            margin_count=2,
            mae=50.0,
            keydowns=3,
            overloaded=True,
        ),
    ]

    metrics = strict_dataset._aggregate(results)

    assert metrics.charts == 2
    assert metrics.hits == 10
    assert metrics.targets == 12
    assert metrics.misses == 2
    assert metrics.too_early_presses == 3
    assert metrics.physical_keydowns == 15
    assert metrics.overloaded
    assert abs(metrics.x_accuracy_percent - (100.0 * 8.5 / 12.0)) < 1e-9
    assert abs(metrics.perfect_rate - (7.0 / 12.0)) < 1e-9
    assert abs(metrics.mean_abs_error_ms - 23.0) < 1e-9


def test_manifest_difficulty_mapping_accepts_windows_paths(tmp_path: Path) -> None:
    manifest = {
        "Validation": [
            {
                "path": "Validation\\00123 - Example - Charter.adofai",
                "difficulty_name": "P7",
            },
            {
                "path": "Validation/00456 - Other - Charter.adofai",
                "difficulty_name": "P10",
            },
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    mapping = strict_dataset._load_difficulty_by_chart_name(tmp_path, "Validation")

    assert mapping == {
        "00123 - Example - Charter": "P7",
        "00456 - Other - Charter": "P10",
    }


def test_p_difficulty_sort_is_numeric() -> None:
    values = ["P10", "P8", "P7", "P9"]
    assert sorted(values, key=strict_dataset._difficulty_sort_key) == ["P7", "P8", "P9", "P10"]
