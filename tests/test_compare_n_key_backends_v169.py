from __future__ import annotations

from types import SimpleNamespace

from scripts.compare_n_key_backends_v169 import summarize


def _stats(*, targets, hits, early, overloaded, x_points, x_den):
    return SimpleNamespace(
        targets=targets,
        hits=hits,
        too_early_presses=early,
        overloaded=overloaded,
        x_accuracy_points=x_points,
        x_accuracy_denominator=x_den,
    )


def test_summary_normalizes_backend_metrics_by_common_targets() -> None:
    results = [
        (_stats(targets=100, hits=25, early=5, overloaded=False, x_points=40, x_den=80), 30),
        (_stats(targets=50, hits=10, early=4, overloaded=True, x_points=12, x_den=20), 20),
    ]

    summary = summarize("fly_connectome", results)

    assert summary.hits == 35
    assert summary.targets == 150
    assert summary.hit_rate_percent == 100.0 * 35 / 150
    assert summary.x_accuracy_percent == 52.0
    assert summary.too_early_per_target == 9 / 150
    assert summary.overload_charts == 1
    assert summary.chart_count == 2
    assert summary.keydowns_per_target == 50 / 150
