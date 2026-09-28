from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_geometry_ppo_v042 as trainer  # noqa: E402


def _slice(
    bpm: float,
    *,
    hit: float = 1.0,
    full: float = 1.0,
    xacc: float = 100.0,
    pp: float = 1.0,
):
    episodes = 4
    targets = 4
    return trainer.base.AccuracyBpmSlice(
        bpm=bpm,
        episodes=episodes,
        hits=round(targets * hit),
        targets=targets,
        full=round(episodes * full),
        clean=round(episodes * full),
        errors_ms=(),
        x_accuracy_percent=xacc,
        perfect_rate=pp,
    )


def _probe(*slices):
    targets = sum(item.targets for item in slices)
    hits = sum(item.hits for item in slices)
    episodes = sum(item.episodes for item in slices)
    full = sum(item.full for item in slices)
    xacc = sum(item.x_accuracy_percent * item.targets for item in slices) / targets
    pp = sum(item.perfect_rate * item.targets for item in slices) / targets
    return trainer.base.AccuracyProbe(
        episodes=episodes,
        hits=hits,
        targets=targets,
        full=full,
        clean=full,
        overloads=0,
        too_early=0,
        errors_ms=(),
        bpm_slices=tuple(slices),
        x_accuracy_percent=xacc,
        perfect_rate=pp,
    )


def _p4_baseline():
    # Approximate the observed v0.4.0 P4 screen 00:
    # H/F=.95, X~=91.4%, PP~=90.5%, minX=64%, minPP=60%.
    return _probe(
        _slice(135.0, xacc=98.0, pp=1.00),
        _slice(166.25, xacc=98.0, pp=1.00),
        _slice(197.5, xacc=98.0, pp=1.00),
        _slice(228.75, xacc=99.0, pp=0.925),
        _slice(260.0, hit=0.75, full=0.75, xacc=64.0, pp=0.60),
    )


def test_broad_precision_collapse_rolls_back_before_best_selection():
    baseline = _p4_baseline()
    collapsed_clear = _probe(
        _slice(135.0, xacc=95.0, pp=0.80),
        _slice(166.25, xacc=95.0, pp=0.80),
        _slice(197.5, xacc=90.0, pp=0.75),
        _slice(228.75, xacc=90.0, pp=0.745),
        _slice(260.0, xacc=69.0, pp=0.00),
    )
    retention = trainer.core.Retention(())
    args = SimpleNamespace(rollback_drop=0.10)

    assert trainer.material_gate_regression(
        collapsed_clear, retention, baseline, retention
    )
    assert trainer.candidate_decision(
        collapsed_clear, retention, baseline, retention, args
    ) == "rollback"


def test_screen18_style_tradeoff_is_kept_as_real_progress_not_rolled_back():
    baseline = _p4_baseline()
    improved_tradeoff = _probe(
        _slice(135.0, xacc=98.0, pp=1.000),
        _slice(166.25, xacc=98.0, pp=1.000),
        _slice(197.5, xacc=99.0, pp=0.950),
        _slice(228.75, xacc=100.0, pp=0.935),
        _slice(260.0, xacc=74.0, pp=0.400),
    )
    retention = trainer.core.Retention(())
    args = SimpleNamespace(rollback_drop=0.10)

    # This mirrors the real screen 18 shape: completion and XAcc improve while
    # minPP falls. The aggregate worst/total gate deficit is still better.
    assert not trainer.material_gate_regression(
        improved_tradeoff, retention, baseline, retention
    )

    old_rank = trainer.base.rank_key
    try:
        trainer.base.rank_key = trainer.v040.rank_key
        assert trainer.candidate_decision(
            improved_tradeoff, retention, baseline, retention, args
        ) == "best"
    finally:
        trainer.base.rank_key = old_rank


def test_gate_rollback_requires_both_worst_and_total_regression():
    baseline = _p4_baseline()
    retention = trainer.core.Retention(())
    best_worst, best_total = trainer.gate_deficit_score(baseline, retention)

    assert best_worst > 0.0
    assert best_total >= best_worst
    assert trainer.GATE_ROLLBACK_WORST_DELTA > 0.0
    assert trainer.GATE_ROLLBACK_TOTAL_DELTA > trainer.GATE_ROLLBACK_WORST_DELTA
