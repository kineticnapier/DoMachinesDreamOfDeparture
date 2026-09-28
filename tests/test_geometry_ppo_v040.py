from __future__ import annotations

import math
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_geometry_ppo_v040 as trainer  # noqa: E402


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


def test_rank_does_not_trade_large_precision_collapse_for_one_clear_gain():
    # Shape mirrors the v0.3.9 P4 failure: the baseline has one weak completion
    # slice but strong Perfect rate, while PPO clears everything by destroying
    # timing quality on an interior BPM.
    baseline = _probe(
        _slice(135.0, xacc=98.0, pp=1.0),
        _slice(166.25, xacc=98.0, pp=1.0),
        _slice(197.5, xacc=98.0, pp=1.0),
        _slice(228.75, xacc=99.0, pp=0.95),
        _slice(260.0, hit=0.75, full=0.75, xacc=64.0, pp=0.60),
    )
    degraded_clear = _probe(
        _slice(135.0, xacc=95.0, pp=0.90),
        _slice(166.25, xacc=90.0, pp=0.80),
        _slice(197.5, xacc=85.0, pp=0.50),
        _slice(228.75, xacc=75.0, pp=0.00),
        _slice(260.0, xacc=69.0, pp=0.90),
    )
    retention = trainer.base.core.Retention(())

    assert baseline.hit_rate < degraded_clear.hit_rate
    assert baseline.full_rate < degraded_clear.full_rate
    assert trainer.rank_key(baseline, retention) > trainer.rank_key(
        degraded_clear, retention
    )


def test_precision_guard_rolls_back_large_pp_or_xacc_regressions():
    best = _probe(
        _slice(145.0, xacc=90.0, pp=0.80),
        _slice(192.5, xacc=96.0, pp=0.95),
        _slice(240.0, xacc=90.0, pp=0.80),
    )
    pp_collapse = _probe(
        _slice(145.0, xacc=90.0, pp=0.80),
        _slice(192.5, xacc=92.0, pp=0.40),
        _slice(240.0, xacc=90.0, pp=0.80),
    )
    small_noise = _probe(
        _slice(145.0, xacc=88.0, pp=0.75),
        _slice(192.5, xacc=95.0, pp=0.92),
        _slice(240.0, xacc=88.0, pp=0.75),
    )

    assert trainer.precision_regressed(pp_collapse, best)
    assert not trainer.precision_regressed(small_noise, best)


def test_behavior_anchor_decays_from_half_to_zero_over_sixteen_updates():
    old = trainer._PHASE_PPO_UPDATES
    try:
        trainer._PHASE_PPO_UPDATES = 1
        assert math.isclose(trainer._anchor_coef(), 0.50)
        trainer._PHASE_PPO_UPDATES = 9
        assert math.isclose(trainer._anchor_coef(), 0.25)
        trainer._PHASE_PPO_UPDATES = 17
        assert math.isclose(trainer._anchor_coef(), 0.0)
    finally:
        trainer._PHASE_PPO_UPDATES = old
