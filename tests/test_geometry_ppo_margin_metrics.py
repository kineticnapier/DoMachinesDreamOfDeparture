from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_geometry_ppo_v035 as trainer  # noqa: E402


def test_accuracy_terms_use_hit_margin_denominators():
    stats = SimpleNamespace(
        targets=1,
        perfects=1,
        x_accuracy_percent=60.0,
        x_accuracy_points=1.2,
        x_accuracy_denominator=2,
        hit_margin_count=2,
    )

    xpoints, xden, perfects, ppden = trainer._accuracy_terms(stats)
    assert xpoints == pytest.approx(1.2)
    assert xden == 2
    assert perfects == 1
    assert ppden == 2
    assert 100.0 * xpoints / xden == pytest.approx(60.0)
    assert perfects / ppden == pytest.approx(0.5)


def test_accuracy_terms_keep_pre_dll_stats_compatible():
    stats = SimpleNamespace(
        targets=4,
        perfects=3,
        x_accuracy_percent=87.5,
    )

    xpoints, xden, perfects, ppden = trainer._accuracy_terms(stats)
    assert xpoints == pytest.approx(3.5)
    assert xden == 4
    assert perfects == 3
    assert ppden == 4


def test_training_env_uses_stronger_stray_too_early_penalty():
    env = trainer.make_env(
        bpm=180.0,
        notes=1,
        start_s=0.75,
        control_dt=0.010,
        config=trainer.base.core.clean_vision_config(),
        seed=1,
    )
    assert env._base.reward_config.too_early_penalty == pytest.approx(
        trainer.TOO_EARLY_PENALTY
    )
    assert trainer.TOO_EARLY_PENALTY > 0.20
