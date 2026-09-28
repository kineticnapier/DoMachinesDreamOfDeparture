from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_geometry_ppo_v034 as trainer  # noqa: E402


def _slice(bpm: float, *, xacc: float = 100.0, pp: float = 1.0):
    return trainer.base.AccuracyBpmSlice(
        bpm=bpm,
        episodes=4,
        hits=4,
        targets=4,
        full=4,
        clean=4,
        errors_ms=(),
        x_accuracy_percent=xacc,
        perfect_rate=pp,
    )


def _probe(*, episodes: int, slices):
    targets = sum(item.targets for item in slices)
    xacc = sum(item.x_accuracy_percent * item.targets for item in slices) / targets
    pp = sum(item.perfect_rate * item.targets for item in slices) / targets
    return trainer.base.AccuracyProbe(
        episodes=episodes,
        hits=targets,
        targets=targets,
        full=episodes,
        clean=episodes,
        overloads=0,
        too_early=0,
        errors_ms=(),
        bpm_slices=tuple(slices),
        x_accuracy_percent=xacc,
        perfect_rate=pp,
    )


def _args():
    return SimpleNamespace(
        retention_hit_rate=0.70,
        retention_full_rate=0.60,
        stage1_hit_rate=0.95,
        stage1_full_rate=0.95,
        advance_hit_rate=0.90,
        advance_full_rate=0.80,
    )


def test_near_screen_requests_verification_but_full_probe_stays_strict():
    slices = (
        _slice(120.0, xacc=87.5, pp=0.50),
        _slice(165.0),
        _slice(210.0),
        _slice(255.0),
        _slice(300.0),
    )
    quick = _probe(episodes=trainer.base.QUICK_EVAL_EPISODES, slices=slices)
    full = _probe(episodes=50, slices=slices)
    retention = trainer.base.core.Retention(())

    assert trainer.near_precision_passes(quick)
    assert not trainer.base.precision_passes(quick, _args(), notes=1)
    assert trainer.passes(quick, retention, _args(), notes=1)
    assert not trainer.passes(full, retention, _args(), notes=1)


def test_single_real_deficit_gets_75_percent_focus_with_anchor_coverage():
    probe = _probe(
        episodes=20,
        slices=(
            _slice(120.0, xacc=87.5, pp=0.50),
            _slice(165.0),
            _slice(210.0),
            _slice(255.0),
            _slice(300.0),
        ),
    )
    focus = trainer.weak_bpms(probe)
    assert focus == (120.0,)

    phase = trainer.base.core.CurriculumPhase(
        "test",
        1,
        120.0,
        300.0,
        0.0,
        0.0,
        trainer.base.core.clean_vision_config(),
    )
    schedule = trainer.focused_training_bpm_schedule(
        phase,
        episodes=16,
        points=5,
        rng=random.Random(1),
        focus_bpms=focus,
    )

    assert len(schedule) == 16
    assert set(trainer.base.core.bpm_points(120.0, 300.0, 5)).issubset(schedule)
    assert schedule.count(120.0) >= 12
