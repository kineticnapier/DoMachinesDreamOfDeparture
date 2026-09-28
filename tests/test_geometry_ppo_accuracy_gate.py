from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_geometry_ppo as trainer  # noqa: E402


def _slice(
    bpm: float,
    *,
    hit: float = 1.0,
    full: float = 1.0,
    xacc: float = 100.0,
    pp: float = 1.0,
) -> trainer.AccuracyBpmSlice:
    episodes = 10
    targets = 100
    return trainer.AccuracyBpmSlice(
        bpm=bpm,
        episodes=episodes,
        hits=round(hit * targets),
        targets=targets,
        full=round(full * episodes),
        clean=episodes,
        errors_ms=(),
        x_accuracy_percent=xacc,
        perfect_rate=pp,
    )


def _probe(
    *,
    xacc: float,
    pp: float,
    slices: tuple[trainer.AccuracyBpmSlice, ...],
    clean: int = 20,
) -> trainer.AccuracyProbe:
    episodes = sum(item.episodes for item in slices)
    targets = sum(item.targets for item in slices)
    return trainer.AccuracyProbe(
        episodes=episodes,
        hits=sum(item.hits for item in slices),
        targets=targets,
        full=sum(item.full for item in slices),
        clean=clean,
        overloads=0,
        too_early=0,
        errors_ms=(),
        bpm_slices=slices,
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


def test_rank_prefers_accuracy_over_legacy_clean_metric():
    old = _probe(
        xacc=95.0,
        pp=0.80,
        clean=20,
        slices=(
            _slice(120.0, xacc=75.0, pp=0.0),
            _slice(300.0, xacc=100.0, pp=1.0),
        ),
    )
    improved = _probe(
        xacc=98.0,
        pp=0.92,
        clean=10,
        slices=(
            _slice(120.0, xacc=90.0, pp=0.60),
            _slice(300.0, xacc=100.0, pp=1.0),
        ),
    )

    assert trainer.rank_key(improved, trainer.core.Retention(())) > trainer.rank_key(
        old, trainer.core.Retention(())
    )


def test_precision_gate_applies_to_multi_note_phases(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(trainer, "PRECISION_MIN_BPM_XACC", 90.0)
    monkeypatch.setattr(trainer, "PRECISION_OVERALL_XACC", 95.0)
    monkeypatch.setattr(trainer, "PRECISION_MIN_BPM_PP", 0.60)
    monkeypatch.setattr(trainer, "PRECISION_OVERALL_PP", 0.80)

    bad = _probe(
        xacc=97.0,
        pp=0.85,
        slices=(
            _slice(120.0, xacc=89.0, pp=0.70),
            _slice(300.0, xacc=100.0, pp=1.0),
        ),
    )
    good = _probe(
        xacc=97.0,
        pp=0.85,
        slices=(
            _slice(120.0, xacc=90.0, pp=0.60),
            _slice(300.0, xacc=100.0, pp=1.0),
        ),
    )

    assert not trainer.precision_passes(bad, _args(), notes=16)
    assert trainer.precision_passes(good, _args(), notes=16)


def test_completion_gate_rejects_bad_single_bpm_even_when_average_passes(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(trainer, "COMPLETION_MIN_BPM_HIT", 0.95)
    monkeypatch.setattr(trainer, "COMPLETION_MIN_BPM_FULL", 0.90)

    probe = _probe(
        xacc=98.0,
        pp=0.90,
        slices=(
            _slice(120.0, hit=1.0, full=1.0),
            _slice(300.0, hit=0.90, full=0.80),
        ),
    )

    assert probe.hit_rate == pytest.approx(0.95)
    assert probe.full_rate == pytest.approx(0.90)
    assert not trainer.completion_passes(probe, trainer.core.Retention(()), _args(), notes=16)


def test_pp_gate_can_be_raised_to_true_pp(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(trainer, "PRECISION_MIN_BPM_XACC", 100.0)
    monkeypatch.setattr(trainer, "PRECISION_OVERALL_XACC", 100.0)
    monkeypatch.setattr(trainer, "PRECISION_MIN_BPM_PP", 1.0)
    monkeypatch.setattr(trainer, "PRECISION_OVERALL_PP", 1.0)

    almost = _probe(
        xacc=99.9,
        pp=0.99,
        slices=(_slice(120.0, xacc=99.9, pp=0.99),),
    )
    pp = _probe(
        xacc=100.0,
        pp=1.0,
        slices=(_slice(120.0, xacc=100.0, pp=1.0),),
    )

    assert not trainer.precision_passes(almost, _args(), notes=16)
    assert trainer.precision_passes(pp, _args(), notes=16)
