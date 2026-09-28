from __future__ import annotations

import random
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_geometry_ppo_v036 as trainer  # noqa: E402


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


def _probe(*slices):
    targets = sum(item.targets for item in slices)
    xacc = sum(item.x_accuracy_percent * item.targets for item in slices) / targets
    pp = sum(item.perfect_rate * item.targets for item in slices) / targets
    return trainer.base.AccuracyProbe(
        episodes=sum(item.episodes for item in slices),
        hits=targets,
        targets=targets,
        full=sum(item.episodes for item in slices),
        clean=sum(item.episodes for item in slices),
        overloads=0,
        too_early=0,
        errors_ms=(),
        bpm_slices=tuple(slices),
        x_accuracy_percent=xacc,
        perfect_rate=pp,
    )


def _phase(name: str = "p3"):
    return trainer.base.core.CurriculumPhase(
        name,
        1,
        145.0,
        240.0,
        0.0,
        0.0,
        trainer.base.core.clean_vision_config(),
    )


def test_alternating_bad_edges_remain_jointly_focused_until_three_clear_screens():
    trainer._reset_focus_state()
    trainer._ensure_phase(_phase())

    first = _probe(
        _slice(145.0, xacc=70.0, pp=0.25),
        _slice(168.75),
        _slice(192.5),
        _slice(216.25),
        _slice(240.0),
    )
    assert trainer.weak_bpms(first) == (145.0,)

    # The opposite edge becomes weak. 145 has just recovered, but hysteresis
    # keeps it active so training does not immediately abandon that edge.
    second = _probe(
        _slice(145.0),
        _slice(168.75),
        _slice(192.5),
        _slice(216.25),
        _slice(240.0, xacc=70.0, pp=0.25),
    )
    assert set(trainer.weak_bpms(second)) == {145.0, 240.0}

    third = _probe(
        _slice(145.0),
        _slice(168.75),
        _slice(192.5),
        _slice(216.25),
        _slice(240.0, xacc=70.0, pp=0.25),
    )
    assert set(trainer.weak_bpms(third)) == {145.0, 240.0}

    fourth = _probe(
        _slice(145.0),
        _slice(168.75),
        _slice(192.5),
        _slice(216.25),
        _slice(240.0, xacc=70.0, pp=0.25),
    )
    assert trainer.weak_bpms(fourth) == (240.0,)


def test_two_edge_focus_yields_five_five_two_two_two_default_schedule():
    trainer._reset_focus_state()
    phase = _phase()
    trainer._ensure_phase(phase)
    trainer._FOCUS_WEIGHTS.update({145.0: 1.0, 240.0: 1.0})

    schedule = trainer.focused_training_bpm_schedule(
        phase,
        episodes=16,
        points=5,
        rng=random.Random(1),
        focus_bpms=(145.0, 240.0),
    )
    anchors = trainer.base.core.bpm_points(145.0, 240.0, 5)

    assert len(schedule) == 16
    assert schedule.count(anchors[0]) == 5
    assert schedule.count(anchors[-1]) == 5
    assert [schedule.count(bpm) for bpm in anchors[1:-1]] == [2, 2, 2]


def test_focus_history_resets_when_curriculum_phase_changes_even_with_same_bpm_range():
    trainer._reset_focus_state()
    trainer._ensure_phase(_phase("full-bpm-clean"))
    probe = _probe(
        _slice(145.0, xacc=70.0, pp=0.25),
        _slice(168.75),
        _slice(192.5),
        _slice(216.25),
        _slice(240.0),
    )
    assert trainer.weak_bpms(probe) == (145.0,)
    assert trainer._ACTIVE_FOCUS == {145.0}

    trainer._ensure_phase(_phase("latency-sampling"))
    assert trainer._ACTIVE_FOCUS == set()
    assert trainer._FOCUS_EMA == {}
