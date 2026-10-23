from __future__ import annotations

import pytest

from dmdod.planning.pattern_scheduler_v06 import (
    PatternEvent, PatternFirstScheduler, schedule_metrics,
)


def _events(times, chords=None):
    chords = chords or [1] * len(times)
    return tuple(PatternEvent(i, float(t), int(c))
                 for i, (t, c) in enumerate(zip(times, chords)))


def test_regular_dense_sequence_uses_minimum_valid_cycle():
    scheduler = PatternFirstScheduler(
        logical_keys=("k0", "k1", "k2", "k3"),
        minimum_reuse_s=.080, minimum_cycle_keys=2)
    result = scheduler.schedule(_events([i * .03125 for i in range(12)]))
    assert result.feasible
    assert result.assignments[:6] == (
        ("k0",), ("k1",), ("k2",), ("k0",), ("k1",), ("k2",))
    assert result.patterns[0].width == 3
    assert schedule_metrics(result)["minimum_reuse_ms"] >= 80.0


def test_zero_reuse_still_obeys_declared_structural_cycle_floor():
    result = PatternFirstScheduler(
        logical_keys=("a", "b", "c"), minimum_reuse_s=0.0,
        minimum_cycle_keys=2).schedule(_events([0, .1, .2, .3]))
    assert result.assignments == (("a",), ("b",), ("a",), ("b",))
    assert result.model_contract["minimum_cycle_keys"] == 2
    assert "not a physiological limit" in result.model_contract["cycle_floor_semantics"]


def test_chords_get_distinct_keys_and_reuse_constraint():
    result = PatternFirstScheduler(
        logical_keys=("a", "b", "c", "d"), minimum_reuse_s=.075,
        minimum_cycle_keys=2).schedule(
            _events([0, .05, .10, .15], [2, 1, 2, 1]))
    assert result.feasible
    assert all(len(keys) == len(set(keys)) for keys in result.assignments)
    assert schedule_metrics(result)["reuse_violations"] == 0


def test_bpm_change_selects_new_minimum_width_and_records_switch():
    times = [0, .125, .250, .375, .500, .53125, .5625, .59375, .625]
    result = PatternFirstScheduler(
        logical_keys=("a", "b", "c", "d"), minimum_reuse_s=.080,
        minimum_cycle_keys=2, gap_change_ratio=.20).schedule(_events(times))
    assert [pattern.width for pattern in result.patterns] == [2, 3]
    assert result.pattern_change_indices == (5,)


def test_scheduler_is_deterministic_and_rejects_impossible_chord():
    scheduler = PatternFirstScheduler(
        logical_keys=("a", "b", "c"), minimum_reuse_s=.1,
        minimum_cycle_keys=2)
    events = _events([0, .04, .08, .12])
    assert scheduler.schedule(events) == scheduler.schedule(events)
    impossible = scheduler.schedule(_events([0], [4]))
    assert not impossible.feasible
    assert impossible.failure_reason == "chord_exceeds_logical_key_count"


def test_reuse_history_survives_intervening_single_event_segments():
    result = PatternFirstScheduler(
        logical_keys=("a", "b", "c", "d"), minimum_reuse_s=.080,
        minimum_cycle_keys=2).schedule(_events([0.0, .050, .051, .060]))
    assert result.feasible
    assert schedule_metrics(result)["reuse_violations"] == 0
    assert len({keys[0] for keys in result.assignments}) == 4


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        PatternFirstScheduler(("a", "a"), minimum_reuse_s=.1)
