from __future__ import annotations

import pytest

from dmdod.motor_env import MotorAction
from dmdod.pattern_memory import PatternMemory
from dmdod.planet_perception import PlanetGeometryObservation


def _frames():
    return (
        PlanetGeometryObservation(1.0, 0.0, 1.0, 0.0),
        PlanetGeometryObservation(0.98, 0.20, 1.0, 0.0),
        PlanetGeometryObservation(0.92, 0.39, 1.0, 0.0),
    )


def _lookup_sequence(memory: PatternMemory, chart_id: str):
    memory.begin_episode(chart_id)
    lookup = None
    for frame in _frames():
        lookup = memory.observe(frame)
    assert lookup is not None
    return lookup


def test_pattern_memory_reuses_typical_pattern_and_keeps_chart_specific_layer():
    memory = PatternMemory(history_frames=3, ema_alpha=1.0)
    first = _lookup_sequence(memory, "chart-a")
    assert first.features.shared_confidence == 0.0
    assert first.features.chart_confidence == 0.0

    memory.learn(
        first.keys,
        error_ms=25.0,
        action=MotorAction(-0.6, 0.2),
    )

    same_chart = _lookup_sequence(memory, "chart-a")
    assert same_chart.features.shared_confidence > 0.0
    assert same_chart.features.chart_confidence > 0.0
    assert same_chart.features.shared_timing_correction == pytest.approx(-0.25)
    assert same_chart.features.chart_timing_correction == pytest.approx(-0.25)
    assert same_chart.features.shared_action_left == pytest.approx(-0.6)
    assert same_chart.features.shared_action_right == pytest.approx(0.2)

    other_chart = _lookup_sequence(memory, "chart-b")
    assert other_chart.features.shared_confidence > 0.0
    assert other_chart.features.chart_confidence == 0.0
    assert other_chart.features.shared_timing_correction == pytest.approx(-0.25)


def test_pattern_memory_exposes_two_frame_motion_without_exact_time():
    memory = PatternMemory(history_frames=3)
    memory.begin_episode("chart")
    first, second, _ = _frames()
    a = memory.observe(first)
    b = memory.observe(second)

    assert a.features.delta_orbit_x == 0.0
    assert a.features.delta_orbit_y == 0.0
    assert b.features.delta_orbit_x == pytest.approx(-0.02)
    assert b.features.delta_orbit_y == pytest.approx(0.20)
    assert b.features.delta_next_x == 0.0
    assert b.features.delta_next_y == 0.0


def test_pattern_memory_checkpoint_roundtrip_preserves_shared_and_chart_entries():
    memory = PatternMemory(history_frames=3, ema_alpha=1.0)
    lookup = _lookup_sequence(memory, "chart-a")
    memory.learn(
        lookup.keys,
        error_ms=-40.0,
        action=MotorAction(0.3, -0.4),
    )

    restored = PatternMemory(history_frames=3, ema_alpha=1.0)
    restored.load_state_dict(memory.state_dict())
    again = _lookup_sequence(restored, "chart-a")

    assert restored.shared_entry_count == memory.shared_entry_count
    assert restored.chart_entry_count == memory.chart_entry_count
    assert again.features.shared_timing_correction == pytest.approx(0.40)
    assert again.features.chart_timing_correction == pytest.approx(0.40)
    assert again.features.chart_action_left == pytest.approx(0.3)
    assert again.features.chart_action_right == pytest.approx(-0.4)
