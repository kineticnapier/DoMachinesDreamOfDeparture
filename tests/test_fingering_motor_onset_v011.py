from __future__ import annotations

from types import SimpleNamespace

import pytest

from dmdod.motor.contact_16k import Coupled16ContactMotor
from scripts import audit_fingering_motor_onset_v011 as audit


@pytest.mark.parametrize(
    ("gap", "expected"),
    [
        (None, "first_use"),
        (99.999, "reuse_lt_100ms"),
        (100.0, "reuse_100_200ms"),
        (199.999, "reuse_100_200ms"),
        (200.0, "reuse_200_500ms"),
        (499.999, "reuse_200_500ms"),
        (500.0, "reuse_ge_500ms"),
    ],
)
def test_reuse_interval_boundaries(gap, expected):
    assert audit.reuse_band(gap) == expected


def test_summary_separates_control_latency_and_signed_error():
    rows = [
        dict(signed_timing_error_ms=27.0, command_lead_ms=40.0,
             command_to_keydown_ms=67.0, decision_to_command_ms=0.0,
             position_at_command_mm=1.0, velocity_at_command_mm_s=-8.0,
             activation_at_command=-0.25, finger_fatigue_at_command=.1,
             hand_fatigue_at_command=.2, previous_keyup_to_command_ms=30.,
             pressed_at_command=False),
        dict(signed_timing_error_ms=-2.0, command_lead_ms=40.0,
             command_to_keydown_ms=38.0, decision_to_command_ms=0.0,
             position_at_command_mm=0.0, velocity_at_command_mm_s=12.0,
             activation_at_command=.2, finger_fatigue_at_command=.2,
             hand_fatigue_at_command=.4, previous_keyup_to_command_ms=None,
             pressed_at_command=False),
    ]
    result = audit._stats(rows)
    assert result["n"] == 2
    assert result["late"] == 1
    assert result["early"] == 1
    assert result["mean_signed_error_ms"] == pytest.approx(12.5)
    assert result["mean_command_to_keydown_ms"] == pytest.approx(52.5)
    assert result["mean_previous_keyup_to_command_ms"] == 30.0
    assert result["previous_keyup_observed"] == 1
    assert result["moving_toward_actuation_at_command"] == 1


def test_motor_snapshot_is_read_only_and_uses_physical_key_threshold():
    motor = Coupled16ContactMotor()
    fake_env = SimpleNamespace(motor=motor)
    original = tuple(state.position_m for state in motor.motor.body.states)
    snapshot = audit._state(fake_env, "left_1")
    assert snapshot["position_mm"] == 0.0
    assert snapshot["distance_to_actuation_mm"] == pytest.approx(2.0)
    assert snapshot["distance_to_reset_mm"] == pytest.approx(1.8)
    assert snapshot["pressed"] is False
    assert tuple(state.position_m for state in motor.motor.body.states) == original
