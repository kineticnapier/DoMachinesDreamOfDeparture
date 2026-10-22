from __future__ import annotations

import pytest

from dmdod.motor.n_key import NKeyAction, NKeyMotorEnv
from dmdod.research.state_aware_motor_latency_v01 import (
    EnvelopeStatus,
    execute_immediate_press,
    predict_latency_envelope,
)


def command(env: NKeyMotorEnv, axis: str, value: float) -> NKeyAction:
    return NKeyAction(tuple(value if key == axis else 0.0 for key in env.key_names))


@pytest.mark.parametrize("axis", ["left_1", "left_3", "right_2", "right_6"])
@pytest.mark.parametrize("precondition", ["cold", "preactivated", "braked", "reversing"])
def test_state_envelope_contains_independent_immediate_rollout(axis: str, precondition: str) -> None:
    env = NKeyMotorEnv(12, control_dt_s=0.010, physics_dt_s=0.001)
    env.reset()
    if precondition == "preactivated":
        env.step(command(env, axis, 1.0))
    elif precondition == "braked":
        env.step(command(env, axis, -1.0))
    elif precondition == "reversing":
        env.step(command(env, axis, 1.0))
        env.step(command(env, axis, -1.0))
    envelope = predict_latency_envelope(
        env, axis, exact_full_snapshot=True, reservation_conflict_free=True
    )
    start = env.diagnostics().time_s
    actual = start + execute_immediate_press(env, axis)
    assert envelope.status is EnvelopeStatus.VALID
    assert envelope.contains(actual)
    # A one-control-tick issue delay can move the discrete contact crossing by
    # one additional 1ms physics sample in a reversing state.
    assert envelope.uncertainty_s <= 0.011 + 1e-12


def test_unknown_is_returned_without_state_or_conflict_proof() -> None:
    env = NKeyMotorEnv(12)
    assert predict_latency_envelope(
        env, "left_1", exact_full_snapshot=False, reservation_conflict_free=True
    ).status is EnvelopeStatus.UNKNOWN
    assert predict_latency_envelope(
        env, "left_1", exact_full_snapshot=True, reservation_conflict_free=None
    ).status is EnvelopeStatus.UNKNOWN


def test_pressed_axis_requires_release_model() -> None:
    env = NKeyMotorEnv(12)
    for _ in range(10):
        transition = env.step(command(env, "left_1", 1.0))
        if env.observe().pressed("left_1"):
            break
    assert transition.evaluator_events
    result = predict_latency_envelope(
        env, "left_1", exact_full_snapshot=True, reservation_conflict_free=True
    )
    assert result.status is EnvelopeStatus.UNKNOWN
    assert "release" in result.reason
