from __future__ import annotations

import pytest

from dmdod.motor.n_key import NKeyAction, NKeyMotorEnv


@pytest.mark.parametrize("key_count", [2, 4, 12])
def test_exact_n_key_held_path_matches_public_1khz_path(key_count: int):
    fast = NKeyMotorEnv(key_count)
    reference = NKeyMotorEnv(key_count)
    reference.capture_physics_trace = True
    commands = [
        tuple(1.0 if index % 3 == 0 else -0.7 if index % 3 == 1 else 0.0
              for index in range(key_count)),
        tuple(-1.0 if index % 2 else 0.35 for index in range(key_count)),
        (0.0,) * key_count,
        (1.0,) * key_count,
        (-1.0,) * key_count,
    ] * 20

    for values in commands:
        actual = fast.step(NKeyAction(values))
        expected = reference.step(NKeyAction(values))
        assert actual.observation == expected.observation
        assert actual.evaluator_events == expected.evaluator_events
        assert actual.diagnostics == expected.diagnostics
        assert actual.physics_samples == ()
        assert fast.time_s == reference.time_s
        assert fast.body.states == reference.body.states
        assert fast.keyboard.states == reference.keyboard.states


def test_n_key_trace_request_keeps_all_1khz_samples():
    env = NKeyMotorEnv(12)
    env.capture_physics_trace = True
    transition = env.step(NKeyAction((0.0,) * 12))
    assert len(transition.physics_samples) == env.physics_substeps
