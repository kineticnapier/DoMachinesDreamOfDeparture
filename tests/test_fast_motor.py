from __future__ import annotations

import pytest

from dmdod.body import TwoFingerBody
from dmdod.fast_motor import step_held_exact
from dmdod.profiles import personal_blue_switch_v0_1
from dmdod.simulator import Simulation


def _simulation(*, same_hand: bool) -> Simulation:
    return Simulation(
        body=TwoFingerBody(
            config=personal_blue_switch_v0_1(same_hand=same_hand),
        )
    )


def _reference_held(
    sim: Simulation,
    left: float,
    right: float,
    substeps: int,
):
    events = []
    for _ in range(substeps):
        result = sim.step(left, right)
        events.extend((result.time_s, key, event) for key, event in result.events)
    return tuple(events)


def _assert_state_matches(fast: Simulation, reference: Simulation) -> None:
    assert fast.time_s == reference.time_s
    assert fast.keyboard.left.pressed is reference.keyboard.left.pressed
    assert fast.keyboard.right.pressed is reference.keyboard.right.pressed

    for fast_state, reference_state in (
        (fast.body.left, reference.body.left),
        (fast.body.right, reference.body.right),
    ):
        assert fast_state.position_m == pytest.approx(reference_state.position_m, abs=1e-15)
        assert fast_state.velocity_m_s == pytest.approx(reference_state.velocity_m_s, abs=1e-15)
        assert fast_state.activation == pytest.approx(reference_state.activation, abs=1e-15)
        assert fast_state.fatigue == pytest.approx(reference_state.fatigue, abs=1e-15)
        assert fast_state.last_command == reference_state.last_command
        assert fast_state.last_nonzero_command_sign == reference_state.last_nonzero_command_sign

    assert fast.body.shared_hand.fatigue == pytest.approx(
        reference.body.shared_hand.fatigue,
        abs=1e-15,
    )
    assert fast.body.shared_hand.coordination == pytest.approx(
        reference.body.shared_hand.coordination,
        abs=1e-15,
    )
    assert fast.body.bilateral_state.coordination == pytest.approx(
        reference.body.bilateral_state.coordination,
        abs=1e-15,
    )


@pytest.mark.parametrize("same_hand", [True, False])
def test_exact_fast_held_path_matches_public_1khz_simulator(same_hand: bool):
    fast = _simulation(same_hand=same_hand)
    reference = _simulation(same_hand=same_hand)
    commands = [
        (1.0, 0.0),
        (1.0, 1.0),
        (-1.0, 1.0),
        (0.0, 0.0),
        (-1.0, -1.0),
        (0.35, -0.72),
        (1.0, -1.0),
        (0.0, 0.0),
    ] * 20

    for left, right in commands:
        expected_events = _reference_held(reference, left, right, 10)
        actual_events = step_held_exact(fast, left, right, 10)
        assert actual_events == expected_events
        _assert_state_matches(fast, reference)


def test_exact_fast_zero_substeps_is_noop():
    sim = _simulation(same_hand=True)
    before_left = sim.body.left.copy()
    before_right = sim.body.right.copy()

    assert step_held_exact(sim, 1.0, -1.0, 0) == ()
    assert sim.time_s == 0.0
    assert sim.body.left == before_left
    assert sim.body.right == before_right
