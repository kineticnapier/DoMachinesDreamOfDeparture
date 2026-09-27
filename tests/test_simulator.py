import pytest

from dmdod import BodyConfig, FingerConfig, KeyEvent, Simulation, TwoFingerBody


def test_zero_command_stays_at_rest():
    sim = Simulation()
    results = sim.run_constant(0.0, 0.0, 0.100)
    assert results[-1].left.position_m == pytest.approx(0.0)
    assert results[-1].right.position_m == pytest.approx(0.0)
    assert not any(result.events for result in results)


def test_press_generates_one_down_until_reset():
    sim = Simulation()
    pressing = sim.run_constant(1.0, 0.0, 0.100)
    down = [event for result in pressing for event in result.events if event == ("left", KeyEvent.DOWN)]
    assert len(down) == 1

    # Keeping the finger down must not create repeated digital presses.
    held = sim.run_constant(1.0, 0.0, 0.100)
    assert ("left", KeyEvent.DOWN) not in [event for result in held for event in result.events]

    releasing = sim.run_constant(-1.0, 0.0, 0.100)
    assert ("left", KeyEvent.UP) in [event for result in releasing for event in result.events]


def test_same_actions_are_deterministic():
    commands = [(1.0, 0.0)] * 40 + [(-1.0, 1.0)] * 40 + [(0.0, -1.0)] * 40

    def execute():
        sim = Simulation()
        return [sim.step(*command) for command in commands]

    assert execute() == execute()


def test_fatigue_is_bounded():
    sim = Simulation()
    results = sim.run_constant(1.0, 1.0, 5.0)
    assert 0.0 <= results[-1].left.fatigue <= 1.0
    assert 0.0 <= results[-1].right.fatigue <= 1.0


def test_reversal_fatigue_cannot_be_bypassed_through_zero():
    finger = FingerConfig(
        fatigue_gain_s=0.0,
        fatigue_recovery_s=0.0,
        switch_fatigue_per_reversal=0.10,
    )
    body = TwoFingerBody(BodyConfig(left=finger, right=finger))

    body.step(1.0, 0.0, 0.001)
    body.step(0.0, 0.0, 0.001)
    body.step(-1.0, 0.0, 0.001)

    assert body.left.fatigue == pytest.approx(0.10)
    assert body.left.last_nonzero_command_sign == -1


def test_returning_to_same_direction_through_zero_has_no_reversal_cost():
    finger = FingerConfig(
        fatigue_gain_s=0.0,
        fatigue_recovery_s=0.0,
        switch_fatigue_per_reversal=0.10,
    )
    body = TwoFingerBody(BodyConfig(left=finger, right=finger))

    body.step(1.0, 0.0, 0.001)
    body.step(0.0, 0.0, 0.001)
    body.step(1.0, 0.0, 0.001)

    assert body.left.fatigue == pytest.approx(0.0)
    assert body.left.last_nonzero_command_sign == 1
