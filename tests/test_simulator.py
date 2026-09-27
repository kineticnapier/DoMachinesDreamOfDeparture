import pytest

from dmdod import KeyEvent, Simulation


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
