"""Diagnostic-only: separate motor determinism from future teacher command changes."""
import copy

from dmdod.motor.contact_16k import ContactAction, Coupled16ContactMotor, PHYSICAL_AXES
from dmdod.motor.keyboard import KeyEvent


def _action(*pressed):
    values = [-1.] * len(PHYSICAL_AXES)
    for axis in pressed:
        values[PHYSICAL_AXES.index(axis)] = 1.
    return ContactAction(tuple(values))


def _downs(motor, commands):
    events = []
    for command in commands:
        step = motor.step(command)
        events.extend((event.key, round(event.time_s * 1000))
                      for event in step.logical_events if event.event is KeyEvent.DOWN)
    return events


def test_a_change_of_future_command_shifts_keydown_without_changing_motor():
    start = Coupled16ContactMotor()
    frozen, changed, repeated = (copy.deepcopy(start) for _ in range(3))
    press = _action("left_1")
    idle = _action()
    planned = [press] * 15
    delayed = [idle] * 2 + [press] * 13
    forecast = _downs(frozen, planned)
    actual = _downs(changed, delayed)
    replay = _downs(repeated, planned)
    assert forecast == replay
    forecast_ms = next(t for k, t in forecast if k == "left_1")
    actual_ms = next(t for k, t in actual if k == "left_1")
    assert actual_ms > forecast_ms
    # The 20ms control delay changes activation and velocity; it need not
    # produce exactly 20ms of KeyDown delay (observed: 53ms).
    assert actual_ms - forecast_ms == 53


def test_changed_other_axis_changes_command_history_but_identical_replays_match():
    start = Coupled16ContactMotor()
    press_both = _action("left_1", "left_2")
    release_second = _action("left_1")
    actions = [press_both] * 5 + [release_second] * 10
    a, b = copy.deepcopy(start), copy.deepcopy(start)
    assert _downs(a, actions) == _downs(b, actions)
