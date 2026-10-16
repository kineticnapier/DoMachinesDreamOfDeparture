"""Stage C v0.6.2: cloned motors must replay identical actions and physical events."""
import copy

from dmdod.motor.contact_16k import Coupled16ContactMotor, ContactAction, PHYSICAL_AXES


def _trace(motor, controls):
    trace = []
    for command in controls:
        transition = motor.step(command)
        trace.append((
            round(motor.motor.time_s, 9),
            tuple((e.key, e.event, round(e.time_s, 9))
                  for e in transition.logical_events),
            transition.observation.physical,
        ))
    return trace


def test_same_actions_replay_exactly_across_clones():
    motor = Coupled16ContactMotor()
    n = len(PHYSICAL_AXES)
    idle = ContactAction((-1.,) * n)
    press = [-1.] * n
    press[PHYSICAL_AXES.index("left_1")] = 1.
    first = ContactAction(tuple(press))
    for _ in range(8):
        motor.step(idle)
    a, b = copy.deepcopy(motor), copy.deepcopy(motor)
    actions = [first] * 9 + [idle] * 14 + [first] * 9 + [idle] * 12
    assert _trace(a, actions) == _trace(b, actions)
    assert abs(motor.motor.time_s - .08) < 1e-9


def test_control_transition_contains_1ms_physics_event_timestamps():
    motor = Coupled16ContactMotor()
    n = len(PHYSICAL_AXES)
    commands = [-1.] * n
    commands[PHYSICAL_AXES.index("left_1")] = 1.
    observed = _trace(motor, [ContactAction(tuple(commands))] * 13)
    down = [ev[2] for _, events, _ in observed for ev in events
            if ev[0] == "left_1" and ev[1].value == "down"]
    assert down, "motor never reached KeyDown"
    assert abs(down[0]*1000-round(down[0]*1000)) < 1e-6
    assert observed[0][0] == .01
