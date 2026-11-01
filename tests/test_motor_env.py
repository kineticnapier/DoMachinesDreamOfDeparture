import pytest

from dmdod import LeftThresholdReflexPolicy, MotorAction, MotorEnv


def test_agent_observation_hides_privileged_state():
    env = MotorEnv()
    observation = env.reset()

    assert not hasattr(observation, "time_s")
    assert not hasattr(observation, "left_fatigue")
    assert not hasattr(observation, "right_fatigue")
    assert not hasattr(observation, "coordination")


def test_motor_action_is_clamped_before_body():
    env = MotorEnv(control_dt_s=0.010)
    env.reset()
    transition = env.step(MotorAction(100.0, -100.0))

    assert -1.0 <= transition.diagnostics.left_activation <= 1.0
    assert -1.0 <= transition.diagnostics.right_activation <= 1.0


def test_reflex_policy_generates_physical_down_events():
    env = MotorEnv(control_dt_s=0.010)
    policy = LeftThresholdReflexPolicy()
    observation = env.reset()
    downs = 0

    for _ in range(500):
        transition = env.step(policy.act(observation))
        observation = transition.observation
        downs += sum(1 for event in transition.evaluator_events if event.event.value == "down")

    assert downs > 0
    assert env.diagnostics().left_fatigue > 0.0

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("side", ["left", "right"])
def test_nonfinite_motor_action_rejected_without_advancing_physics(bad, side):
    env = MotorEnv()
    before = env.reset()
    before_diagnostics = env.diagnostics()
    action = MotorAction(bad, 0.0) if side == "left" else MotorAction(0.0, bad)

    with pytest.raises(ValueError, match="finite"):
        env.step(action)

    assert env.observe() == before
    assert env.diagnostics() == before_diagnostics


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_motor_env_rejects_nonfinite_control_clock(bad):
    with pytest.raises(ValueError, match="control_dt_s must be finite"):
        MotorEnv(control_dt_s=bad)


def test_motor_env_preserves_finite_control_clock():
    env = MotorEnv(control_dt_s=0.010)
    assert env.physics_substeps == 10


from dmdod.motor.keyboard import KeyEvent, TwoKeyKeyboard


@pytest.mark.parametrize(
    "bad",
    (float("nan"), float("inf"), float("-inf"), True, "0.02", None, 10**1000),
)
@pytest.mark.parametrize("bad_side", ("left", "right"))
def test_two_key_keyboard_rejects_malformed_positions_atomically(
    bad: object, bad_side: str
) -> None:
    keyboard = TwoKeyKeyboard()
    # The valid key would otherwise be pressed before a malformed right key
    # is inspected; neither key may change after invalid input.
    positions = (bad, 1.0) if bad_side == "left" else (1.0, bad)
    with pytest.raises(ValueError, match="keyboard positions must be finite numbers"):
        keyboard.step(*positions)
    assert not keyboard.left.pressed
    assert not keyboard.right.pressed
    assert keyboard.step(1.0, 1.0) == [
        ("left", KeyEvent.DOWN), ("right", KeyEvent.DOWN)
    ]


@pytest.mark.parametrize(
    "bad",
    (float("nan"), float("inf"), float("-inf"), True, "0.02", None, 10**1000),
)
def test_two_key_keyboard_rejects_malformed_position_without_partial_keyup(
    bad: object,
) -> None:
    keyboard = TwoKeyKeyboard()
    assert keyboard.step(1.0, 1.0) == [
        ("left", KeyEvent.DOWN), ("right", KeyEvent.DOWN)
    ]
    with pytest.raises(ValueError, match="keyboard positions must be finite numbers"):
        keyboard.step(0.0, bad)
    assert keyboard.left.pressed
    assert keyboard.right.pressed
    assert keyboard.step(0.0, 0.0) == [
        ("left", KeyEvent.UP), ("right", KeyEvent.UP)
    ]


@pytest.mark.parametrize("bad", (True, False, "0.010", b"0.010", None, 10**1000))
def test_two_key_motor_env_rejects_malformed_control_clock(bad: object) -> None:
    # isfinite(True) previously accepted one second; a huge int raised
    # OverflowError instead of a configuration ValueError.
    with pytest.raises(ValueError, match="control_dt_s must be finite"):
        MotorEnv(control_dt_s=bad)


def test_two_key_motor_env_rejects_overflowed_clock_ratio() -> None:
    # The clock itself is finite, but dividing by the 1 ms physics step
    # overflows to infinity before round().
    with pytest.raises(ValueError, match="integer multiple of physics dt"):
        MotorEnv(control_dt_s=1e308)


def test_two_key_motor_env_preserves_integer_control_clock() -> None:
    assert MotorEnv(control_dt_s=1).physics_substeps == 1000


@pytest.mark.parametrize("bad", (True, False, "1", b"1", None, 10**1000))
@pytest.mark.parametrize("side", ("left", "right"))
def test_two_key_motor_env_rejects_malformed_commands_without_advancing(
    bad: object, side: str
) -> None:
    env = MotorEnv()
    before = env.reset()
    before_diagnostics = env.diagnostics()
    action = MotorAction(bad, 0.0) if side == "left" else MotorAction(0.0, bad)

    with pytest.raises(ValueError, match="motor commands must be finite"):
        env.step(action)

    assert env.observe() == before
    assert env.diagnostics() == before_diagnostics


def test_two_key_motor_env_preserves_integer_commands() -> None:
    env = MotorEnv()
    env.reset()
    transition = env.step(MotorAction(1, -1))
    assert -1.0 <= transition.diagnostics.left_activation <= 1.0
    assert -1.0 <= transition.diagnostics.right_activation <= 1.0
