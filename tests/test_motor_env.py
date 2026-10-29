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

