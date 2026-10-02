from dataclasses import replace

from dmdod.four_key_motor import (
    CENTER_KEY_NAMES,
    FOUR_KEY_NAMES,
    CenterFirstFourKeyRouter,
    FourKeyAction,
    FourKeyMotorEnv,
)
from dmdod.keyboard import KeyEvent


def _idle_observation():
    return FourKeyMotorEnv().reset()


def test_four_key_observation_hides_privileged_state() -> None:
    observation = _idle_observation()

    assert not hasattr(observation, "time_s")
    assert not hasattr(observation, "left_hand_fatigue")
    assert not hasattr(observation, "bilateral_coordination")


def test_center_first_router_uses_inner_pair_before_outer_overflow() -> None:
    router = CenterFirstFourKeyRouter()
    observation = _idle_observation()

    assert router.choose_key(observation) == "left_inner"

    observation = replace(observation, left_inner_pressed=True)
    assert router.choose_key(observation) == "right_inner"

    observation = replace(
        observation,
        left_inner_pressed=True,
        right_inner_pressed=True,
    )
    assert router.choose_key(observation) == "left_outer"

    observation = replace(observation, left_outer_pressed=True)
    assert router.choose_key(observation) == "right_outer"

    observation = replace(observation, right_outer_pressed=True)
    assert router.choose_key(observation) is None


def test_center_first_teacher_action_releases_held_keys_and_prefers_middle() -> None:
    router = CenterFirstFourKeyRouter()
    observation = replace(_idle_observation(), left_inner_pressed=True)

    action = router.teacher_action(observation, press_now=True)

    assert action.left_inner == -1.0
    assert action.right_inner == 1.0
    assert action.left_outer == 0.0
    assert action.right_outer == 0.0


def test_four_key_motor_clamps_actions_and_emits_named_physical_events() -> None:
    env = FourKeyMotorEnv(control_dt_s=0.010)
    observation = env.reset()
    seen_down = None

    for _ in range(300):
        transition = env.step(FourKeyAction(0.0, 100.0, 0.0, -100.0))
        observation = transition.observation
        for event in transition.evaluator_events:
            if event.event is KeyEvent.DOWN:
                seen_down = event
                break
        if seen_down is not None:
            break

    assert seen_down is not None
    assert seen_down.key == "left_inner"
    assert observation.left_inner_pressed
    diagnostics = env.diagnostics()
    assert -1.0 <= diagnostics.left_inner_activation <= 1.0
    assert -1.0 <= diagnostics.right_outer_activation <= 1.0


def test_router_constants_match_physical_left_to_right_layout() -> None:
    assert FOUR_KEY_NAMES == (
        "left_outer",
        "left_inner",
        "right_inner",
        "right_outer",
    )
    assert CENTER_KEY_NAMES == ("left_inner", "right_inner")
