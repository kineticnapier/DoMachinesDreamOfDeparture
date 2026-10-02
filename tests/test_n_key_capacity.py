from __future__ import annotations

import pytest

from dmdod.four_key_motor import FourKeyAction, FourKeyMotorEnv
from dmdod.n_key_capacity import CenterFirstNKeyTeacher, calibrate_n_key_press_lead
from dmdod.n_key_motor import NKeyAction, NKeyMotorEnv, n_key_names, n_key_tiers


def test_n_key_layout_is_center_symmetric() -> None:
    assert n_key_names(4) == ("left_2", "left_1", "right_1", "right_2")
    assert n_key_names(6) == (
        "left_3",
        "left_2",
        "left_1",
        "right_1",
        "right_2",
        "right_3",
    )
    assert n_key_tiers(8) == (
        ("left_1", "right_1"),
        ("left_2", "right_2"),
        ("left_3", "right_3"),
        ("left_4", "right_4"),
    )


@pytest.mark.parametrize("bad", [0, 1, 3, 5])
def test_n_key_layout_rejects_non_even_counts(bad: int) -> None:
    with pytest.raises(ValueError):
        n_key_names(bad)


def test_generic_4k_motor_matches_existing_four_key_body() -> None:
    old = FourKeyMotorEnv(control_dt_s=0.010, physics_dt_s=0.001)
    new = NKeyMotorEnv(4, control_dt_s=0.010, physics_dt_s=0.001)
    old.reset()
    new.reset()

    patterns = [
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0, 1.0),
        (-1.0, -1.0, 1.0, 1.0),
        (-1.0, -1.0, -1.0, -1.0),
        (0.0, 0.0, 0.0, 0.0),
    ]

    old_to_new = {
        "left_outer": "left_2",
        "left_inner": "left_1",
        "right_inner": "right_1",
        "right_outer": "right_2",
    }

    for step_index in range(120):
        values = patterns[(step_index // 20) % len(patterns)]
        old_step = old.step(FourKeyAction(*values))
        new_step = new.step(NKeyAction(values))

        old_obs = old_step.observation
        new_obs = new_step.observation
        assert new_obs.positions_m == pytest.approx(
            (
                old_obs.left_outer_position_m,
                old_obs.left_inner_position_m,
                old_obs.right_inner_position_m,
                old_obs.right_outer_position_m,
            ),
            abs=1e-12,
        )
        assert new_obs.velocities_m_s == pytest.approx(
            (
                old_obs.left_outer_velocity_m_s,
                old_obs.left_inner_velocity_m_s,
                old_obs.right_inner_velocity_m_s,
                old_obs.right_outer_velocity_m_s,
            ),
            abs=1e-12,
        )
        assert new_obs.pressed_flags == (
            old_obs.left_outer_pressed,
            old_obs.left_inner_pressed,
            old_obs.right_inner_pressed,
            old_obs.right_outer_pressed,
        )
        assert [
            (old_to_new[event.key], event.event, event.time_s)
            for event in old_step.evaluator_events
        ] == [
            (event.key, event.event, event.time_s)
            for event in new_step.evaluator_events
        ]


def test_center_first_n_key_teacher_reserves_from_inner_pair_outward() -> None:
    env = NKeyMotorEnv(8)
    observation = env.reset()
    teacher = CenterFirstNKeyTeacher(8)
    targets = tuple((index, 0.0) for index in range(6))

    action = teacher.pipeline_action(
        observation,
        now_s=0.0,
        targets=targets,
        lead_s=0.05,
    )
    commands = dict(zip(observation.key_names, action.values))

    assert commands["left_1"] == 1.0
    assert commands["right_1"] == 1.0
    assert commands["left_2"] == 1.0
    assert commands["right_2"] == 1.0
    assert commands["left_3"] == 1.0
    assert commands["right_3"] == 1.0
    assert commands["left_4"] == 0.0
    assert commands["right_4"] == 0.0


def test_n_key_calibration_matches_existing_4k_single_press_latency() -> None:
    calibration = calibrate_n_key_press_lead(4)
    assert calibration.lead_s == pytest.approx(0.044, abs=1e-12)
    assert dict(calibration.key_latencies_s) == pytest.approx(
        {
            "left_2": 0.039,
            "left_1": 0.039,
            "right_1": 0.039,
            "right_2": 0.039,
        },
        abs=1e-12,
    )
