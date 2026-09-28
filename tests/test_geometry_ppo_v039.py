from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_geometry_ppo_v039 as trainer  # noqa: E402
from dmdod.privileged_teacher import (  # noqa: E402
    PrivilegedLeadTeacher,
    calibrate_single_press_lead,
)


def test_privileged_teacher_calibrates_body_and_hits_p3_edges_near_center():
    control_dt = 0.010
    calibration = calibrate_single_press_lead(control_dt_s=control_dt)
    assert 0.05 < calibration.press_latency_s < 0.50
    assert math.isclose(
        calibration.lead_s,
        calibration.press_latency_s + control_dt / 2.0,
        abs_tol=1e-12,
    )

    teacher = PrivilegedLeadTeacher(calibration)
    for bpm in (trainer.IMITATION_BPM_MIN, trainer.IMITATION_BPM_MAX):
        env, target = trainer._make_teacher_env(
            bpm=bpm,
            start_s=0.450,
            control_dt=control_dt,
            seed=1,
        )
        observation = env.reset()
        while True:
            action = teacher.action(
                now_s=env.motor.diagnostics().time_s,
                target_time_s=target.time_s,
                motor=observation.motor,
            )
            transition = env.step(action)
            observation = transition.observation
            if transition.done:
                break

        assert env.stats.hits == 1
        assert len(env.timing_errors_ms) == 1
        assert abs(env.timing_errors_ms[0]) <= control_dt * 1000.0 + 1.0


def test_demonstrations_expose_only_14d_student_observation_and_teacher_actions():
    calibration = calibrate_single_press_lead(control_dt_s=0.010)
    demos = trainer.collect_demonstrations(
        PrivilegedLeadTeacher(calibration),
        episodes=4,
        control_dt=0.010,
        device=torch.device("cpu"),
        seed=3,
    )

    assert len(demos) == 4
    assert all(demo.observations.ndim == 2 for demo in demos)
    assert all(demo.observations.shape[1] == trainer.MOTION_GEOMETRY_INPUT_DIM for demo in demos)
    assert all(demo.actions.shape == (demo.observations.shape[0], 2) for demo in demos)
    assert any(torch.count_nonzero(demo.actions[:, 0] > 0.5).item() > 0 for demo in demos)


def test_small_imitation_pass_is_finite():
    calibration = calibrate_single_press_lead(control_dt_s=0.010)
    demos = trainer.collect_demonstrations(
        PrivilegedLeadTeacher(calibration),
        episodes=4,
        control_dt=0.010,
        device=torch.device("cpu"),
        seed=5,
    )
    model = trainer.PredictiveRecurrentActorCritic(
        input_dim=trainer.MOTION_GEOMETRY_INPUT_DIM,
        hidden_dim=16,
    )
    loss = trainer.imitation_pretrain(model, demos, epochs=1, lr=1e-3)
    assert math.isfinite(loss)
    assert loss >= 0.0
