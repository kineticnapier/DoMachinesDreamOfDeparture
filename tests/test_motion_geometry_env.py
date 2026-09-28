from __future__ import annotations

import pytest

from dmdod.evaluator import TargetHit
from dmdod.motion_geometry_env import MotionGeometryEnv
from dmdod.motor_env import MotorAction
from dmdod.planet_perception import PlanetVisionConfig


def test_motion_geometry_exposes_only_visible_frame_difference() -> None:
    env = MotionGeometryEnv(
        [TargetHit(1.0, "left")],
        bpm=180.0,
        control_dt_s=0.010,
        vision_config=PlanetVisionConfig(
            latency_s=0.0,
            latency_jitter_s=0.0,
            sample_period_s=0.0,
            position_noise_std=0.0,
            dropout_probability=0.0,
        ),
        perception_seed=1,
    )

    first = env.reset()
    assert first.motion.delta_orbit_x == 0.0
    assert first.motion.delta_orbit_y == 0.0
    assert first.motion.delta_next_x == 0.0
    assert first.motion.delta_next_y == 0.0

    transition = env.step(MotorAction(0.0, 0.0))
    second = transition.observation

    assert second.motion.delta_orbit_x == pytest.approx(
        second.geometry.orbit_x - first.geometry.orbit_x
    )
    assert second.motion.delta_orbit_y == pytest.approx(
        second.geometry.orbit_y - first.geometry.orbit_y
    )
    assert second.motion.delta_next_x == pytest.approx(
        second.geometry.next_x - first.geometry.next_x
    )
    assert second.motion.delta_next_y == pytest.approx(
        second.geometry.next_y - first.geometry.next_y
    )
    assert abs(second.motion.delta_orbit_x) + abs(second.motion.delta_orbit_y) > 0.0
