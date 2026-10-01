from __future__ import annotations

import pytest

from dmdod.motor_env import MotorObservation
from dmdod.visual_observation import (
    PROPRIOCEPTION_DIM,
    RgbFrame,
    VisualFrameScheduler,
    VisualPolicyObservation,
    encode_proprioception,
)


def _motor() -> MotorObservation:
    return MotorObservation(
        left_position_m=0.003,
        right_position_m=0.0015,
        left_velocity_m_s=0.25,
        right_velocity_m_s=-0.5,
        left_pressed=True,
        right_pressed=False,
    )


def test_rgb_frame_requires_tightly_packed_rgb888() -> None:
    frame = RgbFrame(width=2, height=1, data=bytes([1, 2, 3, 4, 5, 6]))
    assert frame.width == 2
    assert frame.height == 1

    with pytest.raises(ValueError, match="6 bytes"):
        RgbFrame(width=2, height=1, data=b"short")


def test_visual_policy_observation_contains_only_frame_and_motor() -> None:
    frame = RgbFrame(width=1, height=1, data=b"\x00\x00\x00")
    observation = VisualPolicyObservation(frame=frame, motor=_motor())

    assert observation.frame is frame
    assert observation.motor == _motor()
    assert tuple(observation.__dataclass_fields__) == ("frame", "motor")


def test_proprioception_matches_existing_body_normalization() -> None:
    encoded = encode_proprioception(_motor())

    assert len(encoded) == PROPRIOCEPTION_DIM
    assert encoded == pytest.approx((0.5, 0.25, 0.25, -0.5, 1.0, 0.0))


def test_default_60hz_visual_scheduler_holds_frames_at_100hz_policy() -> None:
    scheduler = VisualFrameScheduler()

    indices = [scheduler.frame_index(tick) for tick in range(10)]
    assert indices == [0, 0, 1, 1, 2, 3, 3, 4, 4, 5]

    scheduler.reset()
    changes = [scheduler.needs_new_frame(tick) for tick in range(10)]
    assert changes == [True, False, True, False, True, True, False, True, False, True]


def test_scheduler_uses_exact_integer_tick_mapping() -> None:
    scheduler = VisualFrameScheduler(visual_hz=60, policy_hz=100)

    assert scheduler.frame_index(10_000_000) == 6_000_000
    assert scheduler.frame_time_s(5) == pytest.approx(3 / 60)


def test_scheduler_rejects_invalid_rates_and_ticks() -> None:
    with pytest.raises(ValueError):
        VisualFrameScheduler(visual_hz=0, policy_hz=100)
    with pytest.raises(ValueError):
        VisualFrameScheduler(visual_hz=120, policy_hz=100)

    scheduler = VisualFrameScheduler()
    with pytest.raises(ValueError):
        scheduler.frame_index(-1)
