from __future__ import annotations

"""Level B visual-observation boundary for DMDOD.

This module deliberately contains no ADOFAI chart metadata.  A renderer may use
privileged chart state internally to produce pixels, but the policy-side
observation crosses the boundary as RGB pixels plus the six values of
``MotorObservation`` only.

The first implementation is renderer-agnostic so ExtremeEditor (or any other
backend) can be connected later without changing the policy-visible contract.
"""

from dataclasses import dataclass
from typing import Protocol

from .motor_env import MotorObservation


VISUAL_OBSERVATION_VERSION = "level-b-rgb-proprio-v1"
DEFAULT_VISUAL_WIDTH = 320
DEFAULT_VISUAL_HEIGHT = 180
DEFAULT_VISUAL_HZ = 60
DEFAULT_POLICY_HZ = 100
PROPRIOCEPTION_DIM = 6


@dataclass(frozen=True, slots=True)
class RgbFrame:
    """One tightly packed RGB888 frame crossing the renderer boundary."""

    width: int
    height: int
    data: bytes

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("RGB frame dimensions must be positive")
        expected = int(self.width) * int(self.height) * 3
        if len(self.data) != expected:
            raise ValueError(
                f"RGB888 frame must contain {expected} bytes, got {len(self.data)}"
            )


@dataclass(frozen=True, slots=True)
class VisualPolicyObservation:
    """Complete Level B policy-visible observation.

    ``frame`` is the only chart-derived channel.  ``motor`` is proprioception
    owned by DMDOD's body simulator.  Chart time, floor metadata, BPM, target
    timing, judgement metadata, evaluator state, and renderer scene objects are
    intentionally impossible to attach through this type.
    """

    frame: RgbFrame
    motor: MotorObservation


class ChartFrameRenderer(Protocol):
    """Minimal renderer contract expected by DMDOD.

    ``time_s`` is supplied by the environment to the renderer, not exposed to
    the policy.  Implementations may be in-process, IPC-backed, or pre-rendered.
    """

    def render(self, time_s: float, *, width: int, height: int) -> RgbFrame: ...


def encode_proprioception(motor: MotorObservation) -> tuple[float, ...]:
    """Encode the six body-visible values independently of visual pixels.

    The scale matches the existing structured encoder so a later A/B comparison
    does not accidentally change body-state normalization at the same time as
    changing visual perception.
    """

    values = (
        float(motor.left_position_m) / 0.006,
        float(motor.right_position_m) / 0.006,
        float(motor.left_velocity_m_s) / 1.0,
        float(motor.right_velocity_m_s) / 1.0,
        1.0 if motor.left_pressed else 0.0,
        1.0 if motor.right_pressed else 0.0,
    )
    if len(values) != PROPRIOCEPTION_DIM:
        raise RuntimeError("internal proprioception size mismatch")
    return values


class VisualFrameScheduler:
    """Map policy ticks to a lower-rate visual stream without float drift.

    At policy tick ``n`` the policy sees the newest visual frame whose nominal
    sample time is not later than that tick.  For the default 100 Hz policy and
    60 Hz visual stream the frame index is ``floor(n * 60 / 100)``.

    This class only determines when the frame changes.  It never exposes the
    resulting tick/frame index to the policy.
    """

    def __init__(
        self,
        *,
        visual_hz: int = DEFAULT_VISUAL_HZ,
        policy_hz: int = DEFAULT_POLICY_HZ,
    ) -> None:
        if visual_hz <= 0 or policy_hz <= 0:
            raise ValueError("visual_hz and policy_hz must be positive")
        if visual_hz > policy_hz:
            raise ValueError("visual_hz must not exceed policy_hz for frame holding")
        self.visual_hz = int(visual_hz)
        self.policy_hz = int(policy_hz)
        self._last_frame_index: int | None = None

    def reset(self) -> None:
        self._last_frame_index = None

    def frame_index(self, policy_tick: int) -> int:
        if policy_tick < 0:
            raise ValueError("policy_tick must be non-negative")
        return (int(policy_tick) * self.visual_hz) // self.policy_hz

    def needs_new_frame(self, policy_tick: int) -> bool:
        index = self.frame_index(policy_tick)
        if self._last_frame_index == index:
            return False
        self._last_frame_index = index
        return True

    def frame_time_s(self, policy_tick: int) -> float:
        """Nominal renderer time of the frame visible at this policy tick."""

        return self.frame_index(policy_tick) / float(self.visual_hz)
