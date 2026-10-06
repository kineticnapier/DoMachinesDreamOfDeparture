from __future__ import annotations

"""Open-loop lead calibration for the four-key motor body.

The legacy two-key teacher measures how long one full press takes from rest and
adds half a control frame to compensate for policy-decision quantization.  The
four-key path needs the same measurement on its own body instead of inheriting
the old ~43 ms constant.

Center-first play normally launches one of the two inner keys, so the shared
teacher lead is based on the slower measured inner key.  Outer-key latencies are
still recorded for diagnostics; overflow timing can be specialized later if the
measured high-density experiments show that it is necessary.
"""

from dataclasses import dataclass

from dmdod.legacy.four_key.motor import FOUR_KEY_NAMES, FourKeyAction, FourKeyMotorEnv
from dmdod.motor.keyboard import KeyEvent


@dataclass(frozen=True, slots=True)
class FourKeyLeadCalibration:
    left_outer_press_latency_s: float
    left_inner_press_latency_s: float
    right_inner_press_latency_s: float
    right_outer_press_latency_s: float
    lead_s: float
    control_dt_s: float
    physics_dt_s: float

    @property
    def center_press_latency_s(self) -> float:
        """Conservative single-press latency for the center-first key tier."""

        return max(
            self.left_inner_press_latency_s,
            self.right_inner_press_latency_s,
        )

    def latency_for(self, key: str) -> float:
        if key not in FOUR_KEY_NAMES:
            raise KeyError(key)
        return float(getattr(self, f"{key}_press_latency_s"))


def _single_key_action(key: str) -> FourKeyAction:
    if key not in FOUR_KEY_NAMES:
        raise KeyError(key)
    commands = {name: 0.0 for name in FOUR_KEY_NAMES}
    commands[key] = 1.0
    return FourKeyAction(*(commands[name] for name in FOUR_KEY_NAMES))


def _measure_single_press_latency(
    key: str,
    *,
    control_dt_s: float,
    physics_dt_s: float,
    max_wait_s: float,
) -> float:
    env = FourKeyMotorEnv(
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
    )
    env.reset()
    action = _single_key_action(key)
    max_steps = max(1, int(max_wait_s / control_dt_s) + 1)

    for _ in range(max_steps):
        transition = env.step(action)
        for event in transition.evaluator_events:
            if event.event is KeyEvent.DOWN and event.key == key:
                return float(event.time_s)

    raise RuntimeError(f"four-key single-press calibration produced no {key} key-down event")


def calibrate_four_key_press_lead(
    *,
    control_dt_s: float = 0.010,
    physics_dt_s: float = 0.001,
    max_wait_s: float = 1.0,
) -> FourKeyLeadCalibration:
    """Measure the four-key body's single-press latency from rest.

    Each key is measured in a fresh environment so hand/fatigue state from one
    measurement cannot leak into the next.  The center-first teacher uses one
    global launch lead, chosen conservatively from the slower inner key plus
    half a control step, matching the quantization compensation used by the
    mature two-key calibration.
    """

    if max_wait_s <= 0.0:
        raise ValueError("max_wait_s must be positive")

    latencies = {
        key: _measure_single_press_latency(
            key,
            control_dt_s=control_dt_s,
            physics_dt_s=physics_dt_s,
            max_wait_s=max_wait_s,
        )
        for key in FOUR_KEY_NAMES
    }
    center_latency = max(latencies["left_inner"], latencies["right_inner"])
    return FourKeyLeadCalibration(
        left_outer_press_latency_s=latencies["left_outer"],
        left_inner_press_latency_s=latencies["left_inner"],
        right_inner_press_latency_s=latencies["right_inner"],
        right_outer_press_latency_s=latencies["right_outer"],
        lead_s=center_latency + 0.5 * float(control_dt_s),
        control_dt_s=float(control_dt_s),
        physics_dt_s=float(physics_dt_s),
    )
