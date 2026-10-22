from __future__ import annotations

"""Research-only simulator latency envelope from an exact Motor snapshot."""

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import math

from dmdod.motor.keyboard import KeyEvent
from dmdod.motor.n_key import NKeyAction, NKeyMotorEnv


class EnvelopeStatus(str, Enum):
    VALID = "valid"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class MotorLatencyEnvelope:
    axis: str
    status: EnvelopeStatus
    observed_at_s: float
    earliest_arrival_s: float | None
    nominal_arrival_s: float | None
    latest_arrival_s: float | None
    uncertainty_s: float | None
    validity_conditions: tuple[str, ...]
    reason: str

    def contains(self, arrival_s: float) -> bool:
        return (
            self.status is EnvelopeStatus.VALID
            and self.earliest_arrival_s is not None
            and self.latest_arrival_s is not None
            and self.earliest_arrival_s - 1e-12 <= arrival_s <= self.latest_arrival_s + 1e-12
        )


def _action(env: NKeyMotorEnv, axis: str, value: float) -> NKeyAction:
    return NKeyAction(tuple(value if key == axis else 0.0 for key in env.key_names))


def _arrival(cloned: NKeyMotorEnv, axis: str, *, delay_steps: int, max_wait_s: float) -> float | None:
    start = cloned.diagnostics().time_s
    for _ in range(delay_steps):
        cloned.step(_action(cloned, axis, 0.0))
    limit = max(1, math.ceil(max_wait_s / cloned.control_dt_s))
    for _ in range(limit):
        transition = cloned.step(_action(cloned, axis, 1.0))
        for event in transition.evaluator_events:
            if event.key == axis and event.event is KeyEvent.DOWN:
                return float(event.time_s - start)
    return None


def predict_latency_envelope(
    env: NKeyMotorEnv,
    axis: str,
    *,
    exact_full_snapshot: bool,
    reservation_conflict_free: bool | None,
    max_wait_s: float = 0.25,
) -> MotorLatencyEnvelope:
    """Bound arrival under a declared zero-to-one-control-tick issue delay.

    The bound is valid only while all non-selected axes receive zero commands,
    the selected command remains +1, and the copied simulator state is exact.
    It is not a learned hardware model and cannot consume chart truth.
    """

    if axis not in env.key_names:
        raise KeyError(axis)
    now = float(env.diagnostics().time_s)
    conditions = (
        "exact full NKeyMotorEnv state",
        "selected axis held at +1 after issue",
        "all other axes held at 0",
        "issue delay lies in [0, one control tick]",
        "no contact-topology switch",
        "no competing reservation or command",
        "frozen simulator physics",
    )
    if not exact_full_snapshot:
        return MotorLatencyEnvelope(axis, EnvelopeStatus.UNKNOWN, now, None, None, None, None, conditions, "full state is not observed")
    if reservation_conflict_free is not True:
        return MotorLatencyEnvelope(axis, EnvelopeStatus.UNKNOWN, now, None, None, None, None, conditions, "reservation/command conflict is not proven absent")
    if env.observe().pressed(axis):
        return MotorLatencyEnvelope(axis, EnvelopeStatus.UNKNOWN, now, None, None, None, None, conditions, "axis is already pressed; next KeyDown requires an unmodelled release cycle")

    immediate = _arrival(deepcopy(env), axis, delay_steps=0, max_wait_s=max_wait_s)
    delayed = _arrival(deepcopy(env), axis, delay_steps=1, max_wait_s=max_wait_s)
    if immediate is None or delayed is None:
        return MotorLatencyEnvelope(axis, EnvelopeStatus.UNKNOWN, now, None, None, None, None, conditions, "no KeyDown in bounded rollout")
    earliest = now + min(immediate, delayed)
    latest = now + max(immediate, delayed)
    nominal = 0.5 * (earliest + latest)
    return MotorLatencyEnvelope(
        axis, EnvelopeStatus.VALID, now, earliest, nominal, latest,
        latest - earliest, conditions,
        "deterministic frozen-Motor rollouts bracket zero/one-control-tick issue delay",
    )


def execute_immediate_press(env: NKeyMotorEnv, axis: str, *, max_wait_s: float = 0.25) -> float:
    """Independent validation path that advances the supplied environment."""

    arrival = _arrival(env, axis, delay_steps=0, max_wait_s=max_wait_s)
    if arrival is None:
        raise RuntimeError("validation press produced no KeyDown")
    return arrival
