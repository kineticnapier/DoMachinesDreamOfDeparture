from __future__ import annotations

"""Prestige I: causal Timing / Fingering / Motor *interface only*.

This module supplies immutable data contracts, not trained models, a Teacher,
an IRL reward function, or a policy rollout.  Only 263D human-visible HUD
frames enter the proposed Student pipeline.  The chart's true event schedule,
target ordinal, judge windows, environment clock, and teacher action do not
appear in these interfaces.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from dmdod.motor.n_key import NKeyAction, n_key_names

HUD_DIM = 263
KEY_COUNT = 8
CONTROL_DT_S = 0.010
# Current Stage A horizon; the schema explicitly allows a later signed,
# post-event estimator via a versioned contract, not an implicit API change.
FORECAST_HORIZON_S = 0.200
FINGER_ORDER = n_key_names(KEY_COUNT)


@dataclass(frozen=True, slots=True)
class VisibleHUDFrame:
    """Exactly one observable 10ms frame; step is a causal index, not chart time."""

    step: int
    features: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0:
            raise ValueError("HUD step must be a nonnegative integer")
        if len(self.features) != HUD_DIM or any(
            not isfinite(float(value)) for value in self.features
        ):
            raise ValueError("HUD must contain exactly 263 finite values")

    @classmethod
    def from_observation(cls, step: int, observation) -> VisibleHUDFrame:
        """Encode the *current* real visible HUD, without accessing the evaluator.

        Avoid implementing adapters from environment internals or privileged
        target contracts. Only the public human-visible encoder is used.
        """
        from dmdod.envs.n_key import (
            NKeyHudRealChartObservation,
            encode_n_key_hud_real_chart_observation,
        )

        if not isinstance(observation, NKeyHudRealChartObservation):
            raise TypeError("Expected current NKeyHudRealChartObservation")
        features = encode_n_key_hud_real_chart_observation(observation)
        return cls(step=step, features=tuple(features))

    @property
    def pressed(self) -> tuple[bool, ...]:
        """Public 8-key pressed flags, positions in the 263D HUD are unchanged."""
        return tuple(bool(x >= 0.5) for x in self.features[2 * KEY_COUNT : 3 * KEY_COUNT])

    @property
    def normalized_positions(self) -> tuple[float, ...]:
        return self.features[:KEY_COUNT]

    @property
    def normalized_velocities(self) -> tuple[float, ...]:
        return self.features[KEY_COUNT : 2 * KEY_COUNT]


@dataclass(frozen=True, slots=True)
class CausalHUDHistory:
    """Consecutive observable frames only; no future or privileged channels."""

    frames: tuple[VisibleHUDFrame, ...]

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("Causal observation history cannot be empty")
        for previous, current in zip(self.frames, self.frames[1:]):
            if current.step != previous.step + 1:
                raise ValueError("Causal history must contain consecutive frames")

    @property
    def current(self) -> VisibleHUDFrame:
        return self.frames[-1]

    def append(self, frame: VisibleHUDFrame, *, max_frames: int) -> CausalHUDHistory:
        if type(max_frames) is not int or max_frames < 1:
            raise ValueError("max_frames must be a positive integer")
        if frame.step != self.current.step + 1:
            raise ValueError("Cannot append past, duplicate, or future HUD frame")
        return CausalHUDHistory((self.frames + (frame,))[-max_frames:])


@dataclass(frozen=True, slots=True)
class PredictedEvent:
    """Timing-only estimate. No target ordinal, truth, or finger ID."""

    relative_time_s: float
    chord_size: int
    confidence: float

    def __post_init__(self) -> None:
        if (
            not isfinite(self.relative_time_s)
            or not 0.0 <= self.relative_time_s <= FORECAST_HORIZON_S
        ):
            raise ValueError("Timing estimate outside causal 0..200ms horizon")
        if type(self.chord_size) is not int or not 1 <= self.chord_size <= KEY_COUNT:
            raise ValueError("Predicted chord size must be 1..8")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Predicted event confidence must be finite and 0..1")


@dataclass(frozen=True, slots=True)
class TimingPrediction:
    """Zero or more near events ordered by predicted relative time."""

    events: tuple[PredictedEvent, ...]

    def __post_init__(self) -> None:
        for previous, current in zip(self.events, self.events[1:]):
            if current.relative_time_s < previous.relative_time_s:
                raise ValueError("Timing events must be sorted by relative time")
        # Equal predicted times should be one chord, not duplicate events.
        if len({event.relative_time_s for event in self.events}) != len(self.events):
            raise ValueError("Merge simultaneous predicted events into one chord")


@dataclass(frozen=True, slots=True)
class EventFingering:
    """An *optional* allocation for one predicted event (not a target ID).

    The event_index refers to the TimingPrediction produced in the same
    causal step. A chord uses distinct physical keys. A missing allocation
    means the planner abstained; it is not silently filled with center-first.
    """

    event_index: int
    key_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.event_index) is not int or self.event_index < 0:
            raise ValueError("Fingering index must be nonnegative")
        if any(type(k) is not int or not 0 <= k < KEY_COUNT for k in self.key_indices):
            raise ValueError("Fingering key indices must be 0..7")
        if len(set(self.key_indices)) != len(self.key_indices):
            raise ValueError("One physical key cannot press two notes of a chord")
        if tuple(sorted(self.key_indices)) != self.key_indices:
            raise ValueError("Chord finger indices must follow physical key order")


@dataclass(frozen=True, slots=True)
class FingeringProposal:
    """Separate planner output. Can abstain from any predicted event."""

    allocations: tuple[EventFingering, ...]

    def __post_init__(self) -> None:
        indices = [assignment.event_index for assignment in self.allocations]
        if indices != sorted(set(indices)):
            raise ValueError("Fingering event indices must be sorted and unique")


@dataclass(frozen=True, slots=True)
class TimedFingerIntent:
    """Joint high-level intent, *not* a physical 8-key command."""

    relative_time_s: float
    key_indices: tuple[int, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class JointIntent:
    """Couples predicted timing with chosen fingers, without issuing actions."""

    events: tuple[TimedFingerIntent, ...]
    abstained_event_indices: tuple[int, ...]


def join_predictions(timing: TimingPrediction, fingering: FingeringProposal) -> JointIntent:
    """Validate cross-head consistency; never fabricate missing finger choices."""
    by_event: dict[int, EventFingering] = {}
    for selection in fingering.allocations:
        if selection.event_index >= len(timing.events):
            raise ValueError("Fingering references an absent predicted event")
        target = timing.events[selection.event_index]
        if len(selection.key_indices) != target.chord_size:
            raise ValueError("Finger allocation does not match predicted chord size")
        by_event[selection.event_index] = selection
    events: list[TimedFingerIntent] = []
    abstained: list[int] = []
    for i, target in enumerate(timing.events):
        if i not in by_event:
            abstained.append(i)
        else:
            events.append(TimedFingerIntent(
                relative_time_s=target.relative_time_s,
                key_indices=by_event[i].key_indices,
                confidence=target.confidence,
            ))
    return JointIntent(tuple(events), tuple(abstained))


class TimingEstimator(Protocol):
    """May access *only* causal human-visible observation history."""

    def predict_timing(self, history: CausalHUDHistory) -> TimingPrediction: ...


class FingeringPlanner(Protocol):
    """May access only causal history and current Timing predictions.

    Future hidden target schedules, true per-note timestamps, teacher finger
    assignment, and evaluator-only motor diagnostics are intentionally absent.
    IRL/imitation may be plugged into *the implementation*, not the API.
    """

    def propose_fingering(
        self, history: CausalHUDHistory, timing: TimingPrediction
    ) -> FingeringProposal: ...


class MotorController(Protocol):
    """Converts nonprivileged high-level intent to continuous [-1, 1]^8 action."""

    def command(self, history: CausalHUDHistory, intent: JointIntent) -> NKeyAction: ...


def validate_motor_action(action: NKeyAction) -> NKeyAction:
    """Enforce motor shape/range; chart physics remains authoritative."""
    values = action.as_tuple()
    if len(values) != KEY_COUNT or any(
        not isfinite(value) or not -1.0 <= value <= 1.0 for value in values
    ):
        raise ValueError("Motor action must be eight finite forces in [-1,1]")
    return action
