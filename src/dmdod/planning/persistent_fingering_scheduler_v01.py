from __future__ import annotations

"""Truth-free persistent physical-axis reservations for 12-axis Fingering.

This module does not run an ADOFAI evaluator or choose Motor commands. It
manages forecast-local reservations and real *physical* KeyDown/KeyUp events.
A consumed evaluator target MUST NOT release a physical axis.

Calling order per control frame:
  1. apply_forecast(TimingFrameUpdate, MotorPhysicalSnapshot)
  2. atomically reserve any chosen pending tokens, then arm as appropriate
  3. execute an external Motor command
  4. observe_physics(new snapshot, physical switch edges)

The future Timing v1.4 adapter and integration runner will own that loop.
No evaluator, chart ordinal, or Motor parameter is changed here.
"""

from copy import copy
from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite
from typing import Iterable

from dmdod.motor.contact_16k import Coupled16ContactMotor, PHYSICAL_AXES
from dmdod.motor.keyboard import KeyEvent
from dmdod.planning.timing_fingering_contract_v02 import (
    ABSTAIN, TimingEventToken, TimingFrameSequenceValidator, TimingFrameUpdate,
)

# Do not add unvalidated tolerances that could mask an in-flight stroke.
MAX_ARMED_RETIME_S = 0.0


class ReservationPhase(str, Enum):
    RESERVED = "reserved"
    ARMED_IN_FLIGHT = "armed_in_flight"
    KEYDOWN_OBSERVED = "keydown_observed"
    RELEASING = "releasing"
    CANCELED_IN_FLIGHT = "canceled_in_flight"


@dataclass(frozen=True, slots=True)
class AxisPhysics:
    axis: str
    position_m: float
    velocity_m_s: float
    activation: float
    pressed: bool

    def __post_init__(self) -> None:
        if self.axis not in PHYSICAL_AXES:
            raise ValueError("unknown physical axis")
        if any(not isfinite(x) for x in (
            self.position_m, self.velocity_m_s, self.activation
        )):
            raise ValueError("nonfinite axis state")
        if type(self.pressed) is not bool:
            raise ValueError("pressed must be boolean")


@dataclass(frozen=True, slots=True)
class MotorPhysicalSnapshot:
    """12-axis physical state, explicitly independent of judgement/targets."""

    axes: tuple[AxisPhysics, ...]
    reset_m: float

    def __post_init__(self) -> None:
        if tuple(x.axis for x in self.axes) != PHYSICAL_AXES:
            raise ValueError("physical axes must be complete and in motor order")
        if not isfinite(self.reset_m):
            raise ValueError("reset_m must be finite")

    @classmethod
    def from_motor(cls, motor: Coupled16ContactMotor) -> MotorPhysicalSnapshot:
        if not isinstance(motor, Coupled16ContactMotor):
            raise TypeError("expected Coupled16ContactMotor")
        observation = motor.motor.observe()
        diagnostics = motor.motor.diagnostics()
        if observation.key_names != PHYSICAL_AXES:
            raise ValueError("unexpected physical Motor axis order")
        return cls(
            axes=tuple(AxisPhysics(name, position, velocity, activation, bool(pressed))
                       for name, position, velocity, activation, pressed in zip(
                           observation.key_names, observation.positions_m,
                           observation.velocities_m_s, diagnostics.activations,
                           observation.pressed_flags, strict=True)),
            reset_m=motor.motor.keyboard.config.reset_m,
        )

    def axis(self, name: str) -> AxisPhysics:
        try:
            return self.axes[PHYSICAL_AXES.index(name)]
        except ValueError as exc:
            raise KeyError(name) from exc

    def ready(self, name: str) -> bool:
        """Conservative pre-reservation check, with no scorer information.

        Any pressed, above-reset, positively moving, or positively activated
        axis is unsafe for a new stroke. No epsilon allows "almost reset"
        motor axes to be reassigned.
        """
        x = self.axis(name)
        return (
            not x.pressed
            and x.position_m <= self.reset_m
            and x.velocity_m_s <= 0.0
            and x.activation <= 0.0
        )


@dataclass(frozen=True, slots=True)
class PhysicalFinger:
    """One physical actuator and the number of linked logical contacts."""

    axis: str
    contact_count: int = 1

    def __post_init__(self) -> None:
        if self.axis not in PHYSICAL_AXES:
            raise ValueError("unknown physical axis")
        depth = int(self.axis.rsplit("_", 1)[1])
        allowed = (1, 2) if depth in (5, 6) else (1,)
        if type(self.contact_count) is not int or self.contact_count not in allowed:
            raise ValueError("contact count incompatible with axis topology")


@dataclass(frozen=True, slots=True)
class PhysicalReservation:
    local_id: int
    due_local_s: float
    fingers: tuple[PhysicalFinger, ...]
    phase: ReservationPhase
    confirmed: bool
    down_axes: frozenset[str] = frozenset()
    up_axes: frozenset[str] = frozenset()
    retired_reason: str | None = None

    @property
    def axes(self) -> tuple[str, ...]:
        return tuple(f.axis for f in self.fingers)


class PersistentFingeringScheduler:
    """Physical reservations keyed only by forecast-local event identity.

    - Atomic chord allocation.
    - No reallocation while an axis is in flight, pressed, or not reset.
    - Canceled armed strokes remain quarantined until physical reset.
    - Abstention carries existing reservations forward as unconfirmed.
    - No observer may use evaluator hit/miss state to retire a reservation.
    """

    def __init__(self) -> None:
        self._validator = TimingFrameSequenceValidator()
        self._pending: dict[int, TimingEventToken] = {}
        self._reservations: dict[int, PhysicalReservation] = {}

    @property
    def reservations(self) -> tuple[PhysicalReservation, ...]:
        return tuple(self._reservations[x] for x in sorted(self._reservations))

    @property
    def pending_local_ids(self) -> frozenset[int]:
        return frozenset(self._pending)

    def reservation(self, local_id: int) -> PhysicalReservation:
        return self._reservations[local_id]

    def occupied_axes(self) -> frozenset[str]:
        return frozenset(axis for entry in self._reservations.values()
                         for axis in entry.axes)

    def available_axes(self, snapshot: MotorPhysicalSnapshot) -> tuple[str, ...]:
        busy = self.occupied_axes()
        return tuple(name for name in PHYSICAL_AXES
                     if name not in busy and snapshot.ready(name))

    def apply_forecast(self, frame: TimingFrameUpdate,
                       snapshot: MotorPhysicalSnapshot) -> None:
        """Apply one current/previous-only Timing update; no future truth."""
        # The externally supplied frame must already be a valid causal update.
        candidate_validator = copy(self._validator)
        candidate_validator.accept(frame)
        new_pending = {item.local_id: item for item in frame.pending}
        retired = {
            **{x: "canceled" for x in frame.canceled_ids},
            **{x: "elapsed" for x in frame.elapsed_ids},
        }
        updated = dict(self._reservations)
        for local_id, entry in self._reservations.items():
            if local_id in retired:
                if entry.phase is ReservationPhase.RESERVED:
                    # No command has been issued; this does not release an
                    # existing stroke, because every issued stroke is ARMED.
                    updated.pop(local_id)
                else:
                    updated[local_id] = replace(
                        entry, phase=ReservationPhase.CANCELED_IN_FLIGHT,
                        confirmed=False, retired_reason=retired[local_id],
                    )
                continue
            if local_id in new_pending:
                updated_token = new_pending[local_id]
                if updated_token.chord_size != sum(
                    finger.contact_count for finger in entry.fingers
                ):
                    raise ValueError("cannot change chord size of an existing reservation")
                if (entry.phase is not ReservationPhase.RESERVED
                        and abs(updated_token.due_local_s - entry.due_local_s)
                        > MAX_ARMED_RETIME_S):
                    raise ValueError("cannot silently retime an armed physical stroke")
                updated[local_id] = replace(
                    entry,
                    due_local_s=(updated_token.due_local_s
                                 if entry.phase is ReservationPhase.RESERVED
                                 else entry.due_local_s),
                    confirmed=(frame.status != ABSTAIN
                               and updated_token.confirmed_now),
                )
        self._validator = candidate_validator
        self._pending = new_pending
        self._reservations = updated
        # A retained retired stroke can become free only after the Motor
        # itself has physically reset; no forecast expiry releases it.
        self._retire_recovered(snapshot)

    def reserve(self, local_id: int, fingers: tuple[PhysicalFinger, ...],
                snapshot: MotorPhysicalSnapshot) -> PhysicalReservation:
        """Atomic physical assignment: either all axes are reserved or none."""
        if local_id not in self._pending:
            raise ValueError("cannot allocate a non-pending forecast local ID")
        if local_id in self._reservations:
            raise ValueError("local forecast already reserved")
        token = self._pending[local_id]
        if not token.confirmed_now:
            raise ValueError("cannot create a new reservation from unconfirmed forecast")
        if type(fingers) is not tuple or not fingers:
            raise ValueError("a reservation needs at least one physical finger")
        if any(not isinstance(finger, PhysicalFinger) for finger in fingers):
            raise TypeError("fingers must be PhysicalFinger objects")
        if len({finger.axis for finger in fingers}) != len(fingers):
            raise ValueError("duplicate physical axis in chord")
        if sum(f.contact_count for f in fingers) != token.chord_size:
            raise ValueError("allocation does not match forecast chord size")
        available = set(self.available_axes(snapshot))
        if any(f.axis not in available for f in fingers):
            raise ValueError("a chosen physical axis is reserved or not reset")
        entry = PhysicalReservation(
            local_id=local_id, due_local_s=token.due_local_s,
            fingers=fingers, phase=ReservationPhase.RESERVED, confirmed=True,
        )
        self._reservations[local_id] = entry
        return entry

    def arm(self, local_id: int, snapshot: MotorPhysicalSnapshot) -> None:
        entry = self._reservations[local_id]
        if entry.phase is not ReservationPhase.RESERVED:
            raise ValueError("only a reserved physical intent can be armed")
        if not entry.confirmed or local_id not in self._pending:
            raise ValueError("cannot arm an unconfirmed/retired forecast")
        if any(not snapshot.ready(axis) for axis in entry.axes):
            raise ValueError("reserved physical axis is no longer ready")
        self._reservations[local_id] = replace(
            entry, phase=ReservationPhase.ARMED_IN_FLIGHT
        )

    def observe_physics(
        self, snapshot: MotorPhysicalSnapshot,
        physical_edges: Iterable[tuple[str, KeyEvent]] = (),
    ) -> None:
        """Consume ONLY physical switch events from the 12-axis Motor.

        Use NKeyTransition.evaluator_events from the *physical Motor layer*,
        not ADOFAI's target judgement trace or 16-contact Hit/Miss results.
        A KeyDown closes the strike intent but never proves an evaluator Hit.
        """
        updated = dict(self._reservations)
        owner = {axis: local_id for local_id, entry in updated.items()
                 for axis in entry.axes}
        for axis, edge in physical_edges:
            if axis not in PHYSICAL_AXES:
                raise ValueError("unknown physical switch axis")
            if edge not in (KeyEvent.DOWN, KeyEvent.UP):
                raise ValueError("expected physical KeyDown/KeyUp")
            local_id = owner.get(axis)
            if local_id is None:
                continue
            entry = updated[local_id]
            if edge is KeyEvent.DOWN:
                down = entry.down_axes | frozenset((axis,))
                phase = entry.phase
                if phase in (ReservationPhase.ARMED_IN_FLIGHT,
                             ReservationPhase.RESERVED) and set(entry.axes) <= down:
                    phase = (ReservationPhase.RELEASING if entry.up_axes
                             else ReservationPhase.KEYDOWN_OBSERVED)
                elif phase is ReservationPhase.RESERVED:
                    # An unexpected physical stroke cannot be treated as an
                    # unarmed, instantly cancellable reservation.
                    phase = ReservationPhase.CANCELED_IN_FLIGHT
                updated[local_id] = replace(entry, down_axes=down, phase=phase)
            else:
                up = entry.up_axes | frozenset((axis,))
                phase = entry.phase
                if phase is ReservationPhase.KEYDOWN_OBSERVED:
                    phase = ReservationPhase.RELEASING
                updated[local_id] = replace(entry, up_axes=up, phase=phase)
        self._reservations = updated
        self._retire_recovered(snapshot)

    def _retire_recovered(self, snapshot: MotorPhysicalSnapshot) -> None:
        for local_id, entry in tuple(self._reservations.items()):
            if entry.phase in (ReservationPhase.RELEASING,
                               ReservationPhase.CANCELED_IN_FLIGHT):
                if all(snapshot.ready(axis) for axis in entry.axes):
                    self._reservations.pop(local_id)

    def cancel_unarmed(self, local_id: int) -> None:
        """Explicit external cancellation, safe only before command issuance."""
        entry = self._reservations[local_id]
        if entry.phase is not ReservationPhase.RESERVED:
            raise ValueError("armed reservation requires physical cancellation")
        self._reservations.pop(local_id)
