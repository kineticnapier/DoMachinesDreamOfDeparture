from __future__ import annotations

"""Fail-closed Hold command overlay for the frozen 12-axis/16-contact Motor.

This module does not predict Hold timing, choose a finger, or judge success.
It receives an explicitly versioned interval/command plan, keeps the already
assigned physical axis depressed, and leaves all other Motor axes untouched.
Oracle diagnostic plans are opt-in and remain labelled as such.
"""

from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite

from dmdod.evaluation.hold_judgement_v01 import TimeWindow
from dmdod.motor.contact_16k import (
    ContactAction,
    Coupled16ContactMotor,
    PHYSICAL_AXES,
    ThumbMode,
)
from dmdod.planning.persistent_fingering_scheduler_v01 import PhysicalFinger


class HoldTimingProvenance(str, Enum):
    CAUSAL_VISIBLE_INTERVAL = "causal_visible_interval"
    ORACLE_DIAGNOSTIC = "oracle_diagnostic"
    UNVERIFIED = "unverified"


class HoldMotorPhase(str, Enum):
    RESERVED = "reserved"
    PRESSING = "pressing"
    HOLDING = "holding"
    RELEASING = "releasing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class HoldMotorPlan:
    """Timing and assignment supplied by upstream modules, never chart truth.

    Due intervals describe evaluator uncertainty.  Command times are separate:
    no hidden physical-delay compensation is inferred in this module.
    """

    local_id: int
    finger: PhysicalFinger
    start_due: TimeWindow
    release_due: TimeWindow
    start_command_s: float
    release_command_s: float
    provenance: HoldTimingProvenance

    def __post_init__(self) -> None:
        if type(self.local_id) is not int or self.local_id < 0:
            raise ValueError("local_id must be a nonnegative integer")
        if not isinstance(self.finger, PhysicalFinger):
            raise TypeError("finger must be PhysicalFinger")
        if not isinstance(self.start_due, TimeWindow) or not isinstance(
            self.release_due, TimeWindow
        ):
            raise TypeError("Hold due values must be TimeWindow objects")
        if self.release_due.lower_s < self.start_due.upper_s:
            raise ValueError("release due interval overlaps start due interval")
        if not isfinite(self.start_command_s) or self.start_command_s < 0.0:
            raise ValueError("start command must be finite and nonnegative")
        if not isfinite(self.release_command_s):
            raise ValueError("release command must be finite")
        if self.release_command_s < self.start_command_s:
            raise ValueError("release command cannot precede start command")
        if self.start_command_s > self.start_due.upper_s:
            raise ValueError("start command occurs after the start window")
        if self.release_command_s > self.release_due.upper_s:
            raise ValueError("release command occurs after the release window")
        if not isinstance(self.provenance, HoldTimingProvenance):
            raise TypeError("provenance must be HoldTimingProvenance")


@dataclass(frozen=True, slots=True)
class ActiveHoldMotorPlan:
    plan: HoldMotorPlan
    phase: HoldMotorPhase = HoldMotorPhase.RESERVED
    physical_keydown_s: float | None = None
    physical_keyup_s: float | None = None
    failure_reason: str | None = None


class HoldMotorOverlay:
    """Compose Hold ownership over an ordinary Fingering Motor action.

    Register only after the normal Fingering policy selected ``plan.finger``.
    The same physical reservation can then remain pressed after its ordinary
    KeyDown while unrelated axes continue to follow the base action.
    """

    def __init__(self, *, allow_oracle_diagnostic: bool = False) -> None:
        if type(allow_oracle_diagnostic) is not bool:
            raise TypeError("allow_oracle_diagnostic must be bool")
        self.allow_oracle_diagnostic = allow_oracle_diagnostic
        self.reset()

    def reset(self) -> None:
        self._active: dict[int, ActiveHoldMotorPlan] = {}
        self._completed: list[int] = []

    @property
    def active(self) -> tuple[ActiveHoldMotorPlan, ...]:
        return tuple(self._active[key] for key in sorted(self._active))

    @property
    def completed_ids(self) -> tuple[int, ...]:
        return tuple(self._completed)

    @property
    def occupied_axes(self) -> frozenset[str]:
        return frozenset(item.plan.finger.axis for item in self._active.values())

    def register(self, plan: HoldMotorPlan, motor: Coupled16ContactMotor) -> None:
        if not isinstance(plan, HoldMotorPlan):
            raise TypeError("expected HoldMotorPlan")
        if not isinstance(motor, Coupled16ContactMotor):
            raise TypeError("expected Coupled16ContactMotor")
        if plan.provenance is HoldTimingProvenance.UNVERIFIED:
            raise ValueError("unverified Hold timing cannot drive Motor")
        if (
            plan.provenance is HoldTimingProvenance.ORACLE_DIAGNOSTIC
            and not self.allow_oracle_diagnostic
        ):
            raise ValueError("Oracle Hold timing requires explicit diagnostic opt-in")
        if plan.local_id in self._active or plan.local_id in self._completed:
            raise ValueError("Hold local ID cannot be reused")
        if plan.finger.axis in self.occupied_axes:
            raise ValueError("Hold physical axis is already occupied")
        physical = motor.motor.observe()
        axis_index = PHYSICAL_AXES.index(plan.finger.axis)
        if physical.pressed_flags[axis_index]:
            raise ValueError("Hold physical axis is already pressed")
        self._active[plan.local_id] = ActiveHoldMotorPlan(plan)

    @staticmethod
    def _topology(
        finger: PhysicalFinger,
        modes: list[ThumbMode],
        edges: list[bool],
    ) -> None:
        hand, depth = finger.axis.rsplit("_", 1)
        side = 0 if hand == "left" else 1
        if depth == "5":
            modes[side] = (
                ThumbMode.FLAT if finger.contact_count == 2 else ThumbMode.TIP
            )
        elif depth == "6":
            edges[side] = True

    def command(
        self,
        *,
        now_s: float,
        base_action: ContactAction,
        motor: Coupled16ContactMotor,
    ) -> ContactAction:
        if not isfinite(now_s) or now_s < 0.0:
            raise ValueError("now_s must be finite and nonnegative")
        if not isinstance(base_action, ContactAction):
            raise TypeError("base_action must be ContactAction")
        if not isinstance(motor, Coupled16ContactMotor):
            raise TypeError("motor must be Coupled16ContactMotor")
        commands = dict(zip(PHYSICAL_AXES, base_action.commands))
        modes = list(base_action.thumb_modes)
        edges = list(base_action.edge_enabled)
        physical = motor.motor.observe()

        for local_id, item in tuple(self._active.items()):
            plan = item.plan
            axis = plan.finger.axis
            axis_index = PHYSICAL_AXES.index(axis)
            pressed = physical.pressed_flags[axis_index]
            phase = item.phase

            if phase is HoldMotorPhase.RESERVED:
                if pressed:
                    self._active[local_id] = replace(
                        item,
                        phase=HoldMotorPhase.FAILED,
                        failure_reason="axis_pressed_before_hold_command",
                    )
                    raise ValueError("Hold axis pressed before Hold command")
                if now_s < plan.start_command_s:
                    if commands[axis] > 0.0:
                        raise ValueError("base action drives occupied axis before Hold command")
                    commands[axis] = min(commands[axis], 0.0)
                    continue
                phase = HoldMotorPhase.PRESSING
                item = replace(item, phase=phase)
                self._active[local_id] = item

            if phase is HoldMotorPhase.PRESSING:
                if pressed:
                    phase = HoldMotorPhase.HOLDING
                    item = replace(
                        item, phase=phase, physical_keydown_s=float(now_s)
                    )
                    self._active[local_id] = item
                elif now_s > plan.start_due.upper_s + 1e-12:
                    self._active[local_id] = replace(
                        item,
                        phase=HoldMotorPhase.FAILED,
                        failure_reason="physical_keydown_missed_start_window",
                    )
                    commands[axis] = -1.0
                    continue

            if phase in (HoldMotorPhase.PRESSING, HoldMotorPhase.HOLDING):
                self._topology(plan.finger, modes, edges)
                if now_s < plan.release_command_s:
                    # Hold ownership deliberately overrides the base policy's
                    # ordinary post-KeyDown release command.
                    commands[axis] = 1.0
                else:
                    self._active[local_id] = replace(
                        item, phase=HoldMotorPhase.RELEASING
                    )
                    commands[axis] = -1.0
            elif phase is HoldMotorPhase.RELEASING:
                self._topology(plan.finger, modes, edges)
                commands[axis] = -1.0
            elif phase is HoldMotorPhase.FAILED:
                commands[axis] = -1.0

        return ContactAction(
            tuple(commands[axis] for axis in PHYSICAL_AXES),
            tuple(modes),
            tuple(edges),
        )

    def observe_physics(
        self, *, now_s: float, motor: Coupled16ContactMotor
    ) -> None:
        """Advance from physical state only; evaluator Hit/Miss is forbidden."""
        if not isfinite(now_s) or now_s < 0.0:
            raise ValueError("now_s must be finite and nonnegative")
        if not isinstance(motor, Coupled16ContactMotor):
            raise TypeError("motor must be Coupled16ContactMotor")
        physical = motor.motor.observe()
        for local_id, item in tuple(self._active.items()):
            axis_index = PHYSICAL_AXES.index(item.plan.finger.axis)
            pressed = physical.pressed_flags[axis_index]
            if item.phase is HoldMotorPhase.PRESSING and pressed:
                self._active[local_id] = replace(
                    item,
                    phase=HoldMotorPhase.HOLDING,
                    physical_keydown_s=float(now_s),
                )
            elif item.phase is HoldMotorPhase.RELEASING and not pressed:
                self._completed.append(local_id)
                self._active.pop(local_id)
