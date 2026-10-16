from __future__ import annotations

"""Experimental two-hand 16-contact keyboard driven by twelve motor axes.

Each hand uses four independent ordinary fingers, ONE thumb axis with an
'off'/'tip'/'flat' contact selection, and ONE coupled hand-edge axis touching
both edge keys.  The contact layer never creates eight independent actuators.

This is a simplifying *contact topology*, not a calibrated human hand, thumb
posture transition, forearm fatigue, or a trained DMDOD Student.  The underlying
12-axis NKeyBody physical constants are reused WITHOUT modification.
"""

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from dmdod.motor.env import TimedKeyEvent
from dmdod.motor.keyboard import KeyEvent
from dmdod.motor.n_key import (
    NKeyAction, NKeyDiagnostics, NKeyMotorEnv, NKeyObservation, n_key_names,
)


HANDS = ("left", "right")
PHYSICAL_AXES = n_key_names(12)
LOGICAL_KEYS = n_key_names(16)


class ThumbMode(str, Enum):
    OFF = "off"
    TIP = "tip"  # One key, contact #5
    FLAT = "flat"  # Two linked contacts, #5 and #6


@dataclass(frozen=True)
class ContactAction:
    """Commands use physical NKeyBody(12) order, *not* 16 logical keys.

    On each hand, positions 1-4 represent four independent fingers,
    position 5 the thumb axis, and position 6 the edge axis.
    """

    commands: tuple[float, ...]
    thumb_modes: tuple[ThumbMode, ThumbMode] = (ThumbMode.OFF, ThumbMode.OFF)
    edge_enabled: tuple[bool, bool] = (False, False)

    def __post_init__(self) -> None:
        if len(self.commands) != len(PHYSICAL_AXES):
            raise ValueError("Exactly 12 physical commands required")
        if any(not isfinite(v) or not -1. <= v <= 1. for v in self.commands):
            raise ValueError("Motor commands must be finite in [-1, 1]")
        if len(self.thumb_modes) != 2 or any(not isinstance(x, ThumbMode)
                                                   for x in self.thumb_modes):
            raise ValueError("One valid ThumbMode per hand required")
        if len(self.edge_enabled) != 2 or any(type(x) is not bool
                                                   for x in self.edge_enabled):
            raise ValueError("One edge-enabled bool per hand required")
        for index, hand in enumerate(HANDS):
            commands = dict(zip(PHYSICAL_AXES, self.commands))
            if self.thumb_modes[index] is ThumbMode.OFF and commands[f"{hand}_5"] > 0:
                raise ValueError("Cannot actuate a disabled thumb contact")
            if not self.edge_enabled[index] and commands[f"{hand}_6"] > 0:
                raise ValueError("Cannot actuate a disabled hand-edge contact")


@dataclass(frozen=True)
class ContactObservation:
    physical: NKeyObservation  # Only twelve physical actuators
    logical_keys: tuple[str, ...]
    logical_pressed: tuple[bool, ...]
    thumb_modes: tuple[ThumbMode, ThumbMode]
    edge_enabled: tuple[bool, bool]

    def pressed(self, key: str) -> bool:
        return self.logical_pressed[self.logical_keys.index(key)]


@dataclass(frozen=True)
class ContactTransition:
    observation: ContactObservation
    logical_events: tuple[TimedKeyEvent, ...]
    diagnostics: NKeyDiagnostics


class Coupled16ContactMotor:
    """12 real motor axes -> 16 named contact switches; no privileged chart.

    Switching thumb pose or engaging/disengaging the hand edge is permitted
    ONLY after that physical axis has returned below the key's reset point.
    This forbids phantom extra KeyDown events from mid-stroke pose toggles.
    A complete real thumb posture model is not present; changing pose at
    rest incurs no time cost and is an explicit modeling limitation.
    """

    def __init__(self, *, control_dt_s: float = .010,
                 physics_dt_s: float = .001):
        self.motor = NKeyMotorEnv(12, control_dt_s=control_dt_s,
                                  physics_dt_s=physics_dt_s)
        self.reset()

    def reset(self) -> ContactObservation:
        self.motor.reset()
        self._modes = (ThumbMode.OFF, ThumbMode.OFF)
        self._edge = (False, False)
        self._logical_pressed = {key: False for key in LOGICAL_KEYS}
        return self.observe()

    @staticmethod
    def visible_keys_for_hand(hand: str, mode: ThumbMode,
                              *, edge_enabled: bool) -> tuple[str, ...]:
        if hand not in HANDS or not isinstance(mode, ThumbMode):
            raise ValueError("Invalid hand or thumb mode")
        digits = [f"{hand}_{depth}" for depth in range(1, 5)]
        if mode is ThumbMode.TIP:
            digits.append(f"{hand}_5")
        elif mode is ThumbMode.FLAT:
            digits.extend((f"{hand}_5", f"{hand}_6"))
        if edge_enabled:
            digits.extend((f"{hand}_7", f"{hand}_8"))
        return tuple(digits)

    @staticmethod
    def contact_keys(hand: str, physical_index: int,
                     mode: ThumbMode, *, edge_enabled: bool) -> tuple[str, ...]:
        if hand not in HANDS or not isinstance(mode, ThumbMode):
            raise ValueError("Invalid hand or thumb mode")
        if 1 <= physical_index <= 4:
            return (f"{hand}_{physical_index}",)
        if physical_index == 5:
            return (() if mode is ThumbMode.OFF else
                    (f"{hand}_5",) if mode is ThumbMode.TIP else
                    (f"{hand}_5", f"{hand}_6"))
        if physical_index == 6:
            return (f"{hand}_7", f"{hand}_8") if edge_enabled else ()
        raise ValueError("Physical axis must be 1..6")

    def observe(self) -> ContactObservation:
        return ContactObservation(
            self.motor.observe(), LOGICAL_KEYS,
            tuple(self._logical_pressed[k] for k in LOGICAL_KEYS),
            self._modes, self._edge,
        )

    def step(self, action: ContactAction) -> ContactTransition:
        # Check all possible mode switches before mutating the simulation.
        obs = self.motor.observe()
        positions = dict(zip(obs.key_names, obs.positions_m))
        velocities = dict(zip(obs.key_names, obs.velocities_m_s))
        pressed = dict(zip(obs.key_names, obs.pressed_flags))
        release = self.motor.keyboard.config.reset_m
        for i, hand in enumerate(HANDS):
            for physical_depth, switched in (
                (5, action.thumb_modes[i] != self._modes[i]),
                (6, action.edge_enabled[i] != self._edge[i]),
            ):
                name = f"{hand}_{physical_depth}"
                if switched and (pressed[name] or positions[name] > release
                                 or (positions[name] > 0. and velocities[name] > .0001)):
                    raise ValueError("Cannot switch contact topology during physical stroke")
        self._modes = action.thumb_modes
        self._edge = action.edge_enabled
        output: list[TimedKeyEvent] = []
        transition = self.motor.step(NKeyAction(action.commands))
        for event in transition.evaluator_events:
            hand, number = event.key.split("_")
            i = HANDS.index(hand)
            destinations = self.contact_keys(hand, int(number), self._modes[i],
                                             edge_enabled=self._edge[i])
            for key in destinations:
                if event.event is KeyEvent.DOWN:
                    if self._logical_pressed[key]:
                        raise RuntimeError("Duplicate logical KeyDown")
                    self._logical_pressed[key] = True
                elif event.event is KeyEvent.UP:
                    if not self._logical_pressed[key]:
                        raise RuntimeError("Logical KeyUp without KeyDown")
                    self._logical_pressed[key] = False
                output.append(TimedKeyEvent(event.time_s, key, event.event))
        return ContactTransition(self.observe(), tuple(output),
                                 transition.diagnostics)
