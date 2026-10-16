from __future__ import annotations

"""Evaluator-only reservation swap forecasts. Never an input to HUD-only Student."""
from dataclasses import dataclass
from math import isfinite
import copy

from dmdod.motor.contact_16k import ContactAction, Coupled16ContactMotor, PHYSICAL_AXES
from dmdod.motor.keyboard import KeyEvent


@dataclass(frozen=True)
class ReservationForecast:
    source_axis: str
    candidate_axis: str
    reserved_ordinals: tuple[int, ...]
    projected: tuple[tuple[int | None, str, float], ...]
    missing_ordinals: tuple[int, ...]
    extra_keydowns: tuple[str, ...]
    order_inversions: int
    timing_mae_ms: float | None
    feasible: bool


def _contact_keys(motor: Coupled16ContactMotor, action: ContactAction, axis: str) -> tuple[str, ...]:
    hand, depth = axis.split("_")
    side = 0 if hand == "left" else 1
    return motor.contact_keys(hand, int(depth), action.thumb_modes[side],
                              edge_enabled=action.edge_enabled[side])


def forecast_reservation_swap(
    motor: Coupled16ContactMotor,
    action: ContactAction,
    reservations: dict[str, tuple[int, ...]],
    note_times: dict[int, float],
    source_axis: str,
    candidate_axis: str,
    *,
    horizon_frames: int = 7,
) -> ReservationForecast:
    """Compare a *single* reservation moved from source to candidate.

    Current physical state and all other active commands are preserved. The
    source is released and candidate pressed. Each logical DOWN is attributed
    to its candidate-reserved ordinal, in scorer emission order. This does
    not run the score evaluator or subsequent teacher replanning.
    """
    if source_axis not in reservations:
        raise ValueError("source axis has no reservation")
    if source_axis not in PHYSICAL_AXES or candidate_axis not in PHYSICAL_AXES:
        raise ValueError("unknown axis")
    if source_axis == candidate_axis or candidate_axis in reservations:
        raise ValueError("candidate must be a different, unreserved physical axis")
    if not 1 <= horizon_frames <= 100:
        raise ValueError("horizon_frames outside [1,100]")
    tokens = tuple(reservations[source_axis])
    if not tokens or len(set(tokens)) != len(tokens):
        raise ValueError("invalid reserved note ordinals")
    if any(t not in note_times or not isfinite(note_times[t]) for t in tokens):
        raise ValueError("note times missing or invalid")
    if len(_contact_keys(motor, action, candidate_axis)) != len(tokens):
        raise ValueError("candidate physical axis has different contact multiplicity")
    physical = motor.motor.observe()
    candidate_index = PHYSICAL_AXES.index(candidate_axis)
    if physical.pressed_flags[candidate_index]:
        raise ValueError("candidate is already physically pressed")
    if physical.positions_m[candidate_index] > motor.motor.keyboard.config.reset_m:
        raise ValueError("candidate has not returned to reset position")

    commands = list(action.commands)
    source_index = PHYSICAL_AXES.index(source_axis)
    commands[source_index] = -1.
    commands[candidate_index] = 1.
    candidate_action = ContactAction(tuple(commands), action.thumb_modes, action.edge_enabled)
    clone = copy.deepcopy(motor)
    selected = dict(reservations)
    del selected[source_axis]
    selected[candidate_axis] = tokens
    key_to_ordinal = {}
    for axis, ordinal_tuple in selected.items():
        contact_keys = _contact_keys(clone, candidate_action, axis)
        for key, ordinal in zip(contact_keys, ordinal_tuple):
            key_to_ordinal[key] = ordinal
    projected = []
    seen = set()
    extras = []
    for _ in range(horizon_frames):
        transition = clone.step(candidate_action)
        for event in transition.logical_events:
            if event.event is not KeyEvent.DOWN:
                continue
            ordinal = key_to_ordinal.get(event.key)
            projected.append((ordinal, event.key, event.time_s))
            if ordinal is None or ordinal in seen:
                extras.append(event.key)
            else:
                seen.add(ordinal)
    missing = tuple(t for t in tokens if t not in seen)
    ordinals = [entry[0] for entry in projected if entry[0] is not None]
    inversions = sum(a > b for i, a in enumerate(ordinals) for b in ordinals[i+1:])
    error = [abs(ts-note_times[ordinal])*1000 for ordinal, _, ts in projected
             if ordinal is not None and ordinal in note_times]
    # A local no-error forecast is not proof of a successful scored note.
    feasible = not missing and not extras and not inversions
    return ReservationForecast(
        source_axis, candidate_axis, tokens, tuple(projected),
        missing, tuple(extras), inversions,
        sum(error)/len(error) if error else None, feasible,
    )
