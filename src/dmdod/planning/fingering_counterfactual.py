from __future__ import annotations

"""Offline physical counterfactuals for fingering allocation (oracle diagnostics ONLY).

Each candidate starts from an identical copied 12-axis physical state. This
tests local actuation feasibility, NOT whole-chart Hits or learned fingering.
Never pass privileged due_s or results to the frozen Student as observations.
"""
from dataclasses import dataclass
import copy
from math import isfinite

from dmdod.motor.contact_16k import PHYSICAL_AXES, Coupled16ContactMotor
from dmdod.motor.keyboard import KeyEvent
from dmdod.motor.n_key import NKeyAction


@dataclass(frozen=True)
class AxisOutcome:
    axis: str
    first_down_s: float | None
    abs_error_ms: float | None
    extra_down_axes: tuple[str, ...]
    already_pressed: bool


def evaluate_axis_alternatives(
    motor: Coupled16ContactMotor,
    due_s: float,
    *,
    base_commands: tuple[float, ...] | None = None,
    candidates: tuple[str, ...] | None = None,
    horizon_frames: int = 7,
) -> tuple[AxisOutcome, ...]:
    """Compare one-axis press replacements under the same physical state.

    Other axes follow base_commands, but any positive command on a candidate
    axis is replaced with +1. This is a LOCAL counterfactual; future policy
    adaptation, reservations, topology changes and scorer are not simulated.
    """
    if not isfinite(due_s) or horizon_frames < 1 or horizon_frames > 100:
        raise ValueError("finite due_s and horizon_frames in [1,100] required")
    axes = PHYSICAL_AXES
    base = tuple(0.0 for _ in axes) if base_commands is None else tuple(base_commands)
    if len(base) != len(axes) or any(not isfinite(v) or not -1 <= v <= 1 for v in base):
        raise ValueError("invalid base commands")
    selected = axes if candidates is None else tuple(candidates)
    if len(set(selected)) != len(selected) or any(a not in axes for a in selected):
        raise ValueError("unknown or repeated physical axis")
    physical = motor.motor
    obs = physical.observe()
    outcomes = []
    for axis in selected:
        index = axes.index(axis)
        already_pressed = obs.pressed_flags[index]
        commands = list(base)
        commands[index] = 1.
        branch = copy.deepcopy(physical)
        first = None
        extras = set()
        for _ in range(horizon_frames):
            transition = branch.step(NKeyAction(tuple(commands)))
            for ev in transition.evaluator_events:
                if ev.event is KeyEvent.DOWN:
                    if ev.key == axis and first is None:
                        first = ev.time_s
                    elif ev.key != axis:
                        extras.add(ev.key)
        outcomes.append(AxisOutcome(
            axis=axis, first_down_s=first,
            abs_error_ms=None if first is None else abs(first-due_s)*1000,
            extra_down_axes=tuple(sorted(extras)),
            already_pressed=already_pressed,
        ))
    return tuple(outcomes)


def rank_feasible_axes(outcomes: tuple[AxisOutcome, ...]) -> tuple[AxisOutcome, ...]:
    """Deterministic descriptive ranking, not an optimized learned policy."""
    return tuple(sorted(outcomes, key=lambda x: (
        x.already_pressed, x.first_down_s is None,
        len(x.extra_down_axes),
        float("inf") if x.abs_error_ms is None else x.abs_error_ms,
        x.axis,
    )))
