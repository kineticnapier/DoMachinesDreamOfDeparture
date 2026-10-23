from __future__ import annotations

"""Research-only state-aware Motor command timing for DMDOGO v0.4."""

import copy
from dataclasses import dataclass
from math import ceil

from dmdod.motor.contact_16k import (
    ContactAction,
    Coupled16ContactMotor,
    HANDS,
    PHYSICAL_AXES,
    ThumbMode,
)
from dmdod.motor.keyboard import KeyEvent


@dataclass(frozen=True, slots=True)
class LatencyPrediction:
    valid: bool
    earliest_arrival_s: float | None
    contacts: tuple[str, ...]
    reason: str
    simulated_steps: int = 0


@dataclass(frozen=True, slots=True)
class LeadDecision:
    start: bool
    fallback: bool
    reason: str
    predicted_keydown_s: float | None


class MotorLeadCompensator:
    """Predict the fastest isolated stroke from current visible Motor state.

    The cloned forward model receives no chart target, scorer state or future
    physical observation. Other axes are neutral or released, making this an
    optimistic (earliest) arrival estimate under the frozen coupled dynamics.
    Commands are latched by the caller once started; this class never controls
    a live Motor itself.
    """

    def __init__(self, *, max_prediction_s: float = .090,
                 legacy_lead_s: float = .044) -> None:
        if max_prediction_s <= 0 or legacy_lead_s <= 0:
            raise ValueError("prediction horizon and legacy lead must be positive")
        self.max_prediction_s = float(max_prediction_s)
        self.legacy_lead_s = float(legacy_lead_s)

    @staticmethod
    def _topology(axis: str, mode: str) -> tuple[tuple[ThumbMode, ThumbMode],
                                                  tuple[bool, bool], tuple[str, ...]]:
        hand, depth_text = axis.split("_")
        if hand not in HANDS or axis not in PHYSICAL_AXES:
            raise ValueError(f"unknown physical axis: {axis}")
        depth = int(depth_text)
        modes = [ThumbMode.OFF, ThumbMode.OFF]
        edges = [False, False]
        hand_index = HANDS.index(hand)
        if depth <= 4 and mode == "single":
            contacts = (axis,)
        elif depth == 5 and mode in {"tip", "flat"}:
            modes[hand_index] = ThumbMode.TIP if mode == "tip" else ThumbMode.FLAT
            contacts = ((f"{hand}_5",) if mode == "tip"
                        else (f"{hand}_5", f"{hand}_6"))
        elif depth == 6 and mode == "edge":
            edges[hand_index] = True
            contacts = (f"{hand}_7", f"{hand}_8")
        else:
            raise ValueError(f"mode {mode!r} is invalid for {axis}")
        return tuple(modes), tuple(edges), contacts

    def predict(self, motor: Coupled16ContactMotor, *, axis: str,
                mode: str) -> LatencyPrediction:
        modes, edges, contacts = self._topology(axis, mode)
        hand, depth_text = axis.split("_")
        depth = int(depth_text)
        hand_index = HANDS.index(hand)
        merged_modes = list(motor._modes)
        merged_edges = list(motor._edge)
        if depth == 5:
            merged_modes[hand_index] = modes[hand_index]
        elif depth == 6:
            merged_edges[hand_index] = edges[hand_index]
        modes = tuple(merged_modes)
        edges = tuple(merged_edges)
        observation = motor.motor.observe()
        index = observation.index(axis)
        if observation.pressed_flags[index]:
            return LatencyPrediction(False, None, contacts, "axis_pressed")
        release = motor.motor.keyboard.config.reset_m
        if ((axis.endswith("_5") or axis.endswith("_6"))
                and (observation.positions_m[index] > release
                     or (observation.positions_m[index] > 0.0
                         and observation.velocities_m_s[index] > .0001))):
            return LatencyPrediction(False, None, contacts, "topology_not_at_rest")

        branch = copy.deepcopy(motor)
        commands = []
        for name, pressed in zip(observation.key_names, observation.pressed_flags):
            commands.append(-1.0 if pressed else 1.0 if name == axis else 0.0)
        action = ContactAction(tuple(commands), modes, edges)
        start_s = float(branch.motor.time_s)
        steps = ceil(self.max_prediction_s / branch.motor.control_dt_s)
        wanted = set(contacts)
        seen: set[str] = set()
        first_s: float | None = None
        for step in range(1, steps + 1):
            transition = branch.step(action)
            for event in transition.logical_events:
                if event.event is KeyEvent.DOWN and event.key in wanted:
                    seen.add(event.key)
                    first_s = event.time_s if first_s is None else min(first_s, event.time_s)
            if seen == wanted:
                return LatencyPrediction(
                    True, float(first_s - start_s), contacts,
                    "isolated_fastest_forward_model", step,
                )
        return LatencyPrediction(False, None, contacts, "no_arrival_within_horizon", steps)

    def decide(self, *, now_s: float, due_s: float,
               prediction: LatencyPrediction) -> LeadDecision:
        if prediction.valid:
            predicted = float(now_s + prediction.earliest_arrival_s)
            start = predicted + 1e-12 >= due_s
            return LeadDecision(
                start, False,
                "earliest_arrival_reached_due" if start else "too_early_guard_wait",
                predicted,
            )
        start = now_s + self.legacy_lead_s + 1e-12 >= due_s
        return LeadDecision(
            start, True,
            "legacy_44ms_fallback" if start else "fallback_wait",
            None,
        )
