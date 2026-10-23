from __future__ import annotations

"""Deterministic physical handoff for the abstract Pattern-first scheduler.

The abstract keys are not independent physical fingers. The fixed default
mapping uses only the eight uncoupled ordinary axes of the 12-axis Motor.
Live Motor eligibility is checked by the caller before any plan is committed.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Sequence


@dataclass(frozen=True, slots=True)
class PhysicalChoice:
    axes: tuple[str, ...] | None
    disposition: str
    desired_reason: str
    attempted: int


@dataclass(frozen=True, slots=True)
class PatternMotorMapping:
    logical_keys: tuple[str, ...]
    physical_axes: tuple[str, ...]
    network_used: bool = False
    max_candidates: int = 32

    def __post_init__(self) -> None:
        if not self.logical_keys or len(self.logical_keys) != len(self.physical_axes):
            raise ValueError("logical key and physical axis counts must be nonzero and equal")
        if len(set(self.logical_keys)) != len(self.logical_keys):
            raise ValueError("logical keys are not unique")
        if len(set(self.physical_axes)) != len(self.physical_axes):
            raise ValueError("physical axes are not unique")
        ordinary = {f"{hand}_{depth}" for hand in ("left", "right")
                    for depth in range(1, 5)}
        if any(axis not in ordinary for axis in self.physical_axes):
            raise ValueError("v0.6 mapping accepts ordinary physical axes 1..4 only")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")

    @classmethod
    def default_eight(cls) -> "PatternMotorMapping":
        # Fixed representation, not an optimized anatomical preference.
        axes = ("left_1", "right_1", "left_2", "right_2",
                "left_3", "right_3", "left_4", "right_4")
        return cls(tuple(f"key_{index}" for index in range(8)), axes)

    def axis_for(self, logical_key: str) -> str:
        try:
            return self.physical_axes[self.logical_keys.index(logical_key)]
        except ValueError as error:
            raise KeyError(logical_key) from error

    def desired_axes(self, logical_keys: Sequence[str]) -> tuple[str, ...]:
        axes = tuple(self.axis_for(key) for key in logical_keys)
        if not axes or len(set(axes)) != len(axes):
            raise ValueError("allocation requires distinct logical keys and physical axes")
        return axes

    def candidate_axes(self, logical_keys: Sequence[str],
                       allow_local_replan: bool) -> tuple[tuple[str, ...], ...]:
        desired = self.desired_axes(logical_keys)
        if not allow_local_replan:
            return (desired,)
        alternatives = (value for value in combinations(self.physical_axes, len(desired))
                        if value != desired)
        result = [desired]
        for item in alternatives:
            if len(result) >= self.max_candidates:
                break
            result.append(item)
        return tuple(result)

    def choose(self, logical_keys: Sequence[str], *, remaining_s: float,
               legacy_lead_s: float,
               eligible: Callable[[tuple[str, ...]], tuple[bool, str]]) -> PhysicalChoice:
        """Defer an unavailable scheduled allocation until the 44ms deadline.

        Physical fallback is only a bounded candidate search once inside the
        legacy selection window. Never assumes a future physical state.
        """
        allow_fallback = remaining_s <= legacy_lead_s + 1e-9
        desired_reason = "not_checked"
        count = 0
        for count, axes in enumerate(self.candidate_axes(logical_keys, allow_fallback), 1):
            allowed, reason = eligible(axes)
            if count == 1:
                desired_reason = reason
            if allowed:
                return PhysicalChoice(axes, "desired" if count == 1 else "local_replan",
                                      desired_reason, count)
        return PhysicalChoice(None, "no_live_candidate" if allow_fallback else "deferred",
                              desired_reason, count)
