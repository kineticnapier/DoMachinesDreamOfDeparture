from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(slots=True)
class TrustRadiusController:
    initial: float
    minimum: float
    reject_shrink: float
    safe_grow: float
    current: float

    @classmethod
    def create(
        cls,
        *,
        initial: float,
        minimum: float,
        reject_shrink: float,
        safe_grow: float,
    ) -> "TrustRadiusController":
        for name, value in (
            ("initial", initial),
            ("minimum", minimum),
            ("reject_shrink", reject_shrink),
            ("safe_grow", safe_grow),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if initial <= 0.0 or minimum <= 0.0:
            raise ValueError("trust radii must be positive")
        if minimum > initial:
            raise ValueError("minimum trust radius cannot exceed initial radius")
        if not (0.0 < reject_shrink < 1.0):
            raise ValueError("reject_shrink must be in (0, 1)")
        if safe_grow < 1.0:
            raise ValueError("safe_grow must be >= 1")
        return cls(
            initial=float(initial),
            minimum=float(minimum),
            reject_shrink=float(reject_shrink),
            safe_grow=float(safe_grow),
            current=float(initial),
        )

    @property
    def at_floor(self) -> bool:
        return self.current <= self.minimum

    def reject(self) -> bool:
        """Shrink after rejection.

        Returns True when another trial should be attempted. The minimum radius
        is always tried exactly once before this returns False.
        """

        if self.at_floor:
            return False
        self.current = max(
            self.minimum,
            self.current * self.reject_shrink,
        )
        return True

    def accept(self) -> float:
        self.current = min(
            self.initial,
            self.current * self.safe_grow,
        )
        return self.current


@dataclass(slots=True)
class NonBestRestartController:
    limit: int
    streak: int = 0
    restarts: int = 0

    @classmethod
    def create(cls, *, limit: int) -> "NonBestRestartController":
        # Do not silently truncate fractional limits or accept bool as 1.
        if type(limit) is not int or limit <= 0:
            raise ValueError("non-best restart limit must be a positive integer")
        return cls(limit=limit)

    def observe(self, *, selected_best: bool) -> bool:
        """Return True when exploration should restart from the selected best."""

        if selected_best:
            self.streak = 0
            return False

        self.streak += 1
        if self.streak < self.limit:
            return False

        self.streak = 0
        self.restarts += 1
        return True
