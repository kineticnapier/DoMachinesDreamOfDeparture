from __future__ import annotations

"""Truth-free candidate ranking for frozen FingeringRanker and 12-axis Motor.

The topology/candidate order mirrors the frozen teacher's *physical candidate
enumeration* without importing the oracle timing or evaluator into policy.
"""

from dataclasses import dataclass
from itertools import combinations

from dmdod.motor.contact_16k import (
    Coupled16ContactMotor, PHYSICAL_AXES, ThumbMode,
)
from dmdod.planning.persistent_fingering_scheduler_v01 import (
    MotorPhysicalSnapshot, PhysicalFinger, PersistentFingeringScheduler,
)


@dataclass(frozen=True, slots=True)
class CandidateFinger:
    axis: str
    mode: str = "single"

    def __post_init__(self) -> None:
        depth = int(self.axis.rsplit("_", 1)[1]) if self.axis in PHYSICAL_AXES else 0
        expected = {
            1: ("single",), 2: ("single",),
            3: ("single",), 4: ("single",),
            5: ("tip", "flat"), 6: ("edge",)
        }.get(depth, ())
        if self.mode not in expected:
            raise ValueError("invalid physical fingering candidate topology")

    @property
    def contact_count(self) -> int:
        return 2 if self.mode in ("flat", "edge") else 1

    @property
    def label(self) -> str:
        return self.axis + ("/" + self.mode if self.mode != "single" else "")


def finger_options(chord_size: int) -> tuple[tuple[CandidateFinger, ...], ...]:
    singles = tuple(CandidateFinger(
        axis, "tip" if axis.endswith("_5") else "single"
    ) for axis in PHYSICAL_AXES if not axis.endswith("_6"))
    doubles = tuple(CandidateFinger(
        axis, "flat" if axis.endswith("_5") else "edge"
    ) for axis in PHYSICAL_AXES if axis.endswith(("_5", "_6")))
    if chord_size == 1:
        return tuple((f,) for f in singles)
    if chord_size == 2:
        return tuple((f,) for f in doubles) + tuple(combinations(singles, 2))
    if 3 <= chord_size <= 8:
        return tuple(combinations(singles, chord_size))[:60]
    return ()


def to_physical_fingers(allocation: tuple[CandidateFinger, ...]
                        ) -> tuple[PhysicalFinger, ...]:
    return tuple(PhysicalFinger(f.axis, f.contact_count) for f in allocation)


def is_eligible(
    allocation: tuple[CandidateFinger, ...],
    scheduler: PersistentFingeringScheduler,
    motor: Coupled16ContactMotor,
    snapshot: MotorPhysicalSnapshot,
) -> bool:
    if not allocation or len({x.axis for x in allocation}) != len(allocation):
        return False
    allowed = set(scheduler.available_axes(snapshot))
    if any(x.axis not in allowed for x in allocation):
        return False
    modes = motor._modes
    edges = motor._edge
    for finger in allocation:
        hand, depth = finger.axis.rsplit("_", 1)
        hand_index = 0 if hand == "left" else 1
        if depth == "5":
            target = ThumbMode.FLAT if finger.mode == "flat" else ThumbMode.TIP
            if modes[hand_index] != target and not snapshot.ready(finger.axis):
                return False
        if depth == "6":
            if edges[hand_index] is False and not snapshot.ready(finger.axis):
                return False
    return True


def center_first(
    candidates: tuple[tuple[CandidateFinger, ...], ...],
    eligible: tuple[bool, ...],
) -> int | None:
    """Deterministic untrained comparator; no frozen ranker or teacher scores."""
    valid = [i for i, ok in enumerate(eligible) if ok]
    if not valid:
        return None
    return min(valid, key=lambda i: (
        sum(int(f.axis.rsplit("_", 1)[1]) for f in candidates[i]),
        tuple(f.axis for f in candidates[i]),
        i,
    ))
