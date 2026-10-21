"""Audited-DLL reference for ordinary Stock floor consumption.

This is a research reference, not a replacement for the production scorer.
It combines Phase 20A's angle classifier with the ordinary two-planet control
flow in ``scrPlayer``/``scrPlanet``.  Unsupported Stock branches stop the
trace instead of silently borrowing repository scorer behaviour.

Chart target times are evaluator-side inputs used to reconstruct the planet
angle.  They are never exposed to a Timing, Fingering, or Motor policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable

from .playable import PlayableChartSegment
from .stock_judgement_reference import (
    StockHitMargin,
    StockTimingDifficulty,
    classify_stock_angle,
    stock_angle_boundaries,
)


class StockRunMode(str, Enum):
    """Explicit audited branches of ``scrPlayer.Die``."""

    FAIL = "fail"
    NO_FAIL = "no_fail"


class StockTransitionKind(str, Enum):
    HIT_ADVANCE = "hit_advance"
    INVALID_INPUT = "invalid_input"
    AUTO_MISS_FAIL = "auto_miss_fail"
    AUTO_MISS_ADVANCE = "auto_miss_advance"
    IGNORED_AFTER_FAIL = "ignored_after_fail"
    UNSUPPORTED = "unsupported"
    END_OF_CHART = "end_of_chart"


class StockReproducibility(str, Enum):
    REPRODUCED = "reproduced"
    UNSUPPORTED = "unsupported"
    NOT_COMPARABLE = "not_comparable"


_VALID_MARGINS = frozenset(
    {
        StockHitMargin.VERY_EARLY,
        StockHitMargin.EARLY_PERFECT,
        StockHitMargin.PERFECT,
        StockHitMargin.LATE_PERFECT,
        StockHitMargin.VERY_LATE,
    }
)


@dataclass(frozen=True, slots=True)
class StockFloorTarget:
    """Evaluator-side description of the next input-bearing Stock floor."""

    ordinal: int
    floor_index: int
    due_s: float
    approach_bpm: float
    pitch: float = 1.0
    margin_scale: float = 1.0
    skipped_midspin_floors: tuple[int, ...] = ()
    supported: bool = True
    unsupported_reason: str | None = None

    def __post_init__(self) -> None:
        finite = (self.due_s, self.approach_bpm, self.pitch, self.margin_scale)
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("floor target values must be finite")
        if self.approach_bpm <= 0.0 or self.pitch <= 0.0:
            raise ValueError("approach BPM and pitch must be positive")
        if self.margin_scale < 0.0:
            raise ValueError("margin_scale must be non-negative")
        if self.supported == (self.unsupported_reason is not None):
            raise ValueError("supported target/reason contract is inconsistent")


@dataclass(frozen=True, slots=True)
class StockStateTransition:
    input_time_s: float | None
    observed_time_s: float
    kind: StockTransitionKind
    current_floor_before: int
    target_ordinal: int | None
    target_floor: int | None
    judgement: StockHitMargin | str | None
    signed_error_deg: float | None
    current_floor_after: int
    transitioned: bool
    automatic_miss: bool
    failed: bool
    dll_flow: tuple[str, ...]
    reproducibility: StockReproducibility
    note: str | None = None


def targets_from_playable_segment(
    segment: PlayableChartSegment,
) -> tuple[StockFloorTarget, ...]:
    """Build ordinary-floor inputs without treating chart truth as policy data.

    The speed used by ``scrPlanet.SwitchChosen`` belongs to the floor occupied
    before the hit.  Consequently an action on the destination floor (for
    example SetSpeed) must not retroactively change that hit's boundary.
    """

    chart = segment.chart
    pitch = segment.pitch_ratio
    targets: list[StockFloorTarget] = []
    prior_landed_floor = 0
    unsupported_markers = {"Pause", "MultiPlanet", "Checkpoint"}

    for playable in segment.targets:
        destination = chart.floors[playable.floor_index]
        bridge = chart.floors[prior_landed_floor + 1 : playable.floor_index]
        reasons: list[str] = []

        if any(not floor.midspin for floor in bridge):
            reasons.append("non-midspin floor omitted by playable target sequence")
        if any(floor.num_planets != 2 for floor in bridge) or destination.num_planets != 2:
            reasons.append("MultiPlanet branch is not implemented")
        if any(floor.hold is not None for floor in bridge) or destination.hold is not None:
            reasons.append("Hold branch is not implemented")
        if unsupported_markers.intersection(destination.event_markers):
            reasons.append("destination has unsupported special marker")

        approach_floor_index = max(0, playable.floor_index - 1)
        approach = chart.floors[approach_floor_index]
        if approach.hold is not None:
            reasons.append("approach floor is a Hold")
        if approach.pause_s > 0.0 or "Pause" in approach.event_markers:
            reasons.append("approach crosses a Pause")
        if approach.num_planets != 2:
            reasons.append("approach uses MultiPlanet geometry")
        # Mid-floor SetSpeed needs its angleOffset control flow, which the
        # compiled average BPM cannot reproduce safely.
        if "SetSpeed" in approach.event_markers and approach.angle_moved_rad != 0.0:
            reasons.append("approach has unresolved SetSpeed profile")

        targets.append(
            StockFloorTarget(
                ordinal=playable.ordinal,
                floor_index=playable.floor_index,
                due_s=playable.episode_time_s,
                approach_bpm=approach.bpm,
                pitch=pitch,
                skipped_midspin_floors=tuple(floor.index for floor in bridge),
                supported=not reasons,
                unsupported_reason="; ".join(dict.fromkeys(reasons)) or None,
            )
        )
        prior_landed_floor = playable.floor_index
    return tuple(targets)


class StockFloorStateMachine:
    """Small ordinary-floor state machine ported from the audited DLL flow."""

    def __init__(
        self,
        targets: Iterable[StockFloorTarget],
        *,
        run_mode: StockRunMode | str = StockRunMode.FAIL,
        difficulty: StockTimingDifficulty | str = StockTimingDifficulty.NORMAL,
        speed_trial: float = 1.0,
        mobile: bool = False,
        initial_floor: int = 0,
    ) -> None:
        self.targets = tuple(targets)
        if any(b.ordinal <= a.ordinal for a, b in zip(self.targets, self.targets[1:])):
            raise ValueError("target ordinals must be strictly increasing")
        self.run_mode = StockRunMode(run_mode)
        self.difficulty = StockTimingDifficulty(difficulty)
        self.speed_trial = speed_trial
        self.mobile = mobile
        self.initial_floor = initial_floor
        self.reset()

    def reset(self) -> None:
        self.next_index = 0
        self.current_floor = self.initial_floor
        self.failed = False
        self.stopped_reason: str | None = None
        self.invalid_damage_count = 0
        self.trace: list[StockStateTransition] = []

    @property
    def next_target(self) -> StockFloorTarget | None:
        if self.next_index >= len(self.targets):
            return None
        return self.targets[self.next_index]

    def _boundaries(self, target: StockFloorTarget):
        return stock_angle_boundaries(
            target.approach_bpm,
            pitch=target.pitch,
            difficulty=self.difficulty,
            speed_trial=self.speed_trial,
            margin_scale=target.margin_scale,
            mobile=self.mobile,
        )

    @staticmethod
    def _signed_error_deg(now_s: float, target: StockFloorTarget) -> float:
        angular_speed_deg_s = 3.0 * target.approach_bpm * target.pitch
        return (now_s - target.due_s) * angular_speed_deg_s

    def _append(
        self,
        *,
        input_time_s: float | None,
        observed_time_s: float,
        kind: StockTransitionKind,
        target: StockFloorTarget | None,
        judgement: StockHitMargin | str | None,
        signed_error_deg: float | None,
        floor_before: int,
        transitioned: bool,
        automatic_miss: bool,
        dll_flow: tuple[str, ...],
        reproducibility: StockReproducibility = StockReproducibility.REPRODUCED,
        note: str | None = None,
    ) -> StockStateTransition:
        row = StockStateTransition(
            input_time_s=input_time_s,
            observed_time_s=observed_time_s,
            kind=kind,
            current_floor_before=floor_before,
            target_ordinal=None if target is None else target.ordinal,
            target_floor=None if target is None else target.floor_index,
            judgement=judgement,
            signed_error_deg=signed_error_deg,
            current_floor_after=self.current_floor,
            transitioned=transitioned,
            automatic_miss=automatic_miss,
            failed=self.failed,
            dll_flow=dll_flow,
            reproducibility=reproducibility,
            note=note,
        )
        self.trace.append(row)
        return row

    def _unsupported(self, now_s: float, target: StockFloorTarget) -> StockStateTransition:
        self.stopped_reason = target.unsupported_reason
        return self._append(
            input_time_s=None,
            observed_time_s=now_s,
            kind=StockTransitionKind.UNSUPPORTED,
            target=target,
            judgement=None,
            signed_error_deg=None,
            floor_before=self.current_floor,
            transitioned=False,
            automatic_miss=False,
            dll_flow=(),
            reproducibility=StockReproducibility.UNSUPPORTED,
            note=target.unsupported_reason,
        )

    def advance(self, now_s: float) -> tuple[StockStateTransition, ...]:
        """Run ``CheckPostHoldFail`` before input, including its floor loop."""

        if not math.isfinite(now_s):
            raise ValueError("observation time must be finite")
        emitted: list[StockStateTransition] = []
        while not self.failed and self.stopped_reason is None:
            target = self.next_target
            if target is None:
                break
            if not target.supported:
                emitted.append(self._unsupported(now_s, target))
                break
            boundaries = self._boundaries(target)
            error_deg = self._signed_error_deg(now_s, target)
            threshold_deg = (
                boundaries.counted_deg * 1.01
                if self.run_mode is StockRunMode.NO_FAIL
                else max(180.0, boundaries.counted_deg * 2.0)
            )
            # scrPlayer.CheckPostHoldFail uses a strict `>` comparison.
            if not error_deg > threshold_deg:
                break

            floor_before = self.current_floor
            if self.run_mode is StockRunMode.FAIL:
                self.failed = True
                emitted.append(
                    self._append(
                        input_time_s=None,
                        observed_time_s=now_s,
                        kind=StockTransitionKind.AUTO_MISS_FAIL,
                        target=target,
                        judgement="fail_miss",
                        signed_error_deg=error_deg,
                        floor_before=floor_before,
                        transitioned=False,
                        automatic_miss=True,
                        dll_flow=("CheckPostHoldFail", "Die"),
                    )
                )
                break

            self.current_floor = target.floor_index
            self.next_index += 1
            emitted.append(
                self._append(
                    input_time_s=None,
                    observed_time_s=now_s,
                    kind=StockTransitionKind.AUTO_MISS_ADVANCE,
                    target=target,
                    judgement="fail_miss",
                    signed_error_deg=error_deg,
                    floor_before=floor_before,
                    transitioned=True,
                    automatic_miss=True,
                    dll_flow=(
                        "CheckPostHoldFail",
                        "Die",
                        "noFailInfiniteMargin",
                        "Hit",
                        "scrPlanet.SwitchChosen",
                        "MoveToNextFloor",
                    ),
                )
            )
        return tuple(emitted)

    def key_down(self, time_s: float) -> tuple[StockStateTransition, ...]:
        """Process one queued KeyDown after the frame's automatic-Miss check."""

        emitted = list(self.advance(time_s))
        if self.failed:
            emitted.append(
                self._append(
                    input_time_s=time_s,
                    observed_time_s=time_s,
                    kind=StockTransitionKind.IGNORED_AFTER_FAIL,
                    target=self.next_target,
                    judgement=None,
                    signed_error_deg=None,
                    floor_before=self.current_floor,
                    transitioned=False,
                    automatic_miss=False,
                    dll_flow=("Simulated_PlayerControl_Update(alive=false)",),
                )
            )
            return tuple(emitted)
        if self.stopped_reason is not None:
            return tuple(emitted)

        target = self.next_target
        if target is None:
            emitted.append(
                self._append(
                    input_time_s=time_s,
                    observed_time_s=time_s,
                    kind=StockTransitionKind.END_OF_CHART,
                    target=None,
                    judgement=None,
                    signed_error_deg=None,
                    floor_before=self.current_floor,
                    transitioned=False,
                    automatic_miss=False,
                    dll_flow=("UpdateHoldKeys",),
                )
            )
            return tuple(emitted)
        if not target.supported:
            emitted.append(self._unsupported(time_s, target))
            return tuple(emitted)

        floor_before = self.current_floor
        error_deg = self._signed_error_deg(time_s, target)
        judgement = classify_stock_angle(error_deg, self._boundaries(target))
        if judgement in _VALID_MARGINS:
            self.current_floor = target.floor_index
            self.next_index += 1
            self.invalid_damage_count = 0
            emitted.append(
                self._append(
                    input_time_s=time_s,
                    observed_time_s=time_s,
                    kind=StockTransitionKind.HIT_ADVANCE,
                    target=target,
                    judgement=judgement,
                    signed_error_deg=error_deg,
                    floor_before=floor_before,
                    transitioned=True,
                    automatic_miss=False,
                    dll_flow=(
                        "HitAutoFloors",
                        "UpdateHoldKeys",
                        "Hit",
                        "scrPlanet.SwitchChosen",
                        "MoveToNextFloor",
                    ),
                    note=(
                        "midspin bridge auto-consumed by queued Hit"
                        if target.skipped_midspin_floors else None
                    ),
                )
            )
        else:
            self.invalid_damage_count += 1
            reproducibility = StockReproducibility.REPRODUCED
            note = "OnDamage leaves the floor unchanged"
            if self.invalid_damage_count >= 3:
                reproducibility = StockReproducibility.NOT_COMPARABLE
                note = (
                    "fail-bar death depends on Unity Update cadence and decay; "
                    "the ordinary floor non-transition remains reproduced"
                )
            emitted.append(
                self._append(
                    input_time_s=time_s,
                    observed_time_s=time_s,
                    kind=StockTransitionKind.INVALID_INPUT,
                    target=target,
                    judgement=judgement,
                    signed_error_deg=error_deg,
                    floor_before=floor_before,
                    transitioned=False,
                    automatic_miss=False,
                    dll_flow=(
                        "HitAutoFloors",
                        "UpdateHoldKeys",
                        "Hit",
                        "scrPlanet.SwitchChosen",
                        "OnDamage",
                    ),
                    reproducibility=reproducibility,
                    note=note,
                )
            )
        return tuple(emitted)

