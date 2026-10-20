from __future__ import annotations

"""Causal, versioned observation of stock Hold icons.

The renderer may use game-internal Hold state to reproduce pixels.  The visual
sensor receives only icon labels already present in one rendered frame.  In
particular, Hold duration and release time never cross the renderer/sensor
boundary.

This contract intentionally models only the stock Short/Long start/release
sprites confirmed in the audited game code.  It does not claim that the
integer loop count can be recovered from pixels.  A missing icon means only
that no Hold icon was visible; another event icon may have occluded it.
"""

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from dmdod.features.countdown_hud_v3 import (
    HUD263_DIM,
    HUD_V3_DIM,
    CountdownHUDV3Frame,
)


HOLD_BEHIND_FLOORS = 2
HOLD_AHEAD_FLOORS = 12
HOLD_FLOOR_SLOTS = HOLD_BEHIND_FLOORS + HOLD_AHEAD_FLOORS + 1
HOLD_ICON_FEATURE_DIM = 6
HOLD_FEATURE_DIM = HOLD_FLOOR_SLOTS * HOLD_ICON_FEATURE_DIM
HUD_V4_DIM = HUD_V3_DIM + HOLD_FEATURE_DIM
HUD_V4_VERSION = "dmdod-visible-hud-v4-stock-hold-icons-v1"
HOLD_ICON_SENSOR_VERSION = "stock-hold-icon-sensor-v1"


class HoldCueKind(str, Enum):
    NO_HOLD_ICON_VISIBLE = "no_hold_icon_visible"
    START_SHORT = "start_short"
    START_LONG = "start_long"
    RELEASE_SHORT = "release_short"
    RELEASE_LONG = "release_long"
    UNKNOWN = "unknown"


_SPRITE_TO_CUE = {
    "HoldArrowShort": HoldCueKind.START_SHORT,
    "HoldArrowLong": HoldCueKind.START_LONG,
    "HoldReleaseShort": HoldCueKind.RELEASE_SHORT,
    "HoldReleaseLong": HoldCueKind.RELEASE_LONG,
}


@dataclass(frozen=True, slots=True)
class RenderedHoldIconFrame:
    """Renderer output containing only currently visible sprite identities."""

    rendered_at_s: float
    icons: tuple[tuple[int, str], ...]
    frame_available: bool = True

    def __post_init__(self) -> None:
        if not isfinite(self.rendered_at_s):
            raise ValueError("rendered_at_s must be finite")
        if type(self.frame_available) is not bool:
            raise TypeError("frame_available must be bool")
        if not self.frame_available and self.icons:
            raise ValueError("an unavailable frame cannot expose icons")
        seen: set[int] = set()
        for relative_index, sprite_name in self.icons:
            if type(relative_index) is not int or not (
                -HOLD_BEHIND_FLOORS <= relative_index <= HOLD_AHEAD_FLOORS
            ):
                raise ValueError("icon relative index is outside visible slots")
            if relative_index in seen:
                raise ValueError("a visible floor slot can expose only one icon")
            if not isinstance(sprite_name, str) or not sprite_name:
                raise TypeError("sprite name must be non-empty text")
            seen.add(relative_index)

    def icon_at(self, relative_index: int) -> str | None:
        return dict(self.icons).get(relative_index)


class StockHoldIconRenderer:
    """Minimal code-parity renderer for confirmed stock Hold icon selection.

    The caller-facing arguments are renderer-side truth and must never be
    forwarded to the Student.  Positive durations deliberately collapse to
    the same Long icon class, matching ``scnGame``.
    """

    @staticmethod
    def render(
        *,
        rendered_at_s: float,
        start_relative_index: int,
        duration: int,
        occluded_relative_indices: tuple[int, ...] = (),
    ) -> RenderedHoldIconFrame:
        if not isfinite(rendered_at_s):
            raise ValueError("rendered_at_s must be finite")
        if type(start_relative_index) is not int:
            raise TypeError("start_relative_index must be int")
        if not -HOLD_BEHIND_FLOORS <= start_relative_index <= HOLD_AHEAD_FLOORS:
            raise ValueError("start floor is outside visible slots")
        if start_relative_index + 1 > HOLD_AHEAD_FLOORS:
            raise ValueError("a stock Hold requires a visible following floor")
        if type(duration) is not int:
            raise TypeError("duration must be an integer")
        if duration < 0:
            raise ValueError("duration must be nonnegative")
        occluded = frozenset(occluded_relative_indices)
        if any(
            type(index) is not int
            or not -HOLD_BEHIND_FLOORS <= index <= HOLD_AHEAD_FLOORS
            for index in occluded
        ):
            raise ValueError("occluded index is outside visible slots")
        suffix = "Short" if duration == 0 else "Long"
        icons = tuple(
            (index, sprite)
            for index, sprite in (
                (start_relative_index, f"HoldArrow{suffix}"),
                (start_relative_index + 1, f"HoldRelease{suffix}"),
            )
            if index not in occluded
        )
        return RenderedHoldIconFrame(float(rendered_at_s), icons)


@dataclass(frozen=True, slots=True)
class HoldFloorCue:
    relative_index: int
    valid: bool
    kind: HoldCueKind

    def __post_init__(self) -> None:
        if type(self.relative_index) is not int or not (
            -HOLD_BEHIND_FLOORS <= self.relative_index <= HOLD_AHEAD_FLOORS
        ):
            raise ValueError("relative_index is outside visible slots")
        if type(self.valid) is not bool:
            raise TypeError("valid must be bool")
        if not isinstance(self.kind, HoldCueKind):
            raise TypeError("kind must be HoldCueKind")
        if not self.valid and self.kind is not HoldCueKind.UNKNOWN:
            raise ValueError("invalid cues must be explicitly unknown")


@dataclass(frozen=True, slots=True)
class HoldIconStudentObservation:
    step: int
    observed_at_s: float
    rendered_at_s: float
    cues: tuple[HoldFloorCue, ...]
    sensor_version: str = HOLD_ICON_SENSOR_VERSION

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0:
            raise ValueError("step must be a nonnegative integer")
        if not isfinite(self.observed_at_s) or not isfinite(self.rendered_at_s):
            raise ValueError("observation times must be finite")
        if self.rendered_at_s > self.observed_at_s + 1e-12:
            raise ValueError("future rendered frames are not causal observations")
        if self.sensor_version != HOLD_ICON_SENSOR_VERSION:
            raise ValueError("unknown Hold icon sensor version")
        expected = tuple(range(-HOLD_BEHIND_FLOORS, HOLD_AHEAD_FLOORS + 1))
        if tuple(cue.relative_index for cue in self.cues) != expected:
            raise ValueError("Hold cues must cover all visible slots in order")

    @property
    def latency_s(self) -> float:
        return self.observed_at_s - self.rendered_at_s

    def cue_at(self, relative_index: int) -> HoldFloorCue:
        return self.cues[relative_index + HOLD_BEHIND_FLOORS]


class HoldIconVisualSensor:
    """Sequential recognizer that consumes current rendered icons only."""

    def __init__(self, *, max_frame_age_s: float = 0.050) -> None:
        if not isfinite(max_frame_age_s) or max_frame_age_s < 0.0:
            raise ValueError("max_frame_age_s must be nonnegative and finite")
        self._max_frame_age_s = float(max_frame_age_s)
        self.reset()

    def reset(self) -> None:
        self._next_step = 0
        self._last_observed_at_s: float | None = None

    def observe(
        self,
        *,
        step: int,
        observed_at_s: float,
        frame: RenderedHoldIconFrame,
    ) -> HoldIconStudentObservation:
        if type(step) is not int or step != self._next_step:
            raise ValueError("sensor steps must be consecutive, starting at zero")
        if not isfinite(observed_at_s):
            raise ValueError("observed_at_s must be finite")
        if self._last_observed_at_s is not None and (
            observed_at_s < self._last_observed_at_s - 1e-12
        ):
            raise ValueError("observation time cannot move backward")
        if frame.rendered_at_s > observed_at_s + 1e-12:
            raise ValueError("sensor cannot consume a future rendered frame")

        unavailable = (
            not frame.frame_available
            or observed_at_s - frame.rendered_at_s > self._max_frame_age_s + 1e-12
        )
        sprites = dict(frame.icons)
        cues: list[HoldFloorCue] = []
        for relative_index in range(-HOLD_BEHIND_FLOORS, HOLD_AHEAD_FLOORS + 1):
            if unavailable:
                valid, kind = False, HoldCueKind.UNKNOWN
            elif relative_index not in sprites:
                valid, kind = True, HoldCueKind.NO_HOLD_ICON_VISIBLE
            else:
                kind = _SPRITE_TO_CUE.get(sprites[relative_index], HoldCueKind.UNKNOWN)
                valid = kind is not HoldCueKind.UNKNOWN
            cues.append(HoldFloorCue(relative_index, valid, kind))

        result = HoldIconStudentObservation(
            step=step,
            observed_at_s=float(observed_at_s),
            rendered_at_s=float(frame.rendered_at_s),
            cues=tuple(cues),
        )
        self._next_step += 1
        self._last_observed_at_s = float(observed_at_s)
        return result


def encode_hold_icon_features(
    observation: HoldIconStudentObservation,
) -> tuple[float, ...]:
    values: list[float] = []
    for cue in observation.cues:
        visible = cue.kind not in {
            HoldCueKind.NO_HOLD_ICON_VISIBLE,
            HoldCueKind.UNKNOWN,
        }
        values.extend(
            (
                float(cue.valid),
                float(visible),
                float(cue.kind in {HoldCueKind.START_SHORT, HoldCueKind.START_LONG}),
                float(
                    cue.kind
                    in {HoldCueKind.RELEASE_SHORT, HoldCueKind.RELEASE_LONG}
                ),
                float(
                    cue.kind
                    in {HoldCueKind.START_SHORT, HoldCueKind.RELEASE_SHORT}
                ),
                float(
                    cue.kind in {HoldCueKind.START_LONG, HoldCueKind.RELEASE_LONG}
                ),
            )
        )
    if len(values) != HOLD_FEATURE_DIM:
        raise RuntimeError("Hold icon feature dimension drift")
    return tuple(values)


@dataclass(frozen=True, slots=True)
class HoldHUDV4Frame:
    step: int
    observed_at_s: float
    features: tuple[float, ...]
    version: str = HUD_V4_VERSION

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0:
            raise ValueError("step must be a nonnegative integer")
        if not isfinite(self.observed_at_s):
            raise ValueError("observed_at_s must be finite")
        if len(self.features) != HUD_V4_DIM or not all(
            isfinite(value) for value in self.features
        ):
            raise ValueError(f"HUD v4 must contain {HUD_V4_DIM} finite values")
        if self.version != HUD_V4_VERSION:
            raise ValueError("unknown HUD v4 version")

    @property
    def legacy_hud263(self) -> tuple[float, ...]:
        return self.features[:HUD263_DIM]

    @property
    def countdown_hud278(self) -> tuple[float, ...]:
        return self.features[:HUD_V3_DIM]


def encode_hud_v4(
    countdown_hud_v3: CountdownHUDV3Frame,
    hold_icons: HoldIconStudentObservation,
) -> HoldHUDV4Frame:
    if not isinstance(countdown_hud_v3, CountdownHUDV3Frame):
        raise TypeError("expected CountdownHUDV3Frame")
    if not isinstance(hold_icons, HoldIconStudentObservation):
        raise TypeError("expected HoldIconStudentObservation")
    if countdown_hud_v3.step != hold_icons.step:
        raise ValueError("countdown and Hold observations must share a step")
    if abs(countdown_hud_v3.observed_at_s - hold_icons.observed_at_s) > 1e-12:
        raise ValueError("countdown and Hold observations must share a timestamp")
    return HoldHUDV4Frame(
        step=hold_icons.step,
        observed_at_s=hold_icons.observed_at_s,
        features=countdown_hud_v3.features + encode_hold_icon_features(hold_icons),
    )
