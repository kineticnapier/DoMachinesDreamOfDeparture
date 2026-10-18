from __future__ import annotations

"""Versioned, causal stock-countdown observation for HUD v3.

The renderer side may know game-internal countdown state in order to reproduce
what stock ADOFAI draws.  The Student side receives only the text that was
actually present in the current rendered frame.  No countdown setting, future
transition, chart target, or target time crosses that boundary.

The frozen 263D HUD remains an exact prefix.  HUD v3 appends 15 fields:
valid, visible, Ready, Number, Go, and one-hot digits 0..9.
"""

from dataclasses import dataclass
from enum import Enum
from math import floor, isfinite


HUD263_DIM = 263
COUNTDOWN_FEATURE_DIM = 15
HUD_V3_DIM = HUD263_DIM + COUNTDOWN_FEATURE_DIM
HUD_V3_VERSION = "dmdod-visible-hud-v3-stock-countdown-v1"
COUNTDOWN_SENSOR_VERSION = "stock-countdown-text-sensor-v1"


class CountdownCueKind(str, Enum):
    HIDDEN = "hidden"
    READY = "ready"
    NUMBER = "number"
    GO = "go"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RenderedCountdownFrame:
    """Renderer output corresponding to pixels available in one frame.

    This is the sole renderer-to-sensor boundary.  Hidden countdown settings
    and scheduled future transitions are intentionally absent.
    """

    rendered_at_s: float
    displayed_text: str | None
    frame_available: bool = True

    def __post_init__(self) -> None:
        if not isfinite(self.rendered_at_s):
            raise ValueError("rendered_at_s must be finite")
        if self.displayed_text is not None and not isinstance(
            self.displayed_text, str
        ):
            raise TypeError("displayed_text must be text or None")
        if type(self.frame_available) is not bool:
            raise TypeError("frame_available must be bool")
        if not self.frame_available and self.displayed_text is not None:
            raise ValueError("an unavailable frame cannot expose text")


class StockCountdownRenderer:
    """Normal-start DLL schedule reproduced on the renderer side only.

    ``first_tick_s`` and ``tick_period_s`` are render-internal schedule data.
    They never appear in :class:`CountdownStudentObservation`.  Fast takeoff,
    checkpoints, practice variants, and custom extra countdowns are explicitly
    unsupported in phase 1 and render as unavailable instead of being guessed.
    """

    def __init__(self, *, countdown_ticks: int, first_tick_s: float,
                 tick_period_s: float, normal_start: bool = True) -> None:
        if type(countdown_ticks) is not int or countdown_ticks < 0:
            raise ValueError("countdown_ticks must be a nonnegative integer")
        if not isfinite(first_tick_s):
            raise ValueError("first_tick_s must be finite")
        if not isfinite(tick_period_s) or tick_period_s <= 0.0:
            raise ValueError("tick_period_s must be positive and finite")
        if type(normal_start) is not bool:
            raise TypeError("normal_start must be bool")
        self._countdown_ticks = countdown_ticks
        self._first_tick_s = float(first_tick_s)
        self._tick_period_s = float(tick_period_s)
        self._normal_start = normal_start

    def render(self, rendered_at_s: float) -> RenderedCountdownFrame:
        if not isfinite(rendered_at_s):
            raise ValueError("render time must be finite")
        if not self._normal_start or self._countdown_ticks <= 0:
            return RenderedCountdownFrame(
                rendered_at_s, None, frame_available=False
            )
        if rendered_at_s + 1e-12 < self._first_tick_s:
            return RenderedCountdownFrame(rendered_at_s, "Ready!")
        elapsed = rendered_at_s - self._first_tick_s
        tick_index = max(0, floor(elapsed / self._tick_period_s + 1e-12))
        if tick_index < self._countdown_ticks - 1:
            return RenderedCountdownFrame(
                rendered_at_s, str(self._countdown_ticks - 1 - tick_index)
            )
        return RenderedCountdownFrame(rendered_at_s, "Go!")


@dataclass(frozen=True, slots=True)
class CountdownStudentObservation:
    """Only a causally observed current-frame cue, never its hidden cause."""

    step: int
    observed_at_s: float
    rendered_at_s: float
    valid: bool
    kind: CountdownCueKind
    digit: int | None = None
    sensor_version: str = COUNTDOWN_SENSOR_VERSION

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0:
            raise ValueError("step must be a nonnegative integer")
        if not isfinite(self.observed_at_s) or not isfinite(self.rendered_at_s):
            raise ValueError("observation times must be finite")
        if self.rendered_at_s > self.observed_at_s + 1e-12:
            raise ValueError("future rendered frames are not causal observations")
        if type(self.valid) is not bool:
            raise TypeError("valid must be bool")
        if not isinstance(self.kind, CountdownCueKind):
            raise TypeError("kind must be CountdownCueKind")
        if self.sensor_version != COUNTDOWN_SENSOR_VERSION:
            raise ValueError("unknown countdown sensor version")
        if self.kind is CountdownCueKind.NUMBER:
            if type(self.digit) is not int or not 0 <= self.digit <= 9:
                raise ValueError("numeric cue requires displayed digit 0..9")
        elif self.digit is not None:
            raise ValueError("non-numeric cue cannot contain a digit")
        if not self.valid and self.kind is not CountdownCueKind.UNKNOWN:
            raise ValueError("invalid observation must be explicitly unknown")

    @property
    def latency_s(self) -> float:
        return self.observed_at_s - self.rendered_at_s


class CountdownVisualSensor:
    """Sequential recognizer over current rendered frames only."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._next_step = 0
        self._last_observed_at_s: float | None = None

    def observe(self, *, step: int, observed_at_s: float,
                frame: RenderedCountdownFrame) -> CountdownStudentObservation:
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

        text = frame.displayed_text
        if not frame.frame_available:
            valid, kind, digit = False, CountdownCueKind.UNKNOWN, None
        elif text is None or not text.strip():
            valid, kind, digit = True, CountdownCueKind.HIDDEN, None
        elif text.strip() == "Ready!":
            valid, kind, digit = True, CountdownCueKind.READY, None
        elif text.strip() == "Go!":
            valid, kind, digit = True, CountdownCueKind.GO, None
        elif len(text.strip()) == 1 and text.strip().isdigit():
            valid, kind, digit = True, CountdownCueKind.NUMBER, int(text.strip())
        else:
            # Unrecognized/localized/occluded content is not guessed.
            valid, kind, digit = False, CountdownCueKind.UNKNOWN, None

        result = CountdownStudentObservation(
            step=step,
            observed_at_s=float(observed_at_s),
            rendered_at_s=float(frame.rendered_at_s),
            valid=valid,
            kind=kind,
            digit=digit,
        )
        self._next_step += 1
        self._last_observed_at_s = float(observed_at_s)
        return result


def encode_countdown_features(
    observation: CountdownStudentObservation,
) -> tuple[float, ...]:
    """Encode observed display semantics; absent/unknown stays explicit."""
    if not isinstance(observation, CountdownStudentObservation):
        raise TypeError("expected CountdownStudentObservation")
    visible = observation.kind in {
        CountdownCueKind.READY,
        CountdownCueKind.NUMBER,
        CountdownCueKind.GO,
    }
    values = [
        float(observation.valid),
        float(visible),
        float(observation.kind is CountdownCueKind.READY),
        float(observation.kind is CountdownCueKind.NUMBER),
        float(observation.kind is CountdownCueKind.GO),
    ]
    values.extend(
        float(observation.kind is CountdownCueKind.NUMBER
              and observation.digit == digit)
        for digit in range(10)
    )
    if len(values) != COUNTDOWN_FEATURE_DIM:
        raise RuntimeError("countdown feature dimension drift")
    return tuple(values)


@dataclass(frozen=True, slots=True)
class CountdownHUDV3Frame:
    """Causal Student view. The first 263 values are byte-for-byte legacy."""

    step: int
    observed_at_s: float
    features: tuple[float, ...]
    countdown_observation_valid: bool
    version: str = HUD_V3_VERSION

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0:
            raise ValueError("step must be a nonnegative integer")
        if not isfinite(self.observed_at_s):
            raise ValueError("observed_at_s must be finite")
        if len(self.features) != HUD_V3_DIM or not all(
            isfinite(value) for value in self.features
        ):
            raise ValueError("HUD v3 must contain 278 finite values")
        if type(self.countdown_observation_valid) is not bool:
            raise TypeError("countdown_observation_valid must be bool")
        if self.version != HUD_V3_VERSION:
            raise ValueError("unknown HUD v3 version")

    @property
    def legacy_hud263(self) -> tuple[float, ...]:
        return self.features[:HUD263_DIM]


def encode_hud_v3(
    legacy_hud263: tuple[float, ...],
    countdown: CountdownStudentObservation,
) -> CountdownHUDV3Frame:
    if len(legacy_hud263) != HUD263_DIM or not all(
        isfinite(value) for value in legacy_hud263
    ):
        raise ValueError("legacy HUD must contain exactly 263 finite values")
    if not isinstance(countdown, CountdownStudentObservation):
        raise TypeError("expected causal countdown observation")
    extra = encode_countdown_features(countdown)
    return CountdownHUDV3Frame(
        step=countdown.step,
        observed_at_s=countdown.observed_at_s,
        features=tuple(legacy_hud263) + extra,
        countdown_observation_valid=countdown.valid,
    )
