from __future__ import annotations

"""Strict, versioned Contact (12-axis) -> HUD263 adapter for Timing v1.4.

Only visible orbit/floor geometry and caller-supplied *displayed* BPM/HUD
feedback enter this encoder. The Motor's 12 axes are NEVER reduced to a fake
8-key Motor observation: the legacy first 24 numbers are explicitly zero.

The contact environment currently exposes geometry but not BPM; the caller
must supply the visible Tile/Real BPM from a future contact-HUD observation
provider. Do not consult env.segment, chart clocks, targets, or the evaluator.

This adapter MUST NOT be passed to frozen neural 8-key MLP/GRU/v3 models.
"""

from dataclasses import dataclass
from math import isfinite

from dmdod.adofai_rules import TimingJudgement
from dmdod.envs.contact_chart_16k import ContactChartObservation
from dmdod.envs.n_key import (
    N_KEY_HUD_FEATURE_VERSION, NKeyHudRealChartObservation,
    encode_n_key_hud_real_chart_observation, n_key_hud_real_chart_input_dim,
)
from dmdod.features.real_chart import (
    DEFAULT_REAL_CHART_FEATURE_CONFIG, REAL_CHART_FLOOR_FEATURE_DIM,
)
from dmdod.motor.n_key import NKeyObservation, n_key_names
from dmdod.planning.hud_timing_tracker_transition_v14 import (
    CausalTimingTracker, TimingUpdate,
)
from dmdod.planning import hud_timing_tracker_transition_v14 as v14


CONTACT_TIMING_HUD_VERSION = "contact-12axis-timing-v1.4-hud263-v1"
SUPPORTED_ESTIMATOR = "hud_timing_tracker_transition_v14"
HUD_DIM = 263
LEGACY_MOTOR_DIM = 24
NEUTRAL_MOTOR_FIELDS = (0.0,) * LEGACY_MOTOR_DIM


@dataclass(frozen=True, slots=True)
class ContactVisibleTimingHUD:
    """Geometry from ContactChartObservation, BPM from CURRENT visible HUD.

    The public contact geometry observation has no BPM fields. Tile/Real BPM
    are therefore required rather than reconstructed from future true notes.
    Timing feedback is optional and must be displayed to be included.
    """

    contact: ContactChartObservation
    tile_bpm: float
    real_bpm: float
    feedback_visible: bool = False
    last_judgement: TimingJudgement | None = None
    last_timing_error_ms: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.contact, ContactChartObservation):
            raise TypeError("expected public ContactChartObservation")
        for name in ("tile_bpm", "real_bpm"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite visible BPM")
        if type(self.feedback_visible) is not bool:
            raise ValueError("feedback_visible must be bool")
        if self.last_judgement is not None and not isinstance(
            self.last_judgement, TimingJudgement
        ):
            raise ValueError("last_judgement must be a TimingJudgement or None")
        if not isfinite(self.last_timing_error_ms):
            raise ValueError("HUD timing error must be finite")
        if self.feedback_visible and self.last_judgement is None:
            raise ValueError("visible feedback requires its visible judgement")


@dataclass(frozen=True, slots=True)
class ContactTimingHUD263Frame:
    """Only Timing v1.4 may consume these 263 features."""

    step: int
    features: tuple[float, ...]
    version: str = CONTACT_TIMING_HUD_VERSION

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0:
            raise ValueError("step must be a nonnegative local control step")
        if self.version != CONTACT_TIMING_HUD_VERSION:
            raise ValueError("unknown Contact Timing HUD feature version")
        if len(self.features) != HUD_DIM or not all(isfinite(v) for v in self.features):
            raise ValueError("Contact Timing HUD must be 263 finite values")
        if self.features[:LEGACY_MOTOR_DIM] != NEUTRAL_MOTOR_FIELDS:
            raise ValueError("legacy Motor slice must be neutral")


def assert_v14_compatibility() -> None:
    """Fail closed on a changed HUD layout or accidental v1 BPM encoding."""
    config = DEFAULT_REAL_CHART_FEATURE_CONFIG
    if not (
        v14.HUD_DIM == HUD_DIM
        and v14.MOTOR_FEATURES == LEGACY_MOTOR_DIM
        and v14.SLOT_START == LEGACY_MOTOR_DIM + 2
        and v14.FLOOR_DIM == REAL_CHART_FLOOR_FEATURE_DIM
        and v14.HUD_OFFSET == (LEGACY_MOTOR_DIM + 2
                              + config.floor_slots * REAL_CHART_FLOOR_FEATURE_DIM)
        and n_key_hud_real_chart_input_dim(8) == HUD_DIM
        and N_KEY_HUD_FEATURE_VERSION == "n-key-tbpm-rbpm-feedback-v2-unclipped"
    ):
        raise RuntimeError("Timing v1.4 / 263D geometry-BPM layout drift")


def _neutral_legacy_motor() -> NKeyObservation:
    """Neutral placeholder, NOT a projection from any 12-axis Motor state."""
    return NKeyObservation(
        key_names=n_key_names(8),
        positions_m=(0.0,) * 8,
        velocities_m_s=(0.0,) * 8,
        pressed_flags=(False,) * 8,
    )


class ContactTimingHUD263Adapter:
    """Codec and sequential v1.4 observation stream; no neural model support.

    The tracker uses the HUD orbit/geometry/BPM, not the neutral Motor slice.
    The v1.4 forecast's finite-input check sees all 263 numbers, but changing
    the first 24 *finite* values does not alter its Timing calculations.
    """

    def __init__(self, *, estimator: str = SUPPORTED_ESTIMATOR,
                 horizon_s: float = .030) -> None:
        if estimator != SUPPORTED_ESTIMATOR:
            raise ValueError(
                "Contact HUD263 adapter supports deterministic Timing v1.4 ONLY; "
                "MLP, GRU and Neural v3 require a separate versioned 12-axis input"
            )
        assert_v14_compatibility()
        self._tracker = CausalTimingTracker(horizon_s=horizon_s)
        self._next_step = 0

    def reset(self) -> None:
        self._tracker.reset()
        self._next_step = 0

    @staticmethod
    def encode(step: int, observation: ContactVisibleTimingHUD) -> ContactTimingHUD263Frame:
        assert_v14_compatibility()
        if not isinstance(observation, ContactVisibleTimingHUD):
            raise TypeError("expected ContactVisibleTimingHUD with explicitly visible BPM")
        contact = observation.contact
        fake_8key_hud = NKeyHudRealChartObservation(
            motor=_neutral_legacy_motor(),
            orbiting_x=contact.orbiting_x,
            orbiting_y=contact.orbiting_y,
            floors=contact.floors,
            tile_bpm=observation.tile_bpm,
            real_bpm=observation.real_bpm,
            feedback_visible=observation.feedback_visible,
            last_judgement=observation.last_judgement,
            last_timing_error_ms=observation.last_timing_error_ms,
        )
        features = encode_n_key_hud_real_chart_observation(
            fake_8key_hud, legacy_bpm_encoding=False
        )
        return ContactTimingHUD263Frame(step=step, features=features)

    def observe(self, frame: ContactTimingHUD263Frame) -> TimingUpdate:
        """Advance deterministic v1.4 only; adapting its output is a next phase."""
        if not isinstance(frame, ContactTimingHUD263Frame):
            raise TypeError("expected explicitly versioned ContactTimingHUD263Frame")
        if frame.step != self._next_step:
            raise ValueError("Timing HUD frames must be consecutive, starting at zero")
        update = self._tracker.observe(frame.step, frame.features)
        self._next_step += 1
        return update
