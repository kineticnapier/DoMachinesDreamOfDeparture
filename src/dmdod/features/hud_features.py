from __future__ import annotations

from math import log2

from dmdod.adofai_rules import TimingJudgement
from dmdod.envs.real_chart import RealChartObservation
from dmdod.features.real_chart import REAL_CHART_INPUT_DIM, encode_real_chart_observation
from dmdod.features.hud import HudRealChartObservation


HUD_FEATURE_VERSION = "tbpm-rbpm-feedback-v1"
HUD_BPM_REFERENCE = 120.0
HUD_ERROR_SCALE_MS = 100.0
HUD_BPM_LOG_CLIP = 4.0
HUD_ERROR_CLIP = 2.0

HUD_JUDGEMENTS = (
    TimingJudgement.PERFECT,
    TimingJudgement.EARLY_PERFECT,
    TimingJudgement.LATE_PERFECT,
    TimingJudgement.VERY_EARLY,
    TimingJudgement.VERY_LATE,
    TimingJudgement.TOO_EARLY,
    TimingJudgement.TOO_LATE,
    TimingJudgement.FAIL_OVERLOAD,
)
HUD_FEATURE_DIM = 2 + 1 + 1 + len(HUD_JUDGEMENTS)
HUD_REAL_CHART_INPUT_DIM = REAL_CHART_INPUT_DIM + HUD_FEATURE_DIM


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def encode_bpm(bpm: float) -> float:
    """Log-scale BPM so octave/double-speed relationships stay simple."""

    safe_bpm = max(1e-6, float(bpm))
    return _clip(log2(safe_bpm / HUD_BPM_REFERENCE), -HUD_BPM_LOG_CLIP, HUD_BPM_LOG_CLIP)


def encode_hud_real_chart_observation(observation: HudRealChartObservation) -> tuple[float, ...]:
    """Encode geometry/motor state plus the small human-visible HUD slice.

    The first 233 values are byte-for-layout compatible with the v0.6.x
    visible-only encoder. New HUD values are appended so older representation
    semantics stay easy to compare:

    - Tile BPM, Real BPM (log2 relative to 120 BPM)
    - feedback-visible bit
    - signed timing error / 100 ms, clipped to [-2, 2]
    - one-hot latest judgement while the feedback is visible

    XAccuracy, chart progress, attempt count, KV, internal overload, exact target
    time, and simulator time are intentionally absent.
    """

    base = RealChartObservation(
        motor=observation.motor,
        orbiting_x=observation.orbiting_x,
        orbiting_y=observation.orbiting_y,
        floors=observation.floors,
    )
    values = list(encode_real_chart_observation(base))
    visible = bool(observation.feedback_visible and observation.last_judgement is not None)
    values.extend(
        (
            encode_bpm(observation.tile_bpm),
            encode_bpm(observation.real_bpm),
            1.0 if visible else 0.0,
            _clip(observation.last_timing_error_ms / HUD_ERROR_SCALE_MS, -HUD_ERROR_CLIP, HUD_ERROR_CLIP)
            if visible
            else 0.0,
        )
    )
    values.extend(
        1.0 if visible and observation.last_judgement is judgement else 0.0
        for judgement in HUD_JUDGEMENTS
    )
    if len(values) != HUD_REAL_CHART_INPUT_DIM:
        raise RuntimeError(
            f"HUD real-chart feature size mismatch: expected {HUD_REAL_CHART_INPUT_DIM}, got {len(values)}"
        )
    return tuple(values)
