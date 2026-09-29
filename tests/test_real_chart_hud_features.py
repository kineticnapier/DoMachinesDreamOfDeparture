from __future__ import annotations

from math import pi

import pytest

from dmdod.adofai_rules import TimingJudgement
from dmdod.motor_env import MotorObservation
from dmdod.real_chart_env import RelativeVisibleFloor
from dmdod.real_chart_features import REAL_CHART_INPUT_DIM
from dmdod.real_chart_hud import HudRealChartObservation
from dmdod.real_chart_hud_features import (
    HUD_FEATURE_DIM,
    HUD_JUDGEMENTS,
    HUD_REAL_CHART_INPUT_DIM,
    encode_bpm,
    encode_hud_real_chart_observation,
)


def _observation(*, visible: bool = True) -> HudRealChartObservation:
    motor = MotorObservation(
        left_position_m=0.003,
        right_position_m=0.006,
        left_velocity_m_s=0.25,
        right_velocity_m_s=-0.5,
        left_pressed=True,
        right_pressed=False,
    )
    floor = RelativeVisibleFloor(
        relative_index=0,
        x=0.0,
        y=0.0,
        entry_angle_rad=pi / 2,
        exit_angle_rad=pi,
        midspin=False,
        is_ccw=False,
        num_planets=2,
        event_markers=(),
    )
    return HudRealChartObservation(
        motor=motor,
        orbiting_x=1.5,
        orbiting_y=0.0,
        floors=(floor,),
        tile_bpm=120.0,
        real_bpm=240.0,
        feedback_visible=visible,
        last_judgement=TimingJudgement.EARLY_PERFECT if visible else None,
        last_timing_error_ms=-25.0 if visible else 0.0,
    )


def test_hud_encoder_appends_features_without_changing_old_prefix():
    values = encode_hud_real_chart_observation(_observation())
    assert len(values) == HUD_REAL_CHART_INPUT_DIM
    assert HUD_REAL_CHART_INPUT_DIM == REAL_CHART_INPUT_DIM + HUD_FEATURE_DIM
    assert REAL_CHART_INPUT_DIM == 233

    hud = values[REAL_CHART_INPUT_DIM:]
    assert hud[0] == pytest.approx(0.0)  # 120 BPM reference
    assert hud[1] == pytest.approx(1.0)  # 240 BPM = one octave faster
    assert hud[2] == 1.0
    assert hud[3] == pytest.approx(-0.25)

    one_hot = hud[4:]
    assert sum(one_hot) == 1.0
    assert one_hot[HUD_JUDGEMENTS.index(TimingJudgement.EARLY_PERFECT)] == 1.0


def test_hidden_feedback_is_zeroed_but_bpm_remains_visible():
    values = encode_hud_real_chart_observation(_observation(visible=False))
    hud = values[REAL_CHART_INPUT_DIM:]
    assert hud[:2] == pytest.approx((0.0, 1.0))
    assert hud[2:] == (0.0,) * (HUD_FEATURE_DIM - 2)


def test_bpm_encoding_preserves_double_speed_relationship():
    assert encode_bpm(60.0) == pytest.approx(-1.0)
    assert encode_bpm(120.0) == pytest.approx(0.0)
    assert encode_bpm(240.0) == pytest.approx(1.0)
    assert encode_bpm(480.0) == pytest.approx(2.0)
