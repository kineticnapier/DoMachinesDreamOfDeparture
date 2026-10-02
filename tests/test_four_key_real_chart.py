from __future__ import annotations

from math import pi

import pytest

from dmdod.adofai_chart import parse_adofai_text
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_rules import TimingJudgement
from dmdod.adofai_timing import compile_adofai
from dmdod.four_key_motor import FourKeyObservation
from dmdod.four_key_real_chart import (
    FOUR_KEY_HUD_REAL_CHART_INPUT_DIM,
    FOUR_KEY_REAL_CHART_INPUT_DIM,
    DiagnosticHudFourKeyRealChartMotorEnv,
    FourKeyHudRealChartObservation,
    FourKeyRealChartMotorEnv,
    FourKeyRealChartObservation,
    encode_four_key_hud_real_chart_observation,
    encode_four_key_real_chart_observation,
)
from dmdod.keyboard import KeyEvent
from dmdod.motor_env import TimedKeyEvent
from dmdod.real_chart_env import RelativeVisibleFloor
from dmdod.real_chart_hud_features import HUD_FEATURE_DIM


def _compiled():
    chart = parse_adofai_text(
        """
        {
          "angleData": [0, 999, 0],
          "settings": {
            "bpm": 120,
            "pitch": 100,
            "countdownTicks": 0,
            "separateCountdownTime": false
          },
          "actions": [
            {"floor": 2, "eventType": "Twirl"}
          ]
        }
        """
    )
    return compile_adofai(chart)


def _motor() -> FourKeyObservation:
    return FourKeyObservation(
        left_outer_position_m=0.001,
        left_inner_position_m=0.002,
        right_inner_position_m=0.003,
        right_outer_position_m=0.004,
        left_outer_velocity_m_s=0.1,
        left_inner_velocity_m_s=0.2,
        right_inner_velocity_m_s=-0.3,
        right_outer_velocity_m_s=-0.4,
        left_outer_pressed=False,
        left_inner_pressed=True,
        right_inner_pressed=True,
        right_outer_pressed=False,
    )


def _floor() -> RelativeVisibleFloor:
    return RelativeVisibleFloor(
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


def test_four_key_level_a_encoder_is_239d_and_preserves_physical_order() -> None:
    observation = FourKeyRealChartObservation(
        motor=_motor(),
        orbiting_x=1.5,
        orbiting_y=0.0,
        floors=(_floor(),),
    )

    values = encode_four_key_real_chart_observation(observation)

    assert FOUR_KEY_REAL_CHART_INPUT_DIM == 239
    assert len(values) == FOUR_KEY_REAL_CHART_INPUT_DIM
    assert values[:4] == pytest.approx((1 / 6, 2 / 6, 3 / 6, 4 / 6))
    assert values[4:8] == pytest.approx((0.1, 0.2, -0.3, -0.4))
    assert values[8:12] == (0.0, 1.0, 1.0, 0.0)


def test_four_key_hud_encoder_grows_level_a_from_245d_to_251d() -> None:
    observation = FourKeyHudRealChartObservation(
        motor=_motor(),
        orbiting_x=1.5,
        orbiting_y=0.0,
        floors=(_floor(),),
        tile_bpm=120.0,
        real_bpm=240.0,
        feedback_visible=True,
        last_judgement=TimingJudgement.EARLY_PERFECT,
        last_timing_error_ms=-25.0,
    )

    values = encode_four_key_hud_real_chart_observation(observation)

    assert FOUR_KEY_HUD_REAL_CHART_INPUT_DIM == 251
    assert FOUR_KEY_HUD_REAL_CHART_INPUT_DIM == FOUR_KEY_REAL_CHART_INPUT_DIM + HUD_FEATURE_DIM
    assert len(values) == FOUR_KEY_HUD_REAL_CHART_INPUT_DIM
    hud = values[FOUR_KEY_REAL_CHART_INPUT_DIM:]
    assert hud[0] == pytest.approx(0.0)
    assert hud[1] == pytest.approx(1.0)
    assert hud[2] == 1.0
    assert hud[3] == pytest.approx(-0.25)


def test_four_key_real_chart_any_keydown_can_claim_next_target() -> None:
    segment = build_playable_segment(_compiled(), start_s=0.0, end_s=2.0)
    env = FourKeyRealChartMotorEnv(segment)
    observation = env.reset()
    assert isinstance(observation.motor, FourKeyObservation)
    target = env.privileged_next_target()
    assert target is not None

    reward = env._score_event(
        TimedKeyEvent(target.episode_time_s, "right_outer", KeyEvent.DOWN)
    )

    assert reward > 0.0
    assert env.stats.hits == 1
    assert env.stats.misses == 0


def test_four_key_diagnostic_hud_counts_all_physical_keydowns() -> None:
    segment = build_playable_segment(_compiled(), start_s=0.0, end_s=2.0)
    env = DiagnosticHudFourKeyRealChartMotorEnv(segment)
    observation = env.reset()
    assert len(encode_four_key_hud_real_chart_observation(observation)) == 251
    target = env.privileged_next_target()
    assert target is not None

    env._score_event(TimedKeyEvent(target.episode_time_s, "left_inner", KeyEvent.DOWN))

    assert env.physical_keydowns == 1
    assert env.stats.hits == 1
