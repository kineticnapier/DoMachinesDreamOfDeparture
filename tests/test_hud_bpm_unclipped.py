from __future__ import annotations

from math import inf, isfinite, log2, nan

import pytest

from dmdod.envs.n_key import (
    NKeyHudRealChartObservation,
    N_KEY_HUD_FEATURE_VERSION,
    encode_n_key_hud_real_chart_observation,
    n_key_hud_real_chart_input_dim,
)
from dmdod.features.hud import HudRealChartObservation
from dmdod.features.hud_features import (
    HUD_FEATURE_DIM,
    HUD_FEATURE_VERSION,
    HUD_REAL_CHART_INPUT_DIM,
    encode_bpm,
    encode_bpm_legacy_v1,
    encode_hud_real_chart_observation,
)
from dmdod.motor.env import MotorObservation
from dmdod.motor.n_key import NKeyObservation, n_key_names


def _two_key_observation(*, bpm:float, real_bpm:float) -> HudRealChartObservation:
    return HudRealChartObservation(
        motor=MotorObservation(
            left_position_m=0.0, right_position_m=0.0,
            left_velocity_m_s=0.0, right_velocity_m_s=0.0,
            left_pressed=False, right_pressed=False),
        orbiting_x=0.0, orbiting_y=1.5, floors=(),
        tile_bpm=bpm, real_bpm=real_bpm,
        feedback_visible=False, last_judgement=None,
        last_timing_error_ms=0.0,
    )


def _eight_key_observation(*, bpm:float, real_bpm:float) -> NKeyHudRealChartObservation:
    keys=n_key_names(8)
    return NKeyHudRealChartObservation(
        motor=NKeyObservation(
            key_names=keys,
            positions_m=(0.0,)*8,
            velocities_m_s=(0.0,)*8,
            pressed_flags=(False,)*8),
        orbiting_x=0.0, orbiting_y=1.5, floors=(),
        tile_bpm=bpm, real_bpm=real_bpm,
        feedback_visible=False,last_judgement=None,
        last_timing_error_ms=0.0,
    )


@pytest.mark.parametrize("bpm",[
    1e-9, 1.0, 7.5, 120.0, 240.0, 1920.0,
    2380.0, 4000.0, 8000.0, 10000.0, 1000000.0,
])
def test_v2_log_bpm_is_unclipped_and_reversible(bpm:float):
    encoded=encode_bpm(bpm)
    assert isfinite(encoded)
    assert encoded==pytest.approx(log2(bpm)-log2(120.),abs=1e-13)
    assert 120.0*(2.0**encoded)==pytest.approx(bpm,rel=1e-12)


def test_v2_retains_double_speed_without_upper_plateau():
    assert encode_bpm(4000.)>4.0
    assert encode_bpm(8000.)==pytest.approx(encode_bpm(4000.)+1.)
    assert encode_bpm(16000.)==pytest.approx(encode_bpm(8000.)+1.)
    assert encode_bpm(2380.)==pytest.approx(4.309855262586788,abs=.0005)


@pytest.mark.parametrize("bpm",[0.0,-1.0,-240.0,nan,inf,-inf])
def test_v2_invalid_bpm_fails_explicitly(bpm:float):
    with pytest.raises(ValueError,match="finite and strictly positive"):
        encode_bpm(bpm)


def test_v1_codec_is_explicitly_legacy_and_byte_for_byte_equivalent():
    assert "v2" in HUD_FEATURE_VERSION and "unclipped" in HUD_FEATURE_VERSION
    assert "v2" in N_KEY_HUD_FEATURE_VERSION
    assert encode_bpm_legacy_v1(2380.)==4.0
    assert encode_bpm_legacy_v1(4000.)==4.0
    assert encode_bpm_legacy_v1(8000.)==4.0
    assert encode_bpm_legacy_v1(1e-9)==-4.0
    for v in (60.,120.,240.,960.,1920.):
        assert encode_bpm_legacy_v1(v)==pytest.approx(encode_bpm(v))


def test_two_key_hud_defaults_to_non_clipped_bpm_with_legacy_opt_in():
    obs=_two_key_observation(bpm=4000.,real_bpm=8000.)
    new=encode_hud_real_chart_observation(obs)
    old=encode_hud_real_chart_observation(obs,legacy_bpm_encoding=True)
    start=HUD_REAL_CHART_INPUT_DIM-HUD_FEATURE_DIM
    assert len(new)==len(old)==HUD_REAL_CHART_INPUT_DIM
    assert new[start:start+2]==pytest.approx((encode_bpm(4000.),encode_bpm(8000.)))
    assert old[start:start+2]==(4.0,4.0)
    assert new[:start]==old[:start]
    assert new[start+2:]==old[start+2:]


def test_eight_key_263d_hud_defaults_to_non_clipped_bpm_with_legacy_opt_in():
    obs=_eight_key_observation(bpm=4000.,real_bpm=8000.)
    new=encode_n_key_hud_real_chart_observation(obs)
    old=encode_n_key_hud_real_chart_observation(obs,legacy_bpm_encoding=True)
    start=n_key_hud_real_chart_input_dim(8)-HUD_FEATURE_DIM
    assert len(new)==len(old)==263
    assert new[start:start+2]==pytest.approx((encode_bpm(4000.),encode_bpm(8000.)))
    assert old[start:start+2]==(4.0,4.0)
    assert new[:start]==old[:start]
    assert new[start+2:]==old[start+2:]


def test_eight_key_invalid_bpm_is_not_silently_clamped():
    obs=_eight_key_observation(bpm=0.,real_bpm=120.)
    with pytest.raises(ValueError,match="finite and strictly positive"):
        encode_n_key_hud_real_chart_observation(obs)


def test_subnormal_positive_bpm_does_not_underflow_during_log_conversion():
    low=float.fromhex("0x0.0000000000001p-1022")
    assert encode_bpm(low)==pytest.approx(log2(low)-log2(120.))
