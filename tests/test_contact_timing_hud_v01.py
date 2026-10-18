from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, replace
import inspect
from math import pi

import pytest

from dmdod.adofai_rules import TimingJudgement
from dmdod.envs.contact_chart_16k import ContactChartObservation, CoupledContactRealChartEnv
from dmdod.envs.n_key import (
    DiagnosticHudNKeyRealChartMotorEnv,
    NKeyHudRealChartObservation,
    encode_n_key_hud_real_chart_observation,
)
from dmdod.envs.real_chart import RelativeVisibleFloor
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG as FLOORS
from dmdod.motor.contact_16k import ContactAction, Coupled16ContactMotor
from dmdod.motor.n_key import NKeyAction, NKeyObservation, n_key_names
from dmdod.planning import contact_timing_hud_v01 as adapter
from dmdod.planning.hud_timing_tracker_transition_v14 import (
    CausalTimingTracker, forecast_visible_window,
)
from scripts.audit_hud_timing_tracker_transition_v14_matrix import generated


def test_12axis_motor_is_not_silently_truncated_to_8keys():
    motor = Coupled16ContactMotor()
    physical = motor.motor
    physical.body.states[0].position_m = .0035
    physical.body.states[4].velocity_m_s = .55
    sample = ContactChartObservation(
        motor=motor.observe(), orbiting_x=1.5, orbiting_y=0.0,
        floors=(RelativeVisibleFloor(
            relative_index=0, x=0.0, y=0.0,
            entry_angle_rad=0.0, exit_angle_rad=pi,
            midspin=False, is_ccw=False, num_planets=2, event_markers=(),
        ),),
    )
    hud = adapter.ContactVisibleTimingHUD(sample, tile_bpm=5000., real_bpm=6200.)
    result = adapter.ContactTimingHUD263Adapter.encode(0, hud)
    assert len(result.features) == 263
    assert result.features[:24] == (0.0,) * 24
    assert result.features[24:26] == (1.0, 0.0)
    # v2 BPM uses unclipped log2; 5000 and 6200 must remain distinguishable.
    from dmdod.planning.hud_timing_tracker_transition_v14 import HUD_OFFSET
    assert result.features[HUD_OFFSET] != result.features[HUD_OFFSET + 1]
    assert result.features[HUD_OFFSET] > 4.0


def test_adapter_exactly_matches_reference_8key_hud_geometry_bpm_feedback():
    motor = Coupled16ContactMotor()
    geometry = ContactChartObservation(
        motor=motor.observe(), orbiting_x=.75, orbiting_y=.25,
        floors=(RelativeVisibleFloor(
            relative_index=0, x=0.0, y=0.0,
            entry_angle_rad=0.0, exit_angle_rad=pi / 2,
            midspin=False, is_ccw=False, num_planets=2,
            event_markers=("SetSpeed",),
        ),),
    )
    for visible in (False, True):
        exposed = adapter.ContactVisibleTimingHUD(
            contact=geometry, tile_bpm=2250., real_bpm=3000.,
            feedback_visible=visible,
            last_judgement=TimingJudgement.PERFECT if visible else None,
            last_timing_error_ms=-9.75 if visible else 0.0,
        )
        got = adapter.ContactTimingHUD263Adapter.encode(7, exposed)
        baseline = encode_n_key_hud_real_chart_observation(
            NKeyHudRealChartObservation(
                motor=NKeyObservation(
                    key_names=n_key_names(8), positions_m=(0.,)*8,
                    velocities_m_s=(0.,)*8, pressed_flags=(False,)*8,
                ),
                orbiting_x=geometry.orbiting_x, orbiting_y=geometry.orbiting_y,
                floors=geometry.floors, tile_bpm=exposed.tile_bpm,
                real_bpm=exposed.real_bpm,
                feedback_visible=exposed.feedback_visible,
                last_judgement=exposed.last_judgement,
                last_timing_error_ms=exposed.last_timing_error_ms,
            )
        )
        assert got.features == baseline
        assert got.step == 7
        assert got.version == adapter.CONTACT_TIMING_HUD_VERSION


@pytest.mark.parametrize("bad", [
    "neural_v3", "mlp", "gru", "v1", "", "hud_timing_tracker_transition_v15"
])
def test_estimator_guard_rejects_legacy_8key_neural_models(bad):
    with pytest.raises(ValueError, match="v1.4 ONLY"):
        adapter.ContactTimingHUD263Adapter(estimator=bad)


@pytest.mark.parametrize("tile,real", [
    (0.,120.), (120.,0.), (-1.,120.), (120.,-2.),
    (float("nan"),120.), (120.,float("inf")),
])
def test_visible_bpm_must_be_finite_and_positive(tile, real):
    motor = Coupled16ContactMotor()
    geometry = ContactChartObservation(motor.observe(), 0., 1., ())
    with pytest.raises(ValueError, match="BPM"):
        adapter.ContactVisibleTimingHUD(geometry, tile, real)


def test_hidden_feedback_is_not_mistaken_for_visible_judgement():
    motor = Coupled16ContactMotor()
    geometry = ContactChartObservation(motor.observe(), 0., 1., ())
    without = adapter.ContactVisibleTimingHUD(
        geometry, 120., 120., feedback_visible=False,
        last_judgement=TimingJudgement.PERFECT, last_timing_error_ms=16.,
    )
    plain = adapter.ContactVisibleTimingHUD(geometry, 120., 120.)
    assert (adapter.ContactTimingHUD263Adapter.encode(0, without).features ==
            adapter.ContactTimingHUD263Adapter.encode(0, plain).features)
    with pytest.raises(ValueError, match="visible feedback"):
        adapter.ContactVisibleTimingHUD(
            geometry, 120., 120., feedback_visible=True
        )


def test_layout_change_fails_closed(monkeypatch):
    assert adapter.ContactTimingHUD263Adapter() is not None
    monkeypatch.setattr(adapter.v14, "MOTOR_FEATURES", 25)
    with pytest.raises(RuntimeError, match="layout drift"):
        adapter.ContactTimingHUD263Adapter()


def test_frame_is_explicitly_versioned_neutral_and_immutable():
    features = (0.,) * 263
    frame = adapter.ContactTimingHUD263Frame(0, features)
    with pytest.raises(FrozenInstanceError):
        frame.step = 2
    with pytest.raises(ValueError, match="legacy Motor"):
        replace(frame, features=(1.,) + features[1:])
    with pytest.raises(ValueError, match="version"):
        replace(frame, version="unknown")
    with pytest.raises(ValueError, match="step"):
        replace(frame, step=-1)


def test_v14_timing_math_is_invariant_to_any_finite_legacy_motor_features():
    segment = generated((0., 90., 180., 270.), 496., "speed_double")
    env = DiagnosticHudNKeyRealChartMotorEnv(
        segment, key_count=8, control_dt_s=.01, physics_dt_s=.001,
        behind_floors=FLOORS.behind_floors, ahead_floors=FLOORS.ahead_floors,
        tail_s=.04,
    )
    obs = env.reset()
    first = encode_n_key_hud_real_chart_observation(obs)
    perturbed = tuple(float(i * (1 if i % 2 else -1)) for i in range(24)) + first[24:]
    assert forecast_visible_window(first) == forecast_visible_window(perturbed)
    ref_tracker, perturbed_tracker = CausalTimingTracker(), CausalTimingTracker()
    for step in range(15):
        vec = encode_n_key_hud_real_chart_observation(obs)
        noisy = tuple(float((step + 1) * (i + 1)) for i in range(24)) + vec[24:]
        assert ref_tracker.observe(step, vec) == perturbed_tracker.observe(step, noisy)
        obs = env.step(NKeyAction((0.,)*8)).observation


@pytest.mark.parametrize("variant", ["speed_double", "pause_1", "twirl_speed"])
def test_real_contact_frames_match_reference_v14_geometry_and_forecast(variant):
    """BPM test input is the same *visible* HUD values from the reference env.

    This checks contact geometry parity. A contact HUD BPM provider remains a
    separate integration step; the adapter itself never reconstructs BPM.
    """
    segment = generated((0., 90., 180., 270.), 496., variant)
    contact_env = CoupledContactRealChartEnv(
        segment, control_dt_s=.010, physics_dt_s=.001,
        behind_floors=FLOORS.behind_floors, ahead_floors=FLOORS.ahead_floors,
        tail_s=.04,
    )
    reference = DiagnosticHudNKeyRealChartMotorEnv(
        segment, key_count=8, control_dt_s=.010, physics_dt_s=.001,
        behind_floors=FLOORS.behind_floors, ahead_floors=FLOORS.ahead_floors,
        tail_s=.04,
    )
    c, n = contact_env.reset(), reference.reset()
    client = adapter.ContactTimingHUD263Adapter()
    ref_tracker = CausalTimingTracker()
    for step in range(20):
        current = adapter.ContactVisibleTimingHUD(
            contact=c,
            tile_bpm=n.tile_bpm,
            real_bpm=n.real_bpm,
            feedback_visible=n.feedback_visible,
            last_judgement=n.last_judgement,
            last_timing_error_ms=n.last_timing_error_ms,
        )
        received = client.encode(step, current)
        ref_hud = encode_n_key_hud_real_chart_observation(n)
        assert received.features == (0.,)*24 + ref_hud[24:]
        assert client.observe(received) == ref_tracker.observe(step, ref_hud)
        c = contact_env.step(ContactAction((0.,)*12)).observation
        n = reference.step(NKeyAction((0.,)*8)).observation


def test_sequential_frame_order_and_reset():
    segment = generated((0.,90.,180.,270.), 496., "speed_double")
    env = CoupledContactRealChartEnv(segment, tail_s=.04)
    c = env.reset()
    client = adapter.ContactTimingHUD263Adapter()
    observation = adapter.ContactVisibleTimingHUD(c, 496., 496.)
    first = client.encode(0, observation)
    client.observe(first)
    with pytest.raises(ValueError, match="consecutive"):
        client.observe(first)
    client.reset()
    client.observe(first)


def test_adapter_does_not_have_privileged_chart_or_evaluator_fields():
    forbidden = (
        "segment", "targets", "_used", "_missed",
        "_next_target_index", "privileged_episode_time_s",
        "chart_time_from_episode", "target_ordinal",
    )
    for data_type in (adapter.ContactVisibleTimingHUD,
                      adapter.ContactTimingHUD263Frame):
        assert not {f.name for f in fields(data_type)}.intersection(forbidden)
    module_ast = ast.parse(inspect.getsource(adapter))
    referenced = {
        node.attr for node in ast.walk(module_ast) if isinstance(node, ast.Attribute)
    }
    assert not referenced.intersection(forbidden)
