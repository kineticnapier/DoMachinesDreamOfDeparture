from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import pytest

from dmdod.envs.n_key import NKeyHudRealChartObservation
from dmdod.motor.n_key import NKeyAction, NKeyObservation, n_key_names
from dmdod.planning import causal_contract as c
from dmdod.training import prestige1 as frozen


def frame(step: int, *, position: float = 0.) -> c.VisibleHUDFrame:
    return c.VisibleHUDFrame(step, (position,) * c.HUD_DIM)


def test_exact_frozen_visible_hud_and_motor_shape():
    assert c.HUD_DIM == frozen.INPUT_DIM == 263
    assert c.KEY_COUNT == frozen.KEY_COUNT == 8
    assert c.CONTROL_DT_S == frozen.CONTROL_DT == .010
    assert c.FORECAST_HORIZON_S == frozen.FORECAST_WINDOW_S == .200
    assert c.FINGER_ORDER == n_key_names(8)
    valid = frame(0)
    assert len(valid.features) == 263
    assert valid.normalized_positions == (0.,) * 8
    assert valid.normalized_velocities == (0.,) * 8
    assert valid.pressed == (False,) * 8


def test_visible_hud_encoder_reads_only_visible_observation():
    motor = NKeyObservation(
        key_names=n_key_names(8),
        positions_m=(0.006,) * 8,
        velocities_m_s=(0.5,) * 8,
        pressed_flags=(True,) + (False,) * 7,
    )
    current = NKeyHudRealChartObservation(
        motor=motor, orbiting_x=0., orbiting_y=0.,
        floors=(), tile_bpm=180., real_bpm=180.,
        feedback_visible=False, last_judgement=None,
        last_timing_error_ms=0.,
    )
    observed = c.VisibleHUDFrame.from_observation(12, current)
    assert observed.step == 12
    assert len(observed.features) == 263
    assert observed.normalized_positions == pytest.approx((1.,) * 8)
    assert observed.normalized_velocities == pytest.approx((.5,) * 8)
    assert observed.pressed == (True,) + (False,) * 7
    with pytest.raises(TypeError, match="current"):
        c.VisibleHUDFrame.from_observation(12, SimpleNamespace(
            motor=motor, target_time_s=.12))


@pytest.mark.parametrize("step,values", [
    (-1, (0.,) * 263),
    (False, (0.,) * 263),
    (1, (0.,) * 262),
    (1, (0.,) * 262 + (float("nan"),)),
    (1, (0.,) * 262 + (float("inf"),)),
])
def test_invalid_hud_frames_rejected(step, values):
    with pytest.raises(ValueError):
        c.VisibleHUDFrame(step, values)


def test_history_is_strictly_causal_and_monotonic():
    start = c.CausalHUDHistory((frame(6), frame(7)))
    assert start.current.step == 7
    assert start.append(frame(8), max_frames=2).frames == (frame(7), frame(8))
    assert start.current.step == 7, "immutable causal sequence"
    with pytest.raises(ValueError, match="Cannot append"):
        start.append(frame(7), max_frames=2)
    with pytest.raises(ValueError, match="Cannot append"):
        start.append(frame(9), max_frames=2)
    with pytest.raises(ValueError, match="consecutive"):
        c.CausalHUDHistory((frame(2), frame(4)))
    with pytest.raises(ValueError, match="empty"):
        c.CausalHUDHistory(())
    with pytest.raises(ValueError, match="max_frames"):
        start.append(frame(8), max_frames=0)


def test_timing_head_output_has_no_finger_identity_or_target_truth():
    event = c.PredictedEvent(.12, 3, .75)
    assert tuple(f.name for f in fields(c.PredictedEvent)) == (
        "relative_time_s", "chord_size", "confidence")
    assert c.TimingPrediction((event,)).events[0].chord_size == 3
    assert c.TimingPrediction(()).events == ()
    with pytest.raises(ValueError, match="horizon"):
        c.PredictedEvent(.21, 1, .8)
    with pytest.raises(ValueError, match="horizon"):
        c.PredictedEvent(-.001, 1, .8)
    with pytest.raises(ValueError, match="chord"):
        c.PredictedEvent(.1, 9, .8)
    with pytest.raises(ValueError, match="confidence"):
        c.PredictedEvent(.1, 2, 1.1)
    with pytest.raises(ValueError, match="sorted"):
        c.TimingPrediction((
            c.PredictedEvent(.15, 1, .5), c.PredictedEvent(.10, 1, .6)))
    with pytest.raises(ValueError, match="simultaneous"):
        c.TimingPrediction((
            c.PredictedEvent(.1, 1, .5), c.PredictedEvent(.1, 1, .6)))


def test_fingering_is_separate_from_timing_and_can_abstain():
    timing = c.TimingPrediction((
        c.PredictedEvent(.01, 1, .5),
        c.PredictedEvent(.07, 2, .9),
    ))
    result = c.join_predictions(
        timing, c.FingeringProposal((c.EventFingering(1, (3, 4)),)))
    assert result.abstained_event_indices == (0,)
    assert result.events == (c.TimedFingerIntent(.07, (3, 4), .9),)
    assert not hasattr(result.events[0], "teacher_action")
    assert not hasattr(result.events[0], "target_index")
    assert c.join_predictions(timing, c.FingeringProposal(())).events == ()
    assert c.join_predictions(c.TimingPrediction(()), c.FingeringProposal(())) == c.JointIntent((), ())


def test_fingering_must_match_predicted_chord():
    timing = c.TimingPrediction((c.PredictedEvent(.06, 2, .95),))
    with pytest.raises(ValueError, match="chord"):
        c.join_predictions(
            timing, c.FingeringProposal((c.EventFingering(0, (3,)),)))
    with pytest.raises(ValueError, match="absent"):
        c.join_predictions(
            timing, c.FingeringProposal((c.EventFingering(1, (3, 4)),)))
    with pytest.raises(ValueError, match="physical key"):
        c.EventFingering(0, (3, 3))
    with pytest.raises(ValueError, match="physical key order"):
        c.EventFingering(0, (4, 3))
    with pytest.raises(ValueError, match="0..7"):
        c.EventFingering(0, (8,))
    with pytest.raises(ValueError, match="sorted and unique"):
        c.FingeringProposal((
            c.EventFingering(1, (3,)), c.EventFingering(1, (4,))))


def test_joint_plan_does_not_implicitly_drive_motor():
    event = c.join_predictions(
        c.TimingPrediction((c.PredictedEvent(.07, 1, .9),)),
        c.FingeringProposal((c.EventFingering(0, (3,)),)),
    )
    assert event.events[0].relative_time_s == .07
    assert c.validate_motor_action(NKeyAction((0.,) * 8)).values == (0.,) * 8
    with pytest.raises(ValueError, match="eight finite forces"):
        c.validate_motor_action(NKeyAction((0.,) * 7))
    with pytest.raises(ValueError, match="eight finite forces"):
        c.validate_motor_action(NKeyAction((0.,) * 7 + (float("nan"),)))
    with pytest.raises(ValueError, match="eight finite forces"):
        c.validate_motor_action(NKeyAction((0.,) * 7 + (1.001,)))
