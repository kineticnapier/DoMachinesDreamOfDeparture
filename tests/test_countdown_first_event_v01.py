from __future__ import annotations

from math import cos, pi, sin

import pytest

from dmdod.planning.countdown_first_event_v01 import (
    CountdownFirstEventEstimator,
    encode_visible_bpm,
)
from dmdod.planning.countdown_phase_tracker_v01 import (
    CountdownPhaseUpdate,
    TimeInterval,
)
from dmdod.planning.hud_timing_tracker_transition_v14 import (
    CURRENT_SLOT, FLOOR_DIM, HUD_DIM, HUD_OFFSET, MOTOR_FEATURES, SLOT_START,
)


def hud(*, bpm: float, orbit: float, entry=0.0, exit_angle=pi / 2,
        marker=None):
    values = [0.0] * HUD_DIM
    values[MOTOR_FEATURES] = sin(orbit)
    values[MOTOR_FEATURES + 1] = cos(orbit)
    start = SLOT_START + CURRENT_SLOT * FLOOR_DIM
    floor = [0.0] * FLOOR_DIM
    floor[2], floor[3] = sin(entry), cos(entry)
    floor[4], floor[5] = sin(exit_angle), cos(exit_angle)
    floor[-1] = 1.0
    if marker is not None:
        floor[marker] = 1.0
    values[start:start + FLOOR_DIM] = floor
    values[HUD_OFFSET] = encode_visible_bpm(bpm)
    return tuple(values)


def phase(step, time_s, *, digit=1, period=(.49, .51),
          onset=(.99, 1.01), valid=True, status="number"):
    return CountdownPhaseUpdate(
        step=step, observed_at_s=time_s, status=status,
        displayed_digit=digit if status == "number" else None,
        transition_interval_s=TimeInterval(*onset) if onset else None,
        measured_tick_period_s=TimeInterval(*period) if period else None,
        remaining_transitions_to_go=digit if status == "number" else 0,
        transition_confirmations=2, evidence_age_s=0.0, valid=valid,
    )


@pytest.mark.parametrize("bpm", [120.0, 240.0, 480.0])
def test_visible_geometry_forecasts_due_interval_and_lead(bpm) -> None:
    estimator = CountdownFirstEventEstimator()
    dt = .01
    speed = pi * bpm / 60.0
    first = estimator.observe(
        phase(0, 1.0, onset=None), hud(bpm=bpm, orbit=0.0)
    )
    result = estimator.observe(
        phase(1, 1.01), hud(bpm=bpm, orbit=speed * dt)
    )
    arc_s = (pi / 2) / speed
    assert not first.eligible
    assert result.due_interval_s.lower_s == pytest.approx(.99 + .49 + arc_s)
    assert result.due_interval_s.upper_s == pytest.approx(1.01 + .51 + arc_s)
    assert result.lead_guaranteed


def test_pitch_effect_is_carried_by_observed_real_bpm_and_period() -> None:
    estimator = CountdownFirstEventEstimator()
    bpm = 240.0  # 120 BPM chart at 200% pitch, as visibly rendered.
    estimator.observe(
        phase(0, .50, period=(.24, .26), onset=None),
        hud(bpm=bpm, orbit=0.0),
    )
    result = estimator.observe(
        phase(1, .51, period=(.24, .26), onset=(.49, .51)),
        hud(bpm=bpm, orbit=pi * bpm / 60.0 * .01),
    )
    assert result.eligible
    assert result.due_interval_s.width_s == pytest.approx(.04)


def test_due_lower_bound_controls_44ms_guarantee() -> None:
    estimator = CountdownFirstEventEstimator(required_lead_s=.044)
    bpm = 120.0
    estimator.observe(
        phase(0, 1.0, status="go", digit=None, period=None, onset=None),
        hud(bpm=bpm, orbit=0.0, exit_angle=.20),
    )
    result = estimator.observe(
        phase(1, 1.01, status="go", digit=None, period=None,
              onset=(.99, 1.01)),
        hud(bpm=bpm, orbit=pi * bpm / 60.0 * .01, exit_angle=.20),
    )
    assert result.eligible
    assert result.lead_interval_s.lower_s < .044
    assert not result.lead_guaranteed
    assert result.status == "eligible_insufficient_lead"


@pytest.mark.parametrize("marker", [6, 7, 8, 9, 10, 11, 12, 13])
def test_special_floor_markers_abstain(marker) -> None:
    estimator = CountdownFirstEventEstimator()
    bpm = 120.0
    estimator.observe(phase(0, 1.0, onset=None), hud(bpm=bpm, orbit=0.0))
    result = estimator.observe(
        phase(1, 1.01),
        hud(bpm=bpm, orbit=pi * bpm / 60.0 * .01, marker=marker),
    )
    assert not result.eligible
    assert result.status == "unsupported_countdown_floor"


def test_invalid_phase_and_direction_alias_abstain() -> None:
    invalid = CountdownFirstEventEstimator().observe(
        phase(0, 1.0, valid=False, status="invalid_or_stale_frame"),
        hud(bpm=120.0, orbit=0.0),
    )
    assert not invalid.eligible and invalid.status.startswith("phase_")

    estimator = CountdownFirstEventEstimator()
    estimator.observe(phase(0, 1.0, onset=None), hud(bpm=4000.0, orbit=0.0))
    alias = estimator.observe(
        phase(1, 1.01), hud(bpm=4000.0, orbit=.1)
    )
    assert not alias.eligible and alias.status == "direction_alias"
