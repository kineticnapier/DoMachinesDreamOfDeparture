from __future__ import annotations

from math import cos, pi, sin

import pytest

from dmdod.features.countdown_hud_v3 import (
    CountdownCueKind, CountdownStudentObservation, encode_hud_v3,
)
from dmdod.planning.countdown_first_event_v01 import encode_visible_bpm
from dmdod.planning.hud_timing_composite_v20 import HUDTimingComposite
from dmdod.planning.hud_timing_tracker_transition_v14 import (
    CURRENT_SLOT, CausalTimingTracker, FLOOR_DIM, HUD_DIM, HUD_OFFSET,
    MOTOR_FEATURES, SLOT_START, WindowForecast,
)


def legacy(*, step: int, bpm=120.0, arc=pi / 2):
    values = [0.0] * HUD_DIM
    angle = pi * bpm / 60.0 * step * .01
    values[MOTOR_FEATURES] = sin(angle)
    values[MOTOR_FEATURES + 1] = cos(angle)
    start = SLOT_START + CURRENT_SLOT * FLOOR_DIM
    floor = [0.0] * FLOOR_DIM
    floor[2], floor[3] = 0.0, 1.0
    floor[4], floor[5] = sin(arc), cos(arc)
    floor[-1] = 1.0
    values[start:start + FLOOR_DIM] = floor
    values[HUD_OFFSET] = encode_visible_bpm(bpm)
    return tuple(values)


def observation(step, kind, digit=None, *, valid=True):
    return CountdownStudentObservation(
        step=step, observed_at_s=step * .01, rendered_at_s=step * .01,
        valid=valid, kind=kind, digit=digit,
    )


def composite_step(system, step, kind, digit=None, *, hud=None, valid=True):
    cue = observation(step, kind, digit, valid=valid)
    old = legacy(step=step) if hud is None else hud
    return system.observe(encode_hud_v3(old, cue), cue)


def test_hidden_normal_play_preserves_raw_v14_update_and_status() -> None:
    composite = HUDTimingComposite()
    direct = CausalTimingTracker(horizon_s=.080)
    for step in range(3):
        old = legacy(step=step)
        expected = direct.observe(step, old)
        actual = composite_step(
            composite, step, CountdownCueKind.HIDDEN, hud=old
        )
        assert actual.v14_update == expected
        assert actual.status == expected.status
        assert all(item.local_id.startswith("v14:") for item in actual.pending)


def test_278d_suffix_must_match_causal_observation() -> None:
    system = HUDTimingComposite()
    cue = observation(0, CountdownCueKind.HIDDEN)
    mismatched = observation(0, CountdownCueKind.READY)
    frame = encode_hud_v3(legacy(step=0), cue)
    with pytest.raises(ValueError, match="suffix"):
        system.observe(frame, mismatched)


def test_countdown_token_id_is_stable_and_reused() -> None:
    system = HUDTimingComposite()
    sequence = [
        (CountdownCueKind.READY, None),
        (CountdownCueKind.NUMBER, 2),
        (CountdownCueKind.NUMBER, 1),
        (CountdownCueKind.GO, None),
        (CountdownCueKind.HIDDEN, None),
    ]
    results = []
    for step, (kind, digit) in enumerate(sequence):
        results.append(composite_step(system, step, kind, digit))
    created = results[3]
    reused = results[4]
    assert created.pending[0].local_id == "countdown:0"
    assert created.new_ids == ("countdown:0",)
    assert reused.pending[0].local_id == "countdown:0"
    assert reused.reused_ids == ("countdown:0",)


def test_visible_number_suppresses_raw_v14_countdown_orbit_candidates() -> None:
    tracker = CausalTimingTracker(
        horizon_s=.080,
        predictor=lambda _hud: WindowForecast((1,), (.05,), "eligible"),
    )
    system = HUDTimingComposite(v14_tracker=tracker)
    ready = composite_step(system, 0, CountdownCueKind.READY)
    number = composite_step(system, 1, CountdownCueKind.NUMBER, 2)
    assert ready.v14_update.pending and number.v14_update.pending
    assert ready.pending == () and number.pending == ()


def test_overlapping_sources_reuse_single_countdown_local_id() -> None:
    calls = 0
    def predictor(_hud):
        nonlocal calls
        calls += 1
        return (
            WindowForecast((1,), (.05,), "eligible")
            if calls >= 3 else WindowForecast((), (), "insufficient_lookahead")
        )
    system = HUDTimingComposite(v14_tracker=CausalTimingTracker(
        horizon_s=.080, predictor=predictor
    ))
    for step, (kind, digit) in enumerate([
        (CountdownCueKind.READY, None),
        (CountdownCueKind.NUMBER, 2),
        (CountdownCueKind.NUMBER, 1),
        (CountdownCueKind.GO, None),
    ]):
        result = composite_step(
            system, step, kind, digit,
            hud=legacy(step=step, arc=pi * .1),
        )
    assert [item.local_id for item in result.pending] == ["countdown:0"]
    assert result.pending[0].source.endswith("+timing_v1.4")


def test_nonoverlapping_sources_cancel_existing_token_and_abstain() -> None:
    calls = 0
    def predictor(_hud):
        nonlocal calls
        calls += 1
        return (
            WindowForecast((1,), (.01,), "eligible")
            if calls >= 5 else WindowForecast((), (), "insufficient_lookahead")
        )
    system = HUDTimingComposite(v14_tracker=CausalTimingTracker(
        horizon_s=.080, predictor=predictor
    ))
    frames = [legacy(step=step, arc=pi * .1) for step in range(6)]
    composite_step(system, 0, CountdownCueKind.READY, hud=frames[0])
    composite_step(system, 1, CountdownCueKind.NUMBER, 2, hud=frames[1])
    composite_step(
        system, 2, CountdownCueKind.NUMBER, 1, hud=frames[2]
    )
    created = composite_step(
        system, 3, CountdownCueKind.GO, hud=frames[3]
    )
    conflict = composite_step(
        system, 4, CountdownCueKind.HIDDEN, hud=frames[4]
    )
    assert created.pending[0].local_id == "countdown:0"
    assert conflict.status == "source_conflict"
    assert conflict.pending == ()
    assert [item.local_id for item in conflict.canceled] == ["countdown:0"]
    assert conflict.new_ids == ()
    after = composite_step(
        system, 5, CountdownCueKind.HIDDEN, hud=frames[5],
    )
    assert all(item.local_id != "countdown:0" for item in after.pending)


def test_invalid_countdown_carries_existing_token_unconfirmed() -> None:
    system = HUDTimingComposite()
    composite_step(system, 0, CountdownCueKind.READY)
    composite_step(system, 1, CountdownCueKind.NUMBER, 2)
    composite_step(system, 2, CountdownCueKind.NUMBER, 1)
    created = composite_step(system, 3, CountdownCueKind.GO)
    invalid = composite_step(
        system, 4, CountdownCueKind.UNKNOWN, valid=False
    )
    assert created.pending[0].local_id == "countdown:0"
    assert invalid.pending[0].local_id == "countdown:0"
    assert not invalid.pending[0].confirmed_now


def test_local_clock_and_step_are_strict() -> None:
    system = HUDTimingComposite()
    cue = CountdownStudentObservation(
        step=0, observed_at_s=.001, rendered_at_s=.001,
        valid=True, kind=CountdownCueKind.HIDDEN,
    )
    with pytest.raises(ValueError, match="10ms"):
        system.observe(encode_hud_v3(legacy(step=0), cue), cue)
