from __future__ import annotations

import ast
from dataclasses import fields
import inspect
from math import cos, pi, sin
import textwrap

import pytest

from dmdod.features.countdown_hud_v3 import (
    CountdownCueKind, CountdownStudentObservation, CountdownVisualSensor,
    RenderedCountdownFrame, encode_hud_v3,
)
from dmdod.planning.countdown_first_event_v01 import (
    CountdownFirstEventEstimate, CountdownFirstEventEstimator,
    encode_visible_bpm,
)
from dmdod.planning.countdown_phase_tracker_v01 import CountdownPhaseUpdate
from dmdod.planning.contact_timing_hud_v01 import (
    MOD_VISIBLE_BPM_CONTRACT,
    STOCK_ONLY_BPM_CONTRACT,
)
from dmdod.planning.hud_timing_composite_v20 import (
    HUDTimingComposite, TimingEventToken, TimingFrameUpdateV20,
)
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


def composite(**kwargs):
    return HUDTimingComposite(
        bpm_observation_contract=MOD_VISIBLE_BPM_CONTRACT,
        **kwargs,
    )


def test_hidden_normal_play_preserves_raw_v14_update_and_status() -> None:
    system = composite()
    direct = CausalTimingTracker(horizon_s=.080)
    for step in range(3):
        old = legacy(step=step)
        expected = direct.observe(step, old)
        actual = composite_step(
            system, step, CountdownCueKind.HIDDEN, hud=old
        )
        assert actual.v14_update == expected
        assert actual.status == expected.status
        assert all(item.local_id.startswith("v14:") for item in actual.pending)


def test_278d_suffix_must_match_causal_observation() -> None:
    system = composite()
    cue = observation(0, CountdownCueKind.HIDDEN)
    mismatched = observation(0, CountdownCueKind.READY)
    frame = encode_hud_v3(legacy(step=0), cue)
    with pytest.raises(ValueError, match="suffix"):
        system.observe(frame, mismatched)


def test_countdown_token_id_is_stable_and_reused() -> None:
    system = composite()
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
    system = composite(v14_tracker=tracker)
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
    system = composite(v14_tracker=CausalTimingTracker(
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
    system = composite(v14_tracker=CausalTimingTracker(
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


def test_invalid_countdown_carries_existing_token_unconfirmed_and_reused() -> None:
    calls = 0

    def predictor(_hud):
        nonlocal calls
        calls += 1
        return (
            WindowForecast((1,), (.05,), "eligible")
            if calls >= 5 else WindowForecast((), (), "insufficient_lookahead")
        )

    tracker = CausalTimingTracker(
        horizon_s=.080,
        predictor=predictor,
    )
    system = composite(v14_tracker=tracker)
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
    assert invalid.reused_ids == ("countdown:0",)
    assert invalid.v14_update.pending
    assert all(item.local_id == "countdown:0" for item in invalid.pending)


@pytest.mark.parametrize("failure", ["missing", "stale"])
def test_missing_or_stale_countdown_frame_never_publishes_v14_candidate(
    failure,
) -> None:
    tracker = CausalTimingTracker(
        horizon_s=.080,
        predictor=lambda _hud: WindowForecast((1,), (.05,), "eligible"),
    )
    system = composite(v14_tracker=tracker)
    sensor = CountdownVisualSensor(max_frame_age_s=.005)
    ready = sensor.observe(
        step=0, observed_at_s=0.0,
        frame=RenderedCountdownFrame(0.0, "Ready!"),
    )
    system.observe(encode_hud_v3(legacy(step=0), ready), ready)
    failed_frame = (
        RenderedCountdownFrame(.01, None, frame_available=False)
        if failure == "missing"
        else RenderedCountdownFrame(0.0, "Ready!")
    )
    cue = sensor.observe(
        step=1, observed_at_s=.01, frame=failed_frame
    )
    invalid = system.observe(encode_hud_v3(legacy(step=1), cue), cue)
    assert not cue.valid and cue.kind is CountdownCueKind.UNKNOWN
    assert invalid.v14_update.pending
    assert invalid.pending == ()
    assert invalid.new_ids == ()


def test_extra_nonoverlapping_v14_candidate_forces_abstention() -> None:
    calls = 0

    def predictor(_hud):
        nonlocal calls
        calls += 1
        return (
            WindowForecast((1, 2), (.05, .07), "eligible")
            if calls >= 3 else WindowForecast((), (), "insufficient_lookahead")
        )

    system = composite(v14_tracker=CausalTimingTracker(
        horizon_s=.080, predictor=predictor
    ))
    for step, (kind, digit) in enumerate([
        (CountdownCueKind.READY, None),
        (CountdownCueKind.NUMBER, 2),
        (CountdownCueKind.NUMBER, 1),
        (CountdownCueKind.GO, None),
    ]):
        result = composite_step(
            system, step, kind, digit, hud=legacy(step=step, arc=pi * .1)
        )
    assert len(result.v14_update.pending) == 2
    assert result.status == "source_conflict"
    assert result.pending == ()
    assert "ambiguous_or_disjoint" in result.abstention_reason


def test_contradictory_countdown_interval_cancels_existing_id() -> None:
    empty = CausalTimingTracker(
        horizon_s=.080,
        predictor=lambda _hud: WindowForecast((), (), "insufficient_lookahead"),
    )
    system = composite(v14_tracker=empty)
    for step, (kind, digit) in enumerate([
        (CountdownCueKind.READY, None),
        (CountdownCueKind.NUMBER, 2),
        (CountdownCueKind.NUMBER, 1),
        (CountdownCueKind.GO, None),
    ]):
        created = composite_step(
            system, step, kind, digit, hud=legacy(step=step, arc=pi * .1)
        )
    conflict = composite_step(
        system, 4, CountdownCueKind.HIDDEN,
        hud=legacy(step=4, arc=pi * 1.5),
    )
    assert created.new_ids == ("countdown:0",)
    assert conflict.status == "countdown_interval_conflict"
    assert conflict.pending == ()
    assert [item.local_id for item in conflict.canceled] == ["countdown:0"]


def test_lifecycle_groups_partition_pending_and_released_ids() -> None:
    empty = CausalTimingTracker(
        horizon_s=.080,
        predictor=lambda _hud: WindowForecast((), (), "insufficient_lookahead"),
    )
    system = composite(v14_tracker=empty)
    updates = []
    cues = [
        (CountdownCueKind.READY, None, True),
        (CountdownCueKind.NUMBER, 2, True),
        (CountdownCueKind.NUMBER, 1, True),
        (CountdownCueKind.GO, None, True),
        (CountdownCueKind.UNKNOWN, None, False),
    ] + [(CountdownCueKind.HIDDEN, None, True)] * 8
    for step, (kind, digit, valid) in enumerate(cues):
        updates.append(composite_step(
            system, step, kind, digit, valid=valid,
            hud=legacy(step=step, arc=pi * .1),
        ))

    previous = set()
    saw_new = saw_reused = saw_elapsed = False
    for update in updates:
        pending = {item.local_id for item in update.pending}
        new = set(update.new_ids)
        reused = set(update.reused_ids)
        elapsed = {item.local_id for item in update.elapsed}
        canceled = {item.local_id for item in update.canceled}
        assert new.isdisjoint(reused)
        assert pending == new | reused
        assert elapsed.isdisjoint(canceled)
        assert previous - pending == elapsed | canceled
        previous = pending
        saw_new |= bool(new)
        saw_reused |= bool(reused)
        saw_elapsed |= bool(elapsed)
    assert saw_new and saw_reused and saw_elapsed


def test_stock_only_contract_cannot_claim_numeric_bpm_observation() -> None:
    with pytest.raises(ValueError, match="stock-only"):
        HUDTimingComposite(bpm_observation_contract=STOCK_ONLY_BPM_CONTRACT)


def test_local_clock_and_step_are_strict() -> None:
    system = composite()
    cue = CountdownStudentObservation(
        step=0, observed_at_s=.001, rendered_at_s=.001,
        valid=True, kind=CountdownCueKind.HIDDEN,
    )
    with pytest.raises(ValueError, match="10ms"):
        system.observe(encode_hud_v3(legacy(step=0), cue), cue)


def test_policy_contract_and_estimators_contain_no_truth_fields() -> None:
    public = {
        item.name
        for cls in (
            CountdownPhaseUpdate, CountdownFirstEventEstimate,
            TimingEventToken, TimingFrameUpdateV20,
        )
        for item in fields(cls)
    }
    forbidden = {
        "countdownTicks", "countdown_ticks", "target_time_s",
        "target_index", "target_ordinal", "floor_index", "chart_time",
        "first_tick_s", "tick_period_s", "judgement", "_used", "_missed",
    }
    assert public.isdisjoint(forbidden)
    for function in (
        CountdownFirstEventEstimator.observe, HUDTimingComposite.observe,
    ):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        attributes = {
            node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        }
        assert attributes.isdisjoint({
            "targets", "target_time_s", "episode_time_s", "chart_time_s",
            "countdown_ticks", "_used", "_missed",
        })
