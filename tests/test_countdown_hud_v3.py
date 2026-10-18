from __future__ import annotations

from dataclasses import fields

import pytest

from dmdod.features.countdown_hud_v3 import (
    CountdownCueKind,
    CountdownHUDV3Frame,
    CountdownStudentObservation,
    CountdownVisualSensor,
    HUD263_DIM,
    HUD_V3_DIM,
    RenderedCountdownFrame,
    StockCountdownRenderer,
    encode_hud_v3,
)


def test_dll_normal_start_ready_number_go_schedule() -> None:
    renderer = StockCountdownRenderer(
        countdown_ticks=4, first_tick_s=1.0, tick_period_s=0.5
    )
    assert renderer.render(0.99).displayed_text == "Ready!"
    assert renderer.render(1.00).displayed_text == "3"
    assert renderer.render(1.50).displayed_text == "2"
    assert renderer.render(2.00).displayed_text == "1"
    assert renderer.render(2.50).displayed_text == "Go!"


def test_three_and_five_ticks_first_differ_on_rendered_digit() -> None:
    three = StockCountdownRenderer(
        countdown_ticks=3, first_tick_s=1.0, tick_period_s=1.0
    )
    five = StockCountdownRenderer(
        countdown_ticks=5, first_tick_s=1.0, tick_period_s=1.0
    )
    sensor_a, sensor_b = CountdownVisualSensor(), CountdownVisualSensor()
    ready_a = sensor_a.observe(step=0, observed_at_s=.5, frame=three.render(.5))
    ready_b = sensor_b.observe(step=0, observed_at_s=.5, frame=five.render(.5))
    assert ready_a == ready_b
    digit_a = sensor_a.observe(step=1, observed_at_s=1.0, frame=three.render(1.0))
    digit_b = sensor_b.observe(step=1, observed_at_s=1.0, frame=five.render(1.0))
    assert (digit_a.kind, digit_a.digit) == (CountdownCueKind.NUMBER, 2)
    assert (digit_b.kind, digit_b.digit) == (CountdownCueKind.NUMBER, 4)
    legacy = (0.0,) * HUD263_DIM
    assert encode_hud_v3(legacy, digit_a).legacy_hud263 == legacy
    assert encode_hud_v3(legacy, digit_a).features != encode_hud_v3(
        legacy, digit_b
    ).features


def test_unknown_or_unsupported_display_is_not_imputed() -> None:
    sensor = CountdownVisualSensor()
    unknown = sensor.observe(
        step=0, observed_at_s=0.0,
        frame=RenderedCountdownFrame(0.0, "localized-or-occluded"),
    )
    assert not unknown.valid
    assert unknown.kind is CountdownCueKind.UNKNOWN
    assert unknown.digit is None
    unsupported = StockCountdownRenderer(
        countdown_ticks=4, first_tick_s=0.0, tick_period_s=1.0,
        normal_start=False,
    ).render(0.0)
    result = sensor.observe(step=1, observed_at_s=.01, frame=unsupported)
    assert not result.valid
    assert result.kind is CountdownCueKind.UNKNOWN


def test_sensor_rejects_future_frame_and_nonsequential_access() -> None:
    sensor = CountdownVisualSensor()
    with pytest.raises(ValueError, match="future"):
        sensor.observe(
            step=0, observed_at_s=.9,
            frame=RenderedCountdownFrame(1.0, "Ready!"),
        )
    with pytest.raises(ValueError, match="consecutive"):
        sensor.observe(
            step=1, observed_at_s=1.0,
            frame=RenderedCountdownFrame(1.0, "Ready!"),
        )


def test_observation_latency_and_hud_v3_compatibility_prefix() -> None:
    sensor = CountdownVisualSensor()
    cue = sensor.observe(
        step=0, observed_at_s=1.010,
        frame=RenderedCountdownFrame(1.000, "2"),
    )
    frame = encode_hud_v3(tuple(float(i) for i in range(HUD263_DIM)), cue)
    assert frame.features[:HUD263_DIM] == tuple(float(i) for i in range(HUD263_DIM))
    assert len(frame.features) == HUD_V3_DIM
    assert cue.latency_s == pytest.approx(.010)
    assert frame.observed_at_s == pytest.approx(1.010)


def test_student_contract_has_no_hidden_count_or_target_truth_fields() -> None:
    public = {
        item.name for cls in (CountdownStudentObservation, CountdownHUDV3Frame)
        for item in fields(cls)
    }
    forbidden = {
        "countdown_ticks", "countdownTicks", "target_time_s", "target_index",
        "target_ordinal", "future_tick_s", "first_tick_s", "tick_period_s",
    }
    assert public.isdisjoint(forbidden)
