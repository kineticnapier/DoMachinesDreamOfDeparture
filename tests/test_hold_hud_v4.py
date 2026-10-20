from __future__ import annotations

from dataclasses import fields
import json

import pytest

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.timing import compile_adofai
from dmdod.features.countdown_hud_v3 import (
    CountdownCueKind,
    CountdownStudentObservation,
    HUD263_DIM,
    encode_hud_v3,
)
from dmdod.features.hold_hud_v4 import (
    HOLD_FLOOR_SLOTS,
    HOLD_ICON_FEATURE_DIM,
    HUD_V4_DIM,
    HoldCueKind,
    HoldIconVisualSensor,
    RenderedHoldIconFrame,
    StockHoldIconRenderer,
    encode_hud_v4,
)
from dmdod.features.real_chart import REAL_CHART_EVENT_MARKERS


def _countdown_frame():
    countdown = CountdownStudentObservation(
        step=0,
        observed_at_s=1.0,
        rendered_at_s=1.0,
        valid=True,
        kind=CountdownCueKind.HIDDEN,
    )
    return encode_hud_v3(tuple(float(i) for i in range(HUD263_DIM)), countdown)


def test_stock_short_and_long_start_and_release_icons() -> None:
    short = StockHoldIconRenderer.render(
        rendered_at_s=1.0, start_relative_index=2, duration=0
    )
    long = StockHoldIconRenderer.render(
        rendered_at_s=1.0, start_relative_index=2, duration=3
    )
    assert short.icon_at(2) == "HoldArrowShort"
    assert short.icon_at(3) == "HoldReleaseShort"
    assert long.icon_at(2) == "HoldArrowLong"
    assert long.icon_at(3) == "HoldReleaseLong"


def test_positive_durations_are_indistinguishable_from_icons_alone() -> None:
    one = StockHoldIconRenderer.render(
        rendered_at_s=1.0, start_relative_index=0, duration=1
    )
    three = StockHoldIconRenderer.render(
        rendered_at_s=1.0, start_relative_index=0, duration=3
    )
    sensor_a = HoldIconVisualSensor()
    sensor_b = HoldIconVisualSensor()
    observation_a = sensor_a.observe(step=0, observed_at_s=1.0, frame=one)
    observation_b = sensor_b.observe(step=0, observed_at_s=1.0, frame=three)
    assert observation_a == observation_b
    assert encode_hud_v4(_countdown_frame(), observation_a).features == encode_hud_v4(
        _countdown_frame(), observation_b
    ).features


def test_same_icon_observation_can_have_different_true_release_times() -> None:
    def compile_duration(duration: int):
        return compile_adofai(parse_adofai_text(json.dumps({
            "angleData": [0, 90, 180],
            "settings": {"bpm": 120, "pitch": 100, "countdownTicks": 0},
            "actions": [
                {"floor": 1, "eventType": "Hold", "duration": duration}
            ],
        })))

    one = compile_duration(1)
    three = compile_duration(3)
    assert one.floors[1].hold.end_time_s != three.floors[1].hold.end_time_s
    assert one.floors[1].hold.end_time_s == pytest.approx(1.75)
    assert three.floors[1].hold.end_time_s == pytest.approx(3.75)

    frame_one = StockHoldIconRenderer.render(
        rendered_at_s=0.0, start_relative_index=0, duration=1
    )
    frame_three = StockHoldIconRenderer.render(
        rendered_at_s=0.0, start_relative_index=0, duration=3
    )
    assert frame_one == frame_three
    assert "Hold" not in REAL_CHART_EVENT_MARKERS


def test_missing_stale_and_unrecognized_frames_are_not_imputed() -> None:
    sensor = HoldIconVisualSensor(max_frame_age_s=0.02)
    missing = sensor.observe(
        step=0,
        observed_at_s=1.0,
        frame=RenderedHoldIconFrame(
            rendered_at_s=1.0, icons=(), frame_available=False
        ),
    )
    stale = sensor.observe(
        step=1,
        observed_at_s=1.1,
        frame=RenderedHoldIconFrame(
            rendered_at_s=1.0, icons=((0, "HoldArrowLong"),)
        ),
    )
    unknown = HoldIconVisualSensor().observe(
        step=0,
        observed_at_s=2.0,
        frame=RenderedHoldIconFrame(
            rendered_at_s=2.0, icons=((0, "localized-or-occluded"),)
        ),
    )
    assert all(cue.kind is HoldCueKind.UNKNOWN for cue in missing.cues)
    assert all(cue.kind is HoldCueKind.UNKNOWN for cue in stale.cues)
    assert unknown.cue_at(0).kind is HoldCueKind.UNKNOWN
    assert not unknown.cue_at(0).valid


def test_occlusion_is_not_evidence_of_duration_zero_or_no_hold() -> None:
    rendered = StockHoldIconRenderer.render(
        rendered_at_s=1.0,
        start_relative_index=0,
        duration=2,
        occluded_relative_indices=(0, 1),
    )
    observation = HoldIconVisualSensor().observe(
        step=0, observed_at_s=1.0, frame=rendered
    )
    assert observation.cue_at(0).kind is HoldCueKind.NO_HOLD_ICON_VISIBLE
    assert observation.cue_at(1).kind is HoldCueKind.NO_HOLD_ICON_VISIBLE


def test_v4_keeps_the_278d_v3_and_263d_legacy_prefixes_exact() -> None:
    rendered = StockHoldIconRenderer.render(
        rendered_at_s=1.0, start_relative_index=-1, duration=0
    )
    observation = HoldIconVisualSensor().observe(
        step=0, observed_at_s=1.0, frame=rendered
    )
    v3 = _countdown_frame()
    v4 = encode_hud_v4(v3, observation)
    assert v4.features[: len(v3.features)] == v3.features
    assert v4.legacy_hud263 == v3.legacy_hud263
    assert len(v4.features) == HUD_V4_DIM
    assert len(v4.features) == len(v3.features) + HOLD_FLOOR_SLOTS * HOLD_ICON_FEATURE_DIM


def test_student_contract_contains_no_duration_or_target_truth() -> None:
    from dmdod.features.hold_hud_v4 import HoldIconStudentObservation, HoldFloorCue

    public = {
        item.name
        for cls in (HoldIconStudentObservation, HoldFloorCue)
        for item in fields(cls)
    }
    forbidden = {
        "duration",
        "hold_length",
        "holdLength",
        "target_time_s",
        "release_time_s",
        "floor_index",
    }
    assert public.isdisjoint(forbidden)


def test_renderer_rejects_final_floor_invalid_duration_and_bad_slot() -> None:
    with pytest.raises(ValueError, match="following floor"):
        StockHoldIconRenderer.render(
            rendered_at_s=0.0,
            start_relative_index=12,
            duration=1,
        )
    for value in (-1, 1.5, float("nan")):
        with pytest.raises((TypeError, ValueError)):
            StockHoldIconRenderer.render(
                rendered_at_s=0.0, start_relative_index=0, duration=value
            )


def test_sensor_rejects_future_or_nonsequential_frames() -> None:
    sensor = HoldIconVisualSensor()
    frame = RenderedHoldIconFrame(1.0, ())
    with pytest.raises(ValueError, match="future"):
        sensor.observe(step=0, observed_at_s=0.9, frame=frame)
    with pytest.raises(ValueError, match="consecutive"):
        sensor.observe(step=1, observed_at_s=1.0, frame=frame)
