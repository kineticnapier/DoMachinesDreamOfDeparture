from __future__ import annotations

from dataclasses import replace
import json

import pytest

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai


def _compile(*, duration=None, floor=1, bpm=120.0, pitch=100.0,
             angle_data=(0.0, 90.0, 180.0), extra_actions=()):
    actions = list(extra_actions)
    if duration is not None:
        actions.append({"floor": floor, "eventType": "Hold", "duration": duration})
    raw = json.dumps({
        "angleData": list(angle_data),
        "settings": {
            "bpm": bpm, "pitch": pitch, "countdownTicks": 0,
        },
        "actions": actions,
    })
    return compile_adofai(parse_adofai_text(raw))


@pytest.mark.parametrize("duration", (0, 1, 2, 3))
def test_hold_adds_stock_full_turns_without_changing_start(duration):
    baseline = _compile()
    held = _compile(duration=duration)
    ordinary = baseline.floors[1]
    floor = held.floors[1]

    assert floor.target_time_s == ordinary.target_time_s
    assert floor.hold is not None
    assert floor.hold.duration_loops == duration
    assert floor.hold.start_time_s == floor.target_time_s
    assert floor.hold.end_time_s == floor.exit_time_s
    assert floor.hold.additional_angle_rad == pytest.approx(
        duration * 2.0 * 3.1415927410125732
    )
    # At 120 BPM, one additional full turn is two crotchets = one second.
    assert floor.hold.additional_time_s == pytest.approx(duration * 1.0)
    assert floor.exit_time_s - ordinary.exit_time_s == pytest.approx(duration)
    assert held.floors[2].target_time_s - baseline.floors[2].target_time_s == pytest.approx(duration)


def test_hold_pitch_changes_episode_interval_not_chart_timing():
    normal = _compile(duration=2, pitch=100)
    pitched = _compile(duration=2, pitch=50)
    assert tuple(f.target_time_s for f in pitched.floors) == tuple(
        f.target_time_s for f in normal.floors
    )

    segment = build_playable_segment(pitched)
    hold = segment.holds[0]
    assert hold.chart_start_s == pitched.floors[1].target_time_s
    assert hold.chart_end_s == pitched.floors[1].exit_time_s
    assert hold.episode_end_s - hold.episode_start_s == pytest.approx(
        (hold.chart_end_s - hold.chart_start_s) / .5
    )


def test_hold_combines_with_setspeed_pause_and_twirl_without_guessing_future_speed():
    actions = (
        {"floor": 1, "eventType": "Twirl"},
        {"floor": 1, "eventType": "SetSpeed", "speedType": "Multiplier",
         "bpmMultiplier": 2.0},
        {"floor": 1, "eventType": "Pause", "duration": 1.0},
    )
    baseline = _compile(extra_actions=actions)
    held = _compile(duration=1, extra_actions=actions)

    assert held.floors[1].is_ccw
    assert held.floors[1].pause_s == pytest.approx(.25)
    assert held.floors[1].bpm == pytest.approx(240.0)
    assert held.floors[1].hold.additional_time_s == pytest.approx(.5)
    assert held.floors[1].exit_time_s - baseline.floors[1].exit_time_s == pytest.approx(.5)


def test_hold_midspin_and_multiplanet_keep_stock_base_geometry_then_add_loops():
    actions = ({"floor": 1, "eventType": "MultiPlanet", "planets": "ThreePlanets"},)
    baseline = _compile(angle_data=(0.0, 999.0, 180.0), extra_actions=actions)
    held = _compile(duration=1, angle_data=(0.0, 999.0, 180.0), extra_actions=actions)

    assert held.floors[1].midspin
    assert held.floors[1].num_planets == 3
    assert baseline.floors[1].exit_time_s - baseline.floors[1].target_time_s == 0.0
    assert held.floors[1].exit_time_s - held.floors[1].target_time_s == pytest.approx(1.0)
    assert held.floors[1].hold.additional_time_s == pytest.approx(1.0)


def test_consecutive_holds_are_distinct_intervals_and_targets_remain_ordinary():
    chart = _compile(
        duration=None,
        extra_actions=(
            {"floor": 1, "eventType": "Hold", "duration": 1},
            {"floor": 2, "eventType": "Hold", "duration": 2},
        ),
    )
    segment = build_playable_segment(chart)

    assert [hold.start_floor_index for hold in segment.holds] == [1, 2]
    assert segment.holds[0].end_floor_index == 2
    assert segment.holds[0].chart_end_s == segment.holds[1].chart_start_s
    assert [target.floor_index for target in segment.targets] == [1, 2, 3]


def test_negative_and_final_floor_hold_are_disabled_but_fractional_is_rejected():
    negative = _compile(duration=-1)
    final = _compile(duration=2, floor=3)
    assert negative.floors[1].hold is None
    assert final.floors[3].hold is None

    with pytest.raises(ValueError, match="Hold duration must be an integer"):
        _compile(duration=1.5)


def test_missing_hold_duration_is_rejected_and_inactive_hold_is_noop():
    missing = json.dumps({
        "angleData": [0, 90],
        "settings": {"bpm": 120, "countdownTicks": 0},
        "actions": [{"floor": 1, "eventType": "Hold"}],
    })
    with pytest.raises(ValueError, match="Hold duration is required"):
        compile_adofai(parse_adofai_text(missing))

    base = _compile()
    inactive = _compile(extra_actions=(
        {"floor": 1, "eventType": "Hold", "duration": 3, "active": False},
    ))
    assert inactive == replace(base, source_path=inactive.source_path)
