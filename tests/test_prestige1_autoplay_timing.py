from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from dmdod.adofai.playable import PlayableChartSegment, PlayableChartTarget
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher
from dmdod.training.prestige1_autoplay_export import compare_one_anchor


def segment(*times: float, start: float = 10., pitch: float = 1.25):
    targets = tuple(
        PlayableChartTarget(
            ordinal=i,
            floor_index=i+2,
            chart_time_s=start + t*pitch,
            episode_time_s=t,
            bpm=180.,
        )
        for i, t in enumerate(times)
    )
    return PlayableChartSegment(
        chart=SimpleNamespace(), start_chart_s=start,
        end_chart_s=start + max(times, default=1.)*pitch + 1.,
        pitch_ratio=pitch, targets=targets,
    )


def test_autoplay_truth_preserves_fractional_physics_tick_and_pitch():
    teacher = AutoplayTimingTeacher.from_segment(
        segment(.123456789, .987654321))
    assert teacher.total_targets == 2
    assert len(teacher.events) == 2
    assert teacher.events[0].episode_time_s == .123456789
    assert teacher.events[0].chart_time_s == pytest.approx(10.+.123456789*1.25)
    assert teacher.future_event(0.).episode_time_s == .123456789
    assert teacher.signed_delta_s(0, .130) == pytest.approx(-.006543211)
    assert teacher.label_at(
        0., current_floor=0, behind_floors=2, ahead_floors=4
    ).relative_time_s == .123456789
    # No rounding to a 10ms or 1ms control/physics tick.
    label = teacher.label_at(
        .000012345, current_floor=0, behind_floors=2, ahead_floors=4)
    assert label.relative_time_s == pytest.approx(.123444444, abs=1e-12)
    assert label.near_200ms and label.visible_floor_window


def test_far_events_are_not_clipped_or_turned_into_motor_commands():
    teacher = AutoplayTimingTeacher.from_segment(segment(.7, 1.01))
    future = teacher.label_at(
        0., current_floor=0, behind_floors=1, ahead_floors=2)
    assert future.relative_time_s == .7
    assert future.near_200ms is False
    assert future.chord_size == 1
    assert not hasattr(future, "key")
    assert not hasattr(future, "motor_force")
    assert teacher.label_at(
        .8, current_floor=0, behind_floors=1, ahead_floors=2
    ).relative_time_s == pytest.approx(.21)
    assert teacher.future_event(1.02) is None
    assert teacher.label_at(
        1.02, current_floor=0, behind_floors=1, ahead_floors=2
    ) is None


def test_simultaneous_input_targets_are_preserved_as_a_chord():
    teacher = AutoplayTimingTeacher.from_segment(
        segment(.50, .50, .50, .71))
    assert teacher.total_targets == 4
    assert len(teacher.events) == 2
    assert teacher.events[0].chord_size == 3
    assert teacher.events[0].target_ordinals == (0, 1, 2)
    assert teacher.events[0].floor_indices == (2, 3, 4)
    label = teacher.label_at(
        .40, current_floor=2, behind_floors=1, ahead_floors=3)
    assert label.chord_size == 3
    assert label.relative_time_s == pytest.approx(.10)
    assert label.visible_floor_window
    assert teacher.future_event(.60).event_index == 1


def test_floors_visibility_is_separate_from_exact_truth():
    teacher = AutoplayTimingTeacher.from_segment(segment(.75))
    hidden = teacher.label_at(
        .70, current_floor=99, behind_floors=1, ahead_floors=2)
    assert hidden is not None
    assert hidden.relative_time_s == pytest.approx(.05)
    assert hidden.near_200ms
    assert not hidden.visible_floor_window


@pytest.mark.parametrize("bad", [
    (),
    (float("nan"),),
    (-.01,),
    (.2, .1),
])
def test_invalid_timeline_fails_closed(bad):
    with pytest.raises(ValueError):
        AutoplayTimingTeacher.from_segment(segment(*bad))


def test_evaluator_oracle_agrees_with_separate_physical_keydowns():
    named = SimpleNamespace(
        segment=segment(.1012345, .456789),
        chart_name="synthetic-train",
        chart_sha256="frozen-digest",
    )
    row = {
        "anchor": 1, "chart": named.chart_name,
        "chart_sha256": named.chart_sha256,
        "final_physics": {
            "hits": 2, "targets": 2, "misses": 0,
            "keydowns": 2, "too_early": 0, "overload": False,
            "unjudged": 0,
        },
        "final_hit_events": [
            {
                "target_index": i, "key": "left_1",
                "time_s": target.episode_time_s + .012,
                "error_ms": 12.,
            }
            for i, target in enumerate(named.segment.targets)
        ],
    }
    result = compare_one_anchor(named, 1, row)
    assert result["targets"] == 2
    assert result["events"] == 2
    assert result["max_independent_target_delta_ms"] < 1e-6
    assert result["timeline"][0]["episode_time_s"] == .1012345
    assert result["timeline"][0]["chord_size"] == 1

    wrong = dict(row, final_hit_events=[
        dict(row["final_hit_events"][0], error_ms=10.),
        row["final_hit_events"][1],
    ])
    with pytest.raises(RuntimeError, match="truth mismatch"):
        compare_one_anchor(named, 1, wrong)


def test_timing_teacher_does_not_modify_frozen_prestige_model():
    from dmdod.training import prestige1 as p

    assert p.FORECAST_WINDOW_S == .200
    assert p.KEY_COUNT == 8 and p.INPUT_DIM == 263
    assert p.STAGE_A_REPORT == "stage_a_v2.json"
    # The new oracle labels are deliberately separate from Stage A's
    # established 0..200ms regression task.
    assert AutoplayTimingTeacher.from_segment(
        segment(.333)).label_at(
        0., current_floor=0, behind_floors=2, ahead_floors=12
    ).relative_time_s == .333
