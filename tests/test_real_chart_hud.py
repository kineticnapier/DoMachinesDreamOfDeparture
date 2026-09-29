from __future__ import annotations

import pytest

from dmdod.adofai_chart import parse_adofai_text
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_rules import TimingJudgement
from dmdod.adofai_timing import compile_adofai
from dmdod.keyboard import KeyEvent
from dmdod.motor_env import TimedKeyEvent
from dmdod.real_chart_hud import HudRealChartMotorEnv, hud_bpms_for_floor


def _segment(*, bpm: float = 120.0, pitch: float = 100.0):
    chart = parse_adofai_text(
        f"""
        {{
          "angleData": [0, 90, 0, 0],
          "settings": {{
            "bpm": {bpm},
            "pitch": {pitch},
            "countdownTicks": 0,
            "separateCountdownTime": false
          }},
          "actions": []
        }}
        """
    )
    compiled = compile_adofai(chart)
    return build_playable_segment(compiled, start_s=0.0, end_s=compiled.duration_s)


def test_hud_reports_tile_and_angle_normalized_real_bpm():
    segment = _segment(bpm=120.0, pitch=100.0)
    angled = next(
        floor
        for floor in segment.chart.floors[1:]
        if floor.exit_time_s - floor.target_time_s - floor.pause_s > 1e-9
        and abs((floor.exit_time_s - floor.target_time_s - floor.pause_s) - 0.5) > 1e-4
    )
    tile_bpm, real_bpm = hud_bpms_for_floor(segment, angled)
    rotation_s = angled.exit_time_s - angled.target_time_s - angled.pause_s

    assert tile_bpm == pytest.approx(120.0)
    assert real_bpm == pytest.approx(60.0 / rotation_s)
    assert real_bpm != pytest.approx(tile_bpm)


def test_hud_bpms_include_pitch_in_wall_clock_speed():
    segment = _segment(bpm=120.0, pitch=150.0)
    floor = segment.chart.floors[1]
    tile_bpm, real_bpm = hud_bpms_for_floor(segment, floor)
    rotation_s = floor.exit_time_s - floor.target_time_s - floor.pause_s

    assert tile_bpm == pytest.approx(180.0)
    if rotation_s > 1e-9:
        assert real_bpm == pytest.approx(60.0 * 1.5 / rotation_s)


def test_hud_timing_feedback_is_transient_and_signed():
    segment = _segment()
    env = HudRealChartMotorEnv(segment, feedback_hold_s=0.75)
    initial = env.reset()
    assert not initial.feedback_visible
    assert initial.last_judgement is None
    assert initial.last_timing_error_ms == 0.0

    target = env.privileged_next_target()
    assert target is not None
    event = TimedKeyEvent(target.episode_time_s + 0.010, "left", KeyEvent.DOWN)
    env._score_event(event)

    visible = env._observation(env.motor.observe(), event.time_s)
    assert visible.feedback_visible
    assert visible.last_judgement is TimingJudgement.PERFECT
    assert visible.last_timing_error_ms == pytest.approx(10.0)

    hidden = env._observation(env.motor.observe(), event.time_s + 0.751)
    assert not hidden.feedback_visible
    assert hidden.last_judgement is None
    assert hidden.last_timing_error_ms == 0.0


def test_hud_does_not_expose_internal_overload_or_target_timestamp_fields():
    observation = HudRealChartMotorEnv(_segment()).reset()
    names = set(observation.__dataclass_fields__)
    assert "tile_bpm" in names
    assert "real_bpm" in names
    assert "last_timing_error_ms" in names
    assert "overload_counter" not in names
    assert "target_time_s" not in names
    assert "chart_time_s" not in names
