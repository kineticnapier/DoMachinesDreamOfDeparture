from dataclasses import fields

import pytest

from dmdod.adofai_chart import parse_adofai_text
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import compile_adofai
from dmdod.keyboard import KeyEvent
from dmdod.motor_env import TimedKeyEvent
from dmdod.real_chart_env import RealChartMotorEnv


def _compiled(*, pitch: float = 100.0):
    chart = parse_adofai_text(
        f"""
        {{
          "angleData": [0, 999, 0],
          "settings": {{
            "bpm": 120,
            "pitch": {pitch},
            "countdownTicks": 0,
            "separateCountdownTime": false
          }},
          "actions": [
            {{"floor": 2, "eventType": "Twirl"}}
          ]
        }}
        """
    )
    return compile_adofai(chart)


def test_playable_targets_exclude_start_floor_and_midspin_but_keep_following_equal_time_floor():
    compiled = _compiled()
    assert compiled.floors[1].midspin is True
    assert compiled.floors[1].target_time_s == pytest.approx(compiled.floors[2].target_time_s)

    segment = build_playable_segment(compiled, start_s=0.0, end_s=2.0)
    assert [target.floor_index for target in segment.targets] == [2, 3]
    assert segment.targets[0].chart_time_s == pytest.approx(0.5, abs=2e-5)
    assert segment.targets[0].episode_time_s == pytest.approx(0.5, abs=2e-5)


def test_playable_target_episode_time_respects_chart_pitch():
    compiled = _compiled(pitch=200.0)
    segment = build_playable_segment(compiled, start_s=0.0, end_s=2.0)
    assert segment.pitch_ratio == pytest.approx(2.0)
    assert segment.targets[0].chart_time_s == pytest.approx(0.5, abs=2e-5)
    assert segment.targets[0].episode_time_s == pytest.approx(0.25, abs=2e-5)
    assert segment.chart_time_from_episode(0.25) == pytest.approx(0.5)


def test_real_chart_policy_observation_contains_geometry_not_privileged_timing():
    segment = build_playable_segment(_compiled(), start_s=0.0, end_s=2.0)
    env = RealChartMotorEnv(segment, ahead_floors=3)
    observation = env.reset()

    observation_names = {field.name for field in fields(observation)}
    assert observation_names == {"motor", "orbiting_x", "orbiting_y", "floors"}
    assert observation.floors

    floor_names = {field.name for field in fields(observation.floors[0])}
    assert "target_time_s" not in floor_names
    assert "bpm" not in floor_names
    assert "index" not in floor_names
    assert "relative_index" in floor_names
    assert any(floor.midspin for floor in observation.floors)


def test_real_chart_any_physical_key_can_claim_next_playable_target():
    segment = build_playable_segment(_compiled(), start_s=0.0, end_s=2.0)
    env = RealChartMotorEnv(segment)
    env.reset()
    target = env.privileged_next_target()
    assert target is not None

    reward = env._score_event(
        TimedKeyEvent(target.episode_time_s, "right", KeyEvent.DOWN)
    )
    assert reward > 0.0
    assert env.stats.hits == 1
    assert env.stats.misses == 0
    assert env.privileged_next_target() is not None


def test_segment_rebases_target_times_without_changing_chart_truth():
    compiled = _compiled()
    start = compiled.floors[2].target_time_s
    end = compiled.floors[3].target_time_s
    segment = build_playable_segment(compiled, start_s=start, end_s=end)

    assert segment.targets[0].floor_index == 2
    assert segment.targets[0].chart_time_s == pytest.approx(start)
    assert segment.targets[0].episode_time_s == pytest.approx(0.0)
    assert segment.targets[-1].chart_time_s == pytest.approx(end)
