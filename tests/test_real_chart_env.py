from dataclasses import fields

import pytest

from dmdod.adofai_chart import parse_adofai_text
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import compile_adofai
from dmdod.keyboard import KeyEvent
from dmdod.motor_env import TimedKeyEvent
from dmdod.real_chart_env import RealChartMotorEnv
from dmdod.motor.capacity import NKeyRealChartMotorEnv


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



@pytest.mark.parametrize("env_type", (RealChartMotorEnv, NKeyRealChartMotorEnv))
@pytest.mark.parametrize(
    "invalid_tail",
    (
        float("nan"), float("inf"), float("-inf"), True, False,
        "0.4", b"0.4", None, -0.1, 10**1000,
    ),
)
def test_real_chart_rejects_invalid_tail_before_initializing_motor(env_type, invalid_tail):
    segment = build_playable_segment(_compiled(), start_s=0.0, end_s=2.0)
    kwargs = {"key_count": 4} if env_type is NKeyRealChartMotorEnv else {}
    with pytest.raises(ValueError, match="tail_s must be finite and non-negative"):
        env_type(segment, tail_s=invalid_tail, **kwargs)


@pytest.mark.parametrize("env_type", (RealChartMotorEnv, NKeyRealChartMotorEnv))
@pytest.mark.parametrize("valid_tail", (0.0, 0.4, 2, -0.0))
def test_real_chart_preserves_finite_tail_and_episode_end(env_type, valid_tail):
    segment = build_playable_segment(_compiled(), start_s=0.0, end_s=2.0)
    kwargs = {"key_count": 4} if env_type is NKeyRealChartMotorEnv else {}
    env = env_type(segment, tail_s=valid_tail, **kwargs)
    expected_end = max(segment.duration_s, segment.targets[-1].episode_time_s) + valid_tail
    assert env.tail_s == float(valid_tail)
    assert env._episode_end_s == pytest.approx(expected_end)
    assert env.reset().floors



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



def test_fail_on_miss_stops_after_first_expired_real_chart_target():
    from dmdod.envs.rhythm import RewardConfig
    from dmdod.adofai_rules import TimingJudgement

    segment = build_playable_segment(_compiled(), start_s=0.0, end_s=2.0)
    assert len(segment.targets) >= 2
    env = RealChartMotorEnv(segment, reward_config=RewardConfig(fail_on_miss=True))
    env.reset()

    assert env._expire_misses(float("inf")) == -env.reward_config.miss_penalty
    assert env._failed_on_miss
    assert env.stats.misses == env.stats.fail_misses == 1
    assert env.hit_margins == (TimingJudgement.FAIL_MISS,)
    assert env._next_target_index() == 1
    assert env._expire_misses(float("inf")) == 0.0
    assert env.stats.misses == 1
    assert env.hit_margins == (TimingJudgement.FAIL_MISS,)


def test_no_fail_mode_keeps_all_expired_real_chart_targets():
    from dmdod.envs.rhythm import RewardConfig
    from dmdod.adofai_rules import TimingJudgement

    segment = build_playable_segment(_compiled(), start_s=0.0, end_s=2.0)
    env = RealChartMotorEnv(segment, reward_config=RewardConfig(fail_on_miss=False))
    env.reset()
    count = len(segment.targets)
    assert env._expire_misses(float("inf")) == -count * env.reward_config.miss_penalty
    assert not env._failed_on_miss
    assert env.stats.misses == env.stats.fail_misses == count
    assert env.hit_margins == (TimingJudgement.FAIL_MISS,) * count
    assert env._expire_misses(float("inf")) == 0.0


def _speed_change_segment(*, pitch: float = 100.0):
    chart = parse_adofai_text(
        f"""
        {{
          "angleData": [0, 0, 0],
          "settings": {{"bpm": 120, "pitch": {pitch}, "countdownTicks": 0}},
          "actions": [
            {{"floor": 1, "eventType": "SetSpeed", "speedType": "Bpm",
              "beatsPerMinute": 240}}
          ]
        }}
        """
    )
    return build_playable_segment(compile_adofai(chart))


@pytest.mark.parametrize("pitch", [50.0, 100.0, 200.0])
def test_overload_recovery_integrates_floor_speed_boundary(pitch: float):
    segment = _speed_change_segment(pitch=pitch)
    assert segment.chart.floors[0].bpm == pytest.approx(120.0)
    assert segment.chart.floors[1].bpm == pytest.approx(240.0)
    boundary = segment.episode_time_from_chart(
        segment.chart.floors[1].target_time_s
    )
    env = RealChartMotorEnv(segment)
    env.reset()
    start_s, end_s = boundary - 0.001, boundary + 0.001
    expected_beats = (0.001 * 120.0 + 0.001 * 240.0) * segment.pitch_ratio / 60.0
    assert env._beat_delta_between(start_s, end_s) == pytest.approx(
        expected_beats, abs=1e-12
    )

    env._overload_clock_s = start_s
    env._advance_overload(end_s)
    assert env._overload.multipress_reset_beats == pytest.approx(
        expected_beats, abs=1e-12
    )
    assert env._overload_clock_s == end_s


def test_overload_floor_boundary_one_step_equals_split_steps():
    segment = _speed_change_segment()
    boundary = segment.episode_time_from_chart(
        segment.chart.floors[1].target_time_s
    )
    full = RealChartMotorEnv(segment)
    split = RealChartMotorEnv(segment)
    for env in (full, split):
        env.reset()
        env._overload_clock_s = boundary - 0.01
        env._overload.value = 2.0
        env._overload.multipress_value = 2.0

    full._advance_overload(boundary + 0.01)
    split._advance_overload(boundary)
    split._advance_overload(boundary + 0.01)

    assert full._overload.value == pytest.approx(split._overload.value)
    assert full._overload.multipress_value == pytest.approx(
        split._overload.multipress_value
    )
    assert full._overload.multipress_reset_beats == pytest.approx(
        split._overload.multipress_reset_beats
    )


def test_overload_constant_bpm_and_equal_time_midspin_preserve_beats():
    segment = build_playable_segment(_compiled(), start_s=0.0, end_s=2.0)
    env = RealChartMotorEnv(segment)
    env.reset()
    # The fixture has a zero-duration Midspin and equal-time floor entries.
    assert segment.chart.floors[1].target_time_s == pytest.approx(
        segment.chart.floors[2].target_time_s
    )
    assert env._beat_delta_between(0.1, 0.9) == pytest.approx(
        (0.9 - 0.1) * 120.0 / 60.0
    )


def test_overload_six_beat_reset_at_floor_speed_boundary():
    segment = _speed_change_segment()
    boundary = segment.episode_time_from_chart(
        segment.chart.floors[1].target_time_s
    )
    env = RealChartMotorEnv(segment)
    env.reset()
    env._overload_clock_s = boundary - 0.001
    env._overload.multipress_reset_beats = (
        env._overload.multipress_reset_after_beats - 0.001
    )
    env._overload.multipress_value = 1.0

    env._advance_overload(boundary + 0.001)

    assert env._overload.multipress_value == 0.0
    assert env._overload.multipress_reset_beats == 0.0
