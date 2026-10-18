from __future__ import annotations

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.real_chart import RealChartMotorEnv


def _env() -> RealChartMotorEnv:
    chart = parse_adofai_text(
        '{"angleData":[0,180,0,180,0],'
        '"settings":{"bpm":120,"pitch":100,"countdownTicks":0},'
        '"actions":[]}'
    )
    return RealChartMotorEnv(build_playable_segment(compile_adofai(chart)))


def test_resolution_cursor_preserves_ordered_expiration_and_reward():
    env = _env()
    env.reset()
    first = env.segment.targets[0]
    just_before = first.episode_time_s + env._windows(first).pass_s
    assert env._expire_misses(just_before) == 0.0
    assert env._next_target_index() == 0

    reward = env._expire_misses(just_before + 1e-9)
    assert reward == -env.reward_config.miss_penalty
    assert env._missed[0] is True
    assert env._next_target_index() == 1


def test_resolution_cursor_resynchronizes_after_external_resolution():
    env = _env()
    env.reset()
    env._used[0] = True
    env._missed[1] = True
    assert env._next_target_index() == 2
