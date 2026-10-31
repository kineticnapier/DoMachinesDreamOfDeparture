from __future__ import annotations

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.real_chart import RealChartMotorEnv
from dmdod.envs.contact_chart_16k import CoupledContactRealChartEnv
from dmdod.motor.capacity import NKeyRealChartMotorEnv
from dmdod.motor.contact_16k import ContactAction
from dmdod.motor.env import MotorAction
from dmdod.motor.n_key import NKeyAction

import pytest


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

def _motor_variant(kind: str):
    segment = _env().segment
    if kind == "2k":
        return RealChartMotorEnv(segment), MotorAction(0.0, 0.0)
    if kind == "8k":
        return NKeyRealChartMotorEnv(segment, key_count=8), NKeyAction((0.0,) * 8)
    if kind == "16contact":
        return CoupledContactRealChartEnv(segment), ContactAction((0.0,) * 12)
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["2k", "8k", "16contact"])
def test_terminal_resolution_uses_shared_cursor_for_all_real_chart_motors(kind):
    env, neutral = _motor_variant(kind)
    env.reset()
    count = len(env.segment.targets)
    assert count >= 3
    assert env._next_target_index() == 0

    # A resolved prefix must not hide a later unresolved target.
    for index in range(count - 1):
        if index % 2:
            env._missed[index] = True
        else:
            env._used[index] = True
        assert env._all_targets_resolved() == all(
            used or missed for used, missed in zip(env._used, env._missed)
        )
    assert env._next_target_index() == count - 1
    before = env.stats
    running = env.step(neutral)
    assert not running.done
    assert running.reward == pytest.approx(0.0)
    assert env.stats == before

    # Both used and missed targets count as resolved; a neutral step must
    # terminate without adding a phantom Miss or changing the reward.
    env._missed[-1] = True
    before = env.stats
    terminal = env.step(neutral)
    assert terminal.done
    assert terminal.reward == pytest.approx(0.0)
    assert env.stats == before
    assert env._next_unresolved_cursor == count

    # A new episode must not inherit the previous terminal cursor.
    env.reset()
    assert env._next_unresolved_cursor == 0
    assert not env._all_targets_resolved()
    assert not env.step(neutral).done


@pytest.mark.parametrize("kind", ["2k", "8k", "16contact"])
def test_all_resolved_cursor_matches_full_scan_for_each_monotone_update(kind):
    env, _ = _motor_variant(kind)
    env.reset()
    for index in range(len(env.segment.targets)):
        if index % 2:
            env._missed[index] = True
        else:
            env._used[index] = True
        expected = all(used or missed for used, missed in zip(env._used, env._missed))
        assert env._all_targets_resolved() is expected
        assert env._next_unresolved_cursor == index + 1
