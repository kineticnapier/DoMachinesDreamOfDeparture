import pytest

from dmdod import (
    KeyEvent,
    MotorAction,
    RhythmMotorEnv,
    TargetHit,
    TimedKeyEvent,
    TimingJudgement,
    make_regular_targets,
)


def test_rhythm_observation_does_not_expose_exact_time():
    env = RhythmMotorEnv(make_regular_targets(bpm=120.0, count=2), bpm=120.0)
    observation = env.reset()

    assert not hasattr(observation, "time_s")
    assert not hasattr(observation, "target_time_s")
    assert not hasattr(observation.motor, "time_s")
    assert 0.0 <= observation.cue.left <= 1.0
    assert 0.0 <= observation.cue.right <= 1.0


def test_rhythm_episode_finishes_and_accounts_for_targets():
    env = RhythmMotorEnv(
        make_regular_targets(bpm=120.0, count=2),
        bpm=120.0,
        control_dt_s=0.010,
    )
    env.reset()

    transition = None
    for _ in range(300):
        transition = env.step(MotorAction(0.0, 0.0))
        if transition.done:
            break

    assert transition is not None and transition.done
    stats = env.stats
    assert stats.targets == 2
    assert stats.hits + stats.misses == 2
    assert not stats.overloaded
    assert stats.fail_misses == 2
    assert stats.hit_margin_count == 2
    assert stats.x_accuracy_percent == pytest.approx(0.0)
    assert env.hit_margins == (
        TimingJudgement.FAIL_MISS,
        TimingJudgement.FAIL_MISS,
    )


def test_third_undecayed_too_early_is_replaced_by_fail_overload():
    env = RhythmMotorEnv([TargetHit(3.0, "left")], bpm=180.0, control_dt_s=0.010)
    env.reset()

    # Feed three simultaneous privileged evaluator events directly so the test
    # isolates DLL fail-bar semantics from body press/release timing. At 0.0 s
    # all three are unambiguously TooEarly and no song-time recovery occurs.
    event = TimedKeyEvent(0.0, "left", KeyEvent.DOWN)
    assert env._score_event(event) < 0.0
    assert env._score_event(event) < 0.0
    assert env._score_event(event) < 0.0

    stats = env.stats
    assert stats.overloaded
    assert stats.overload_counter == pytest.approx(1.5)
    assert stats.too_early_presses == 3
    assert stats.fail_overloads == 1
    assert env.hit_margins == (
        TimingJudgement.TOO_EARLY,
        TimingJudgement.TOO_EARLY,
        TimingJudgement.FAIL_OVERLOAD,
    )
    # XAcc: (0.2 + 0.2 + 0.0) / 3.
    assert stats.x_accuracy_percent == pytest.approx(100.0 * 0.4 / 3.0)



def test_fail_on_miss_stops_after_first_expired_rhythm_target():
    from dmdod.envs.rhythm import RewardConfig

    env = RhythmMotorEnv(
        make_regular_targets(bpm=120.0, count=3),
        bpm=120.0,
        reward_config=RewardConfig(fail_on_miss=True),
    )
    env.reset()
    assert env._expire_misses(float("inf")) == -env.reward_config.miss_penalty
    assert env._failed_on_miss
    assert env.stats.misses == env.stats.fail_misses == 1
    assert env.hit_margins == (TimingJudgement.FAIL_MISS,)
    assert env._expire_misses(float("inf")) == 0.0
    assert env.stats.misses == 1
    assert env.hit_margins == (TimingJudgement.FAIL_MISS,)


def test_no_fail_mode_keeps_all_expired_rhythm_targets():
    from dmdod.envs.rhythm import RewardConfig

    env = RhythmMotorEnv(
        make_regular_targets(bpm=120.0, count=3),
        bpm=120.0,
        reward_config=RewardConfig(fail_on_miss=False),
    )
    env.reset()
    assert env._expire_misses(float("inf")) == -3 * env.reward_config.miss_penalty
    assert not env._failed_on_miss
    assert env.stats.misses == env.stats.fail_misses == 3
    assert env.hit_margins == (TimingJudgement.FAIL_MISS,) * 3
    assert env._expire_misses(float("inf")) == 0.0


def test_rhythm_resolution_cursors_match_full_scan_for_interleaved_keys():
    env = RhythmMotorEnv(
        [TargetHit(0.5, "left"), TargetHit(0.6, "right"),
         TargetHit(0.8, "left"), TargetHit(0.9, "right")],
        bpm=120.0,
    )
    env.reset()

    def check_cursors():
        expected = next(
            (i for i in range(len(env._targets))
             if not env._used[i] and not env._missed[i]),
            None,
        )
        assert env._next_unresolved_target_index() == expected
        for key in ("left", "right", "absent"):
            expected_key = next(
                (i for i, target in enumerate(env._targets)
                 if target.key == key and not env._used[i] and not env._missed[i]),
                None,
            )
            assert env._next_target_index(key) == expected_key
        assert env._remaining == sum(
            not used and not missed for used, missed in zip(env._used, env._missed)
        )

    check_cursors()
    assert env._score_event(TimedKeyEvent(0.5, "left", KeyEvent.DOWN)) > 0
    check_cursors()
    # TooEarly is non-consuming; the next left target remains selectable.
    assert env._score_event(TimedKeyEvent(0.6, "left", KeyEvent.DOWN)) < 0
    check_cursors()
    assert env._expire_misses(0.78) == -env.reward_config.miss_penalty
    assert env._expire_misses(0.78) == 0
    check_cursors()
    assert env._score_event(TimedKeyEvent(0.8, "left", KeyEvent.DOWN)) > 0
    check_cursors()
    assert env._score_event(TimedKeyEvent(0.9, "right", KeyEvent.DOWN)) > 0
    check_cursors()
    assert env._remaining == 0
    assert env.stats.hits == 3
    assert env.stats.misses == 1
    env.reset()
    check_cursors()
    assert env._remaining == 4


def test_rhythm_miss_cursor_respects_fail_mode_and_reset():
    from dmdod.envs.rhythm import RewardConfig

    targets = [TargetHit(0.5, "left"), TargetHit(0.6, "right"),
               TargetHit(0.7, "left")]
    fail_env = RhythmMotorEnv(
        targets, bpm=120.0, reward_config=RewardConfig(fail_on_miss=True),
    )
    fail_env.reset()
    assert fail_env._expire_misses(float("inf")) == -fail_env.reward_config.miss_penalty
    assert fail_env._remaining == 2
    assert fail_env._failed_on_miss
    assert fail_env._expire_misses(float("inf")) == 0
    assert fail_env.stats.misses == 1
    fail_env.reset()
    assert fail_env._remaining == 3
    assert fail_env._next_unresolved_target_index() == 0

    no_fail = RhythmMotorEnv(
        targets, bpm=120.0, reward_config=RewardConfig(fail_on_miss=False),
    )
    no_fail.reset()
    assert no_fail._expire_misses(float("inf")) == -3 * no_fail.reward_config.miss_penalty
    assert no_fail._remaining == 0
    assert no_fail._next_unresolved_target_index() is None
    assert no_fail._expire_misses(float("inf")) == 0
    assert no_fail.hit_margins == (TimingJudgement.FAIL_MISS,) * 3


def test_rhythm_long_target_list_is_consumed_once_by_miss_cursor():
    targets = [TargetHit(0.5 + 0.01 * i, "left" if i % 2 else "right")
               for i in range(2000)]
    env = RhythmMotorEnv(targets, bpm=120.0)
    env.reset()
    assert env._expire_misses(float("inf")) == -2000 * env.reward_config.miss_penalty
    assert env._remaining == 0
    assert env._miss_scan_cursor == 2000
    assert env._next_unresolved_target_index() is None
    assert env._next_target_index("left") is None
    assert env._next_target_index("right") is None
    assert env._expire_misses(float("inf")) == 0
    env.reset()
    assert env._remaining == 2000
    assert env._miss_scan_cursor == 0
