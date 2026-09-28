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
