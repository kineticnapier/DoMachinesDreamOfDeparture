from dmdod import MotorAction, RhythmMotorEnv, make_regular_targets


def test_rhythm_observation_does_not_expose_exact_time():
    env = RhythmMotorEnv(make_regular_targets(bpm=120.0, count=2))
    observation = env.reset()

    assert not hasattr(observation, "time_s")
    assert not hasattr(observation, "target_time_s")
    assert not hasattr(observation.motor, "time_s")
    assert 0.0 <= observation.cue.left <= 1.0
    assert 0.0 <= observation.cue.right <= 1.0


def test_rhythm_episode_finishes_and_accounts_for_targets():
    env = RhythmMotorEnv(make_regular_targets(bpm=120.0, count=2), control_dt_s=0.010)
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
