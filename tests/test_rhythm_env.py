from dmdod import MotorAction, RhythmMotorEnv, TargetHit, make_regular_targets


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


def test_repeated_too_early_inputs_trigger_overload():
    # Put the first target far enough away that a threshold-reflex spammer can
    # generate three Too Early presses before any legitimate hit window begins.
    env = RhythmMotorEnv([TargetHit(3.0, "left")], bpm=180.0, control_dt_s=0.010)
    observation = env.reset()

    transition = None
    for _ in range(300):
        action = MotorAction(-1.0 if observation.motor.left_pressed else 1.0, 0.0)
        transition = env.step(action)
        observation = transition.observation
        if transition.done:
            break

    assert transition is not None and transition.done
    stats = env.stats
    assert stats.overloaded
    assert stats.overload_counter >= 6
    assert stats.too_early_presses >= 3
