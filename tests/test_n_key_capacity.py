from __future__ import annotations

import pytest

from dmdod.four_key_motor import FourKeyAction, FourKeyMotorEnv
from dmdod.n_key_capacity import CenterFirstNKeyTeacher, calibrate_n_key_press_lead
from dmdod.n_key_motor import NKeyAction, NKeyMotorEnv, n_key_names, n_key_tiers
from dmdod.motor.n_key import NKeyBody


def test_n_key_layout_is_center_symmetric() -> None:
    assert n_key_names(4) == ("left_2", "left_1", "right_1", "right_2")
    assert n_key_names(6) == (
        "left_3",
        "left_2",
        "left_1",
        "right_1",
        "right_2",
        "right_3",
    )
    assert n_key_tiers(8) == (
        ("left_1", "right_1"),
        ("left_2", "right_2"),
        ("left_3", "right_3"),
        ("left_4", "right_4"),
    )


@pytest.mark.parametrize(
    "bad",
    (4.0, 4.9, True, False, "4", None, float("nan"), float("inf")),
)
@pytest.mark.parametrize("factory", (n_key_names, n_key_tiers, NKeyBody, NKeyMotorEnv))
def test_n_key_count_rejects_implicit_integer_conversion(factory, bad) -> None:
    with pytest.raises(ValueError, match="key_count must be an even integer >= 2"):
        factory(bad)


@pytest.mark.parametrize("count", (2, 4, 8))
def test_n_key_count_preserves_valid_integer_layout_and_body(count: int) -> None:
    names = n_key_names(count)
    assert len(names) == count
    assert len(n_key_tiers(count)) == count // 2
    assert NKeyBody(count).key_names == names
    assert NKeyMotorEnv(count).key_names == names


@pytest.mark.parametrize("bad", [0, 1, 3, 5])
def test_n_key_layout_rejects_non_even_counts(bad: int) -> None:
    with pytest.raises(ValueError):
        n_key_names(bad)


def test_generic_4k_motor_matches_existing_four_key_body() -> None:
    old = FourKeyMotorEnv(control_dt_s=0.010, physics_dt_s=0.001)
    new = NKeyMotorEnv(4, control_dt_s=0.010, physics_dt_s=0.001)
    old.reset()
    new.reset()

    patterns = [
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0, 1.0),
        (-1.0, -1.0, 1.0, 1.0),
        (-1.0, -1.0, -1.0, -1.0),
        (0.0, 0.0, 0.0, 0.0),
    ]

    old_to_new = {
        "left_outer": "left_2",
        "left_inner": "left_1",
        "right_inner": "right_1",
        "right_outer": "right_2",
    }

    for step_index in range(120):
        values = patterns[(step_index // 20) % len(patterns)]
        old_step = old.step(FourKeyAction(*values))
        new_step = new.step(NKeyAction(values))

        old_obs = old_step.observation
        new_obs = new_step.observation
        assert new_obs.positions_m == pytest.approx(
            (
                old_obs.left_outer_position_m,
                old_obs.left_inner_position_m,
                old_obs.right_inner_position_m,
                old_obs.right_outer_position_m,
            ),
            abs=1e-12,
        )
        assert new_obs.velocities_m_s == pytest.approx(
            (
                old_obs.left_outer_velocity_m_s,
                old_obs.left_inner_velocity_m_s,
                old_obs.right_inner_velocity_m_s,
                old_obs.right_outer_velocity_m_s,
            ),
            abs=1e-12,
        )
        assert new_obs.pressed_flags == (
            old_obs.left_outer_pressed,
            old_obs.left_inner_pressed,
            old_obs.right_inner_pressed,
            old_obs.right_outer_pressed,
        )
        assert [
            (old_to_new[event.key], event.event, event.time_s)
            for event in old_step.evaluator_events
        ] == [
            (event.key, event.event, event.time_s)
            for event in new_step.evaluator_events
        ]


def test_center_first_n_key_teacher_reserves_from_inner_pair_outward() -> None:
    env = NKeyMotorEnv(8)
    observation = env.reset()
    teacher = CenterFirstNKeyTeacher(8)
    targets = tuple((index, 0.0) for index in range(6))

    action = teacher.pipeline_action(
        observation,
        now_s=0.0,
        targets=targets,
        lead_s=0.05,
    )
    commands = dict(zip(observation.key_names, action.values))

    assert commands["left_1"] == 1.0
    assert commands["right_1"] == 1.0
    assert commands["left_2"] == 1.0
    assert commands["right_2"] == 1.0
    assert commands["left_3"] == 1.0
    assert commands["right_3"] == 1.0
    assert commands["left_4"] == 0.0
    assert commands["right_4"] == 0.0


def test_n_key_motor_physics_trace_is_opt_in_and_records_boundary_events() -> None:
    env = NKeyMotorEnv(4, control_dt_s=0.010, physics_dt_s=0.001)
    env.reset()

    plain = env.step(NKeyAction((0.0, 1.0, 0.0, 0.0)))
    assert plain.physics_samples == ()
    assert env.last_transition is plain

    env.reset()
    env.capture_physics_trace = True
    action = NKeyAction((0.0, 1.0, 0.0, 0.0))
    transition = None
    for _ in range(10):
        transition = env.step(action)
        if transition.evaluator_events:
            break

    assert transition is not None
    assert transition.physics_samples
    assert len(transition.physics_samples) == env.physics_substeps
    assert env.last_transition is transition

    traced_events = [
        (sample.time_s, key, event, sample.positions_m)
        for sample in transition.physics_samples
        for key, event in sample.events
    ]
    assert traced_events
    event_time, key, event, positions = traced_events[0]
    assert event_time > 0.0
    assert key == "left_1"
    assert event.value == "down"
    key_index = env.key_names.index(key)
    assert positions[key_index] >= env.keyboard.config.actuation_m


def test_n_key_calibration_matches_existing_4k_single_press_latency() -> None:
    calibration = calibrate_n_key_press_lead(4)
    assert calibration.lead_s == pytest.approx(0.044, abs=1e-12)
    assert dict(calibration.key_latencies_s) == pytest.approx(
        {
            "left_2": 0.039,
            "left_1": 0.039,
            "right_1": 0.039,
            "right_2": 0.039,
        },
        abs=1e-12,
    )

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("axis", [0, 7])
@pytest.mark.parametrize("trace", [False, True])
def test_nonfinite_n_key_action_rejected_before_physics(bad, axis, trace):
    env = NKeyMotorEnv(8)
    before = env.reset()
    before_diagnostics = env.diagnostics()
    env.capture_physics_trace = trace
    commands = [0.0] * 8
    commands[axis] = bad

    with pytest.raises(ValueError, match="finite"):
        env.step(NKeyAction(tuple(commands)))

    assert env.observe() == before
    assert env.diagnostics() == before_diagnostics
    assert env.last_transition is None


def test_due_target_window_keeps_valid_future_reservations_without_tail_copy() -> None:
    from types import SimpleNamespace
    from dmdod.motor.capacity import _due_target_window, _target_token

    targets = tuple(
        SimpleNamespace(ordinal=i, floor_index=i + 1, episode_time_s=i * 0.1)
        for i in range(8)
    )
    segment = SimpleNamespace(targets=targets)
    reservations = {
        "left_1": (_target_token(targets[6]), targets[6].episode_time_s),
        "right_1": (_target_token(targets[1]), targets[1].episode_time_s),
        "left_2": ((99, 99, 99.0), 99.0),
    }
    result = _due_target_window(segment, 2, 4, reservations)
    assert result == (
        (_target_token(targets[2]), targets[2].episode_time_s),
        (_target_token(targets[3]), targets[3].episode_time_s),
        (_target_token(targets[6]), targets[6].episode_time_s),
    )
    assert _due_target_window(segment, 2, 2, reservations) == (
        (_target_token(targets[6]), targets[6].episode_time_s),
    )


def test_compact_due_window_matches_legacy_teacher_and_motor_step_by_step() -> None:
    from bisect import bisect_right
    from types import SimpleNamespace
    from dmdod.motor.capacity import _due_target_window, _target_token

    times = (0.10, 0.10, 0.16, 0.18, 0.25, 0.26, 0.40, 0.50)
    targets = tuple(
        SimpleNamespace(ordinal=i, floor_index=i + 1, episode_time_s=time_s)
        for i, time_s in enumerate(times)
    )
    segment = SimpleNamespace(targets=targets)
    old_motor = NKeyMotorEnv(4)
    new_motor = NKeyMotorEnv(4)
    old_observation = old_motor.reset()
    new_observation = new_motor.reset()
    old_teacher = CenterFirstNKeyTeacher(4)
    new_teacher = CenterFirstNKeyTeacher(4)
    future = (_target_token(targets[-1]), times[-1])
    old_teacher._reservations["left_2"] = future
    new_teacher._reservations["left_2"] = future
    lead_s = 0.044
    start = 0

    for frame in range(70):
        now_s = frame * 0.010
        while start < len(times) and times[start] < now_s - 0.06:
            start += 1
        legacy = tuple(
            (_target_token(target), float(target.episode_time_s))
            for target in targets[start:]
        )
        due_end = bisect_right(times, now_s + lead_s, lo=start)
        compact = _due_target_window(segment, start, due_end, new_teacher._reservations)
        old_due = [token for token, target_time in legacy if now_s + lead_s >= target_time]
        new_due = [token for token, _ in compact[: due_end - start]]
        assert old_due == new_due

        old_action = old_teacher.pipeline_action(
            old_observation, now_s=now_s, targets=legacy, lead_s=lead_s
        )
        new_action = new_teacher.pipeline_action(
            new_observation, now_s=now_s, targets=compact, lead_s=lead_s
        )
        assert new_action == old_action
        assert new_teacher._reservations == old_teacher._reservations
        old_transition = old_motor.step(old_action)
        new_transition = new_motor.step(new_action)
        assert new_transition.evaluator_events == old_transition.evaluator_events
        assert new_transition.observation == old_transition.observation
        old_observation = old_transition.observation
        new_observation = new_transition.observation


def test_capacity_result_matches_legacy_full_tail(monkeypatch) -> None:
    from dmdod.adofai_chart import parse_adofai_text
    from dmdod.adofai_playable import build_playable_segment
    from dmdod.adofai_timing import compile_adofai
    from dmdod.motor import capacity
    from dmdod.motor.capacity import _target_token, diagnose_n_key_teacher_capacity

    chart = parse_adofai_text(
        '{"angleData":[0,90,90,90,90,90,90],'
        '"settings":{"bpm":120,"pitch":100,"countdownTicks":0,'
        '"separateCountdownTime":false},"actions":[]}'
    )
    segment = build_playable_segment(compile_adofai(chart), start_s=0.0, end_s=4.0)
    assert segment.targets
    optimized = capacity._due_target_window

    def legacy_window(segment, start, due_end, reservations):
        return tuple(
            (_target_token(target), float(target.episode_time_s))
            for target in segment.targets[start:]
        )

    monkeypatch.setattr(capacity, "_due_target_window", legacy_window)
    expected = diagnose_n_key_teacher_capacity(
        segment, key_count=4, lead_s=0.044, control_dt_s=0.010
    )
    monkeypatch.setattr(capacity, "_due_target_window", optimized)
    actual = diagnose_n_key_teacher_capacity(
        segment, key_count=4, lead_s=0.044, control_dt_s=0.010
    )
    assert actual == expected


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
@pytest.mark.parametrize("field", ("control_dt_s", "physics_dt_s"))
def test_n_key_env_rejects_nonfinite_clock_configuration(field, bad) -> None:
    with pytest.raises(ValueError, match=rf"{field} must be finite"):
        NKeyMotorEnv(4, **{field: bad})


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
@pytest.mark.parametrize("axis", (0, 3))
def test_n_key_body_rejects_nonfinite_commands_without_mutation(bad, axis) -> None:
    # Direct body callers must receive the same fail-closed protection as
    # NKeyMotorEnv.step; a bad command must not alter physical state.
    body = NKeyBody(4)
    before_states = tuple(state.copy() for state in body.states)
    before_hands = (
        body.left_hand_state.fatigue,
        body.left_hand_state.coordination,
        body.right_hand_state.fatigue,
        body.right_hand_state.coordination,
        body.bilateral_state.coordination,
    )
    commands = [0.0] * 4
    commands[axis] = bad

    with pytest.raises(ValueError, match="N-key motor commands must be finite"):
        body.step(NKeyAction(tuple(commands)), 0.001)

    assert tuple(body.states) == before_states
    assert (
        body.left_hand_state.fatigue,
        body.left_hand_state.coordination,
        body.right_hand_state.fatigue,
        body.right_hand_state.coordination,
        body.bilateral_state.coordination,
    ) == before_hands


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_n_key_body_rejects_nonfinite_dt_without_mutation(bad) -> None:
    body = NKeyBody(4)
    before_states = tuple(state.copy() for state in body.states)
    before_hands = (
        body.left_hand_state.fatigue,
        body.left_hand_state.coordination,
        body.right_hand_state.fatigue,
        body.right_hand_state.coordination,
        body.bilateral_state.coordination,
    )
    with pytest.raises(ValueError, match="dt_s must be finite and positive"):
        body.step(NKeyAction((0.0,) * 4), bad)
    assert tuple(body.states) == before_states
    assert (
        body.left_hand_state.fatigue,
        body.left_hand_state.coordination,
        body.right_hand_state.fatigue,
        body.right_hand_state.coordination,
        body.bilateral_state.coordination,
    ) == before_hands


def test_n_key_finite_clock_configuration_preserves_substeps() -> None:
    env = NKeyMotorEnv(4, control_dt_s=0.010, physics_dt_s=0.001)
    assert env.physics_substeps == 10
    before = env.reset()
    assert env.step(NKeyAction((0.0,) * 4)).observation == before


def test_legacy_oracle_tail_remains_available_for_stage_t() -> None:
    """Stage T still imports the original privileged oracle helper."""
    from types import SimpleNamespace
    from dmdod.motor.capacity import _target_token, _unresolved_tail

    targets = tuple(
        SimpleNamespace(ordinal=i, floor_index=i + 1, episode_time_s=i * 0.1)
        for i in range(4)
    )
    segment = SimpleNamespace(targets=targets)
    env = SimpleNamespace(
        segment=segment,
        privileged_next_target=lambda: targets[2],
    )
    assert _unresolved_tail(env) == tuple(
        (_target_token(target), float(target.episode_time_s))
        for target in targets[2:]
    )
    empty_env = SimpleNamespace(
        segment=segment,
        privileged_next_target=lambda: None,
    )
    assert _unresolved_tail(empty_env) == ()


@pytest.mark.parametrize("bad", [True, False, "1", b"1", None, 10**1000])
def test_n_key_rejects_non_numeric_commands_without_state_mutation(bad) -> None:
    # Previously bool/str/bytes became a real key press; huge integers raised
    # OverflowError instead of the documented invalid-command ValueError.
    commands = [0.0] * 4
    commands[1] = bad
    action = NKeyAction(tuple(commands))

    body = NKeyBody(4)
    before_states = tuple(state.copy() for state in body.states)
    before_hands = (
        body.left_hand_state.fatigue,
        body.left_hand_state.coordination,
        body.right_hand_state.fatigue,
        body.right_hand_state.coordination,
        body.bilateral_state.coordination,
    )
    with pytest.raises(ValueError, match="N-key motor commands must be numeric"):
        body.step(action, 0.001)
    assert tuple(body.states) == before_states
    assert (
        body.left_hand_state.fatigue,
        body.left_hand_state.coordination,
        body.right_hand_state.fatigue,
        body.right_hand_state.coordination,
        body.bilateral_state.coordination,
    ) == before_hands

    for trace in (False, True):
        env = NKeyMotorEnv(4)
        env.capture_physics_trace = trace
        before_observation = env.observe()
        before_diagnostics = env.diagnostics()
        with pytest.raises(ValueError, match="N-key motor commands must be numeric"):
            env.step(action)
        assert env.observe() == before_observation
        assert env.diagnostics() == before_diagnostics
        assert env.last_transition is None


def test_n_key_numeric_commands_still_convert_to_float() -> None:
    assert NKeyAction((1, -1, 0.5, 0)).as_tuple() == (1.0, -1.0, 0.5, 0.0)


@pytest.mark.parametrize("bad", (True, False, "0.001", b"0.001", None, 10**1000))
def test_n_key_body_rejects_malformed_dt_without_mutation(bad) -> None:
    # isfinite(True) used to pass; isfinite(huge_int) raised OverflowError.
    body = NKeyBody(4)
    before_states = tuple(state.copy() for state in body.states)
    before_hands = (
        body.left_hand_state.fatigue,
        body.left_hand_state.coordination,
        body.right_hand_state.fatigue,
        body.right_hand_state.coordination,
        body.bilateral_state.coordination,
    )
    with pytest.raises(ValueError, match="dt_s must be finite and positive"):
        body.step(NKeyAction((0.0,) * 4), bad)
    assert tuple(body.states) == before_states
    assert (
        body.left_hand_state.fatigue,
        body.left_hand_state.coordination,
        body.right_hand_state.fatigue,
        body.right_hand_state.coordination,
        body.bilateral_state.coordination,
    ) == before_hands


@pytest.mark.parametrize("field", ("control_dt_s", "physics_dt_s"))
@pytest.mark.parametrize("bad", (True, False, "0.001", b"0.001", None, 10**1000))
def test_n_key_env_rejects_malformed_clock_configuration(field, bad) -> None:
    with pytest.raises(ValueError, match=rf"{field} must be finite"):
        NKeyMotorEnv(4, **{field: bad})


def test_n_key_env_accepts_finite_integer_clock_configuration() -> None:
    # Integer numeric values remain accepted; only malformed clocks are rejected.
    env = NKeyMotorEnv(4, control_dt_s=1, physics_dt_s=1)
    assert env.physics_substeps == 1
