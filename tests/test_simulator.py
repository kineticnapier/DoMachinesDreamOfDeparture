import pytest

from dmdod import BodyConfig, FingerConfig, KeyEvent, Simulation, TwoFingerBody
from dmdod.envs.simulator import SimulationConfig


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_simulation_config_rejects_nonfinite_dt(bad):
    with pytest.raises(ValueError, match="dt_s must be finite and positive"):
        SimulationConfig(dt_s=bad)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_direct_body_step_rejects_nonfinite_dt_without_mutating_state(bad):
    body = TwoFingerBody()
    body.reset()
    left_before = body.left.copy()
    right_before = body.right.copy()
    hand_before = (body.shared_hand.fatigue, body.shared_hand.coordination)
    bilateral_before = body.bilateral_state.coordination

    with pytest.raises(ValueError, match="dt_s must be finite and positive"):
        body.step(1.0, 0.0, bad)

    assert body.left == left_before
    assert body.right == right_before
    assert (body.shared_hand.fatigue, body.shared_hand.coordination) == hand_before
    assert body.bilateral_state.coordination == bilateral_before


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_run_constant_rejects_nonfinite_duration_without_advancing(bad):
    sim = Simulation()
    with pytest.raises(ValueError, match="duration_s must be finite and non-negative"):
        sim.run_constant(1.0, 0.0, bad)
    assert sim.time_s == 0.0
    assert sim.body.left.position_m == 0.0
    assert sim.body.right.position_m == 0.0


def test_finite_time_boundaries_preserve_existing_behavior():
    assert SimulationConfig(dt_s=0.001).dt_s == 0.001
    sim = Simulation()
    assert sim.run_constant(0.0, 0.0, 0.0) == []
    assert sim.step(0.0, 0.0).time_s == pytest.approx(0.001)



@pytest.mark.parametrize(
    "bad", (True, False, "0.001", b"0.001", None, 10**1000)
)
def test_simulation_config_rejects_malformed_dt(bad: object) -> None:
    with pytest.raises(ValueError, match="dt_s must be finite and positive"):
        SimulationConfig(dt_s=bad)


@pytest.mark.parametrize(
    "bad", (True, False, "0.001", b"0.001", None, 10**1000)
)
def test_run_constant_rejects_malformed_duration_atomically(bad: object) -> None:
    sim = Simulation()
    sim.step(0.5, 0.0)
    before = (
        sim.time_s,
        sim.body.left.copy(),
        sim.body.right.copy(),
        sim.body.shared_hand.fatigue,
        sim.body.shared_hand.coordination,
        sim.body.bilateral_state.coordination,
        sim.keyboard.left.pressed,
        sim.keyboard.right.pressed,
    )
    with pytest.raises(ValueError, match="duration_s must be finite and non-negative"):
        sim.run_constant(1.0, 0.0, bad)
    after = (
        sim.time_s,
        sim.body.left.copy(),
        sim.body.right.copy(),
        sim.body.shared_hand.fatigue,
        sim.body.shared_hand.coordination,
        sim.body.bilateral_state.coordination,
        sim.keyboard.left.pressed,
        sim.keyboard.right.pressed,
    )
    assert after == before


def test_run_constant_rejects_finite_duration_ratio_overflow_without_advancing() -> None:
    sim = Simulation(SimulationConfig(dt_s=1e-308))
    with pytest.raises(ValueError, match="duration_s / dt_s must be finite"):
        sim.run_constant(0.0, 0.0, 1e308)
    assert sim.time_s == 0.0
    assert sim.body.left.position_m == 0.0
    assert sim.body.right.position_m == 0.0
    assert not sim.keyboard.left.pressed
    assert not sim.keyboard.right.pressed


def test_run_constant_accepts_finite_integer_duration_and_preserves_step_count() -> None:
    assert SimulationConfig(dt_s=1).dt_s == 1
    sim = Simulation()
    results = sim.run_constant(0.0, 0.0, 0.003)
    assert len(results) == 3
    assert [result.time_s for result in results] == pytest.approx([0.001, 0.002, 0.003])

def test_zero_command_stays_at_rest():
    sim = Simulation()
    results = sim.run_constant(0.0, 0.0, 0.100)
    assert results[-1].left.position_m == pytest.approx(0.0)
    assert results[-1].right.position_m == pytest.approx(0.0)
    assert not any(result.events for result in results)


def test_press_generates_one_down_until_reset():
    sim = Simulation()
    pressing = sim.run_constant(1.0, 0.0, 0.100)
    down = [event for result in pressing for event in result.events if event == ("left", KeyEvent.DOWN)]
    assert len(down) == 1

    # Keeping the finger down must not create repeated digital presses.
    held = sim.run_constant(1.0, 0.0, 0.100)
    assert ("left", KeyEvent.DOWN) not in [event for result in held for event in result.events]

    releasing = sim.run_constant(-1.0, 0.0, 0.100)
    assert ("left", KeyEvent.UP) in [event for result in releasing for event in result.events]


def test_same_actions_are_deterministic():
    commands = [(1.0, 0.0)] * 40 + [(-1.0, 1.0)] * 40 + [(0.0, -1.0)] * 40

    def execute():
        sim = Simulation()
        return [sim.step(*command) for command in commands]

    assert execute() == execute()


def test_fatigue_is_bounded():
    sim = Simulation()
    results = sim.run_constant(1.0, 1.0, 5.0)
    assert 0.0 <= results[-1].left.fatigue <= 1.0
    assert 0.0 <= results[-1].right.fatigue <= 1.0


def test_reversal_fatigue_cannot_be_bypassed_through_zero():
    finger = FingerConfig(
        fatigue_gain_s=0.0,
        fatigue_recovery_s=0.0,
        switch_fatigue_per_reversal=0.10,
    )
    body = TwoFingerBody(BodyConfig(left=finger, right=finger))

    body.step(1.0, 0.0, 0.001)
    body.step(0.0, 0.0, 0.001)
    body.step(-1.0, 0.0, 0.001)

    assert body.left.fatigue == pytest.approx(0.10)
    assert body.left.last_nonzero_command_sign == -1


def test_returning_to_same_direction_through_zero_has_no_reversal_cost():
    finger = FingerConfig(
        fatigue_gain_s=0.0,
        fatigue_recovery_s=0.0,
        switch_fatigue_per_reversal=0.10,
    )
    body = TwoFingerBody(BodyConfig(left=finger, right=finger))

    body.step(1.0, 0.0, 0.001)
    body.step(0.0, 0.0, 0.001)
    body.step(1.0, 0.0, 0.001)

    assert body.left.fatigue == pytest.approx(0.0)
    assert body.left.last_nonzero_command_sign == 1


@pytest.mark.parametrize(
    "bad",
    (float("nan"), float("inf"), float("-inf"), True, False, "1", b"1", None, 10**1000),
)
@pytest.mark.parametrize("side", ("left", "right"))
def test_direct_two_finger_body_rejects_malformed_commands_atomically(
    bad: object, side: str
) -> None:
    body = TwoFingerBody()
    body.step(0.5, -0.5, 0.001)
    left_before, right_before = body.left.copy(), body.right.copy()
    hand_before = (body.shared_hand.fatigue, body.shared_hand.coordination)
    bilateral_before = body.bilateral_state.coordination

    commands = (bad, 1.0) if side == "left" else (1.0, bad)
    with pytest.raises(ValueError, match="motor commands must be finite"):
        body.step(*commands, 0.001)

    assert body.left == left_before
    assert body.right == right_before
    assert (body.shared_hand.fatigue, body.shared_hand.coordination) == hand_before
    assert body.bilateral_state.coordination == bilateral_before


@pytest.mark.parametrize(
    "bad",
    (True, False, "0.001", b"0.001", None, 10**1000),
)
def test_direct_two_finger_body_rejects_malformed_dt_without_mutation(
    bad: object,
) -> None:
    body = TwoFingerBody()
    body.step(0.5, 0.0, 0.001)
    left_before, right_before = body.left.copy(), body.right.copy()
    hand_before = (body.shared_hand.fatigue, body.shared_hand.coordination)
    bilateral_before = body.bilateral_state.coordination
    with pytest.raises(ValueError, match="dt_s must be finite and positive"):
        body.step(1.0, 0.0, bad)
    assert body.left == left_before
    assert body.right == right_before
    assert (body.shared_hand.fatigue, body.shared_hand.coordination) == hand_before
    assert body.bilateral_state.coordination == bilateral_before


@pytest.mark.parametrize("side", ("left", "right"))
def test_simulation_step_rejects_bad_body_command_without_advancing(
    side: str,
) -> None:
    sim = Simulation()
    commands = (float("nan"), 1.0) if side == "left" else (1.0, float("nan"))
    with pytest.raises(ValueError, match="motor commands must be finite"):
        sim.step(*commands)
    assert sim.time_s == 0.0
    assert sim.body.left == sim.body.left.copy()
    assert sim.body.left.activation == 0.0
    assert sim.body.right.activation == 0.0
    assert not sim.keyboard.left.pressed
    assert not sim.keyboard.right.pressed


def test_direct_two_finger_body_keeps_finite_integer_commands() -> None:
    body = TwoFingerBody()
    left, right = body.step(1, -1, 0.001)
    assert -1.0 <= left.activation <= 1.0
    assert -1.0 <= right.activation <= 1.0
