from __future__ import annotations

"""Exact fast path for held motor commands.

The policy acts every 10 ms by default while the physical model integrates at
1 kHz.  The ordinary simulator API allocates StepResult/FingerState/list objects
for every 1 ms substep, although none of those intermediate objects escape a
MotorEnv control step.  This module keeps the same scalar update equations and
keyboard threshold semantics, but carries the intermediate state in local
variables and materializes body state only once at the end of the held command.
"""

from dmdod.motor.keyboard import KeyEvent
from dmdod.envs.simulator import Simulation


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _fatigue_load(effort: float, threshold: float, exponent: float) -> float:
    threshold = max(0.0, min(0.99, threshold))
    normalized = max(0.0, effort - threshold) / (1.0 - threshold)
    return normalized ** max(1.0, exponent)


def step_held_exact(
    sim: Simulation,
    left_command: float,
    right_command: float,
    substeps: int,
) -> tuple[tuple[float, str, KeyEvent], ...]:
    """Advance ``substeps`` physics ticks with one held command.

    Numerics intentionally mirror ``TwoFingerBody.step`` and ``Simulation.step``
    in the same order.  Returned event timestamps are the post-step simulator
    times, matching ``StepResult.time_s`` from the ordinary path.
    """

    if int(substeps) < 0:
        raise ValueError("substeps must be non-negative")
    if not substeps:
        return ()

    dt_s = sim.config.dt_s
    if dt_s <= 0.0:
        raise ValueError("physics dt must be positive")

    body = sim.body
    body_cfg = body.config
    left_cfg = body_cfg.left
    right_cfg = body_cfg.right
    hand_cfg = body_cfg.hand
    bilateral_cfg = body_cfg.bilateral

    left_command = _clamp(left_command, -1.0, 1.0)
    right_command = _clamp(right_command, -1.0, 1.0)

    # The ordinary body step replaces left/right FingerState objects every
    # physics tick.  Copy once here so references held before this control step
    # still observe the same non-mutating behavior without per-tick allocation.
    left_state = body.left.copy()
    right_state = body.right.copy()
    body.left = left_state
    body.right = right_state

    left_position = left_state.position_m
    left_velocity = left_state.velocity_m_s
    left_activation = left_state.activation
    left_fatigue = left_state.fatigue
    left_last_sign = left_state.last_nonzero_command_sign

    right_position = right_state.position_m
    right_velocity = right_state.velocity_m_s
    right_activation = right_state.activation
    right_fatigue = right_state.fatigue
    right_last_sign = right_state.last_nonzero_command_sign

    hand_fatigue = body.shared_hand.fatigue
    hand_coordination = body.shared_hand.coordination
    bilateral_coordination = body.bilateral_state.coordination

    keyboard = sim.keyboard
    left_pressed = keyboard.left.pressed
    right_pressed = keyboard.right.pressed
    actuation_m = keyboard.config.actuation_m
    reset_m = keyboard.config.reset_m

    left_command_sign = body._command_sign(left_command)
    right_command_sign = body._command_sign(right_command)
    time_s = sim.time_s
    events: list[tuple[float, str, KeyEvent]] = []

    for _ in range(int(substeps)):
        left_activation = left_activation + (
            left_command - left_activation
        ) * dt_s / left_cfg.activation_tau_s
        left_activation = _clamp(left_activation, -1.0, 1.0)
        left_load = _fatigue_load(
            abs(left_activation),
            left_cfg.fatigue_threshold,
            left_cfg.fatigue_exponent,
        )
        left_reversed = (
            left_command_sign != 0
            and left_last_sign != 0
            and left_command_sign != left_last_sign
        )
        left_reversal_cost = left_cfg.switch_fatigue_per_reversal if left_reversed else 0.0
        left_fatigue = (
            left_fatigue
            + (
                left_cfg.fatigue_gain_s * left_load
                - left_cfg.fatigue_recovery_s * left_fatigue
            )
            * dt_s
            + left_reversal_cost
        )
        left_fatigue = _clamp(left_fatigue, 0.0, 1.0)
        if left_command_sign != 0:
            left_last_sign = left_command_sign

        right_activation = right_activation + (
            right_command - right_activation
        ) * dt_s / right_cfg.activation_tau_s
        right_activation = _clamp(right_activation, -1.0, 1.0)
        right_load = _fatigue_load(
            abs(right_activation),
            right_cfg.fatigue_threshold,
            right_cfg.fatigue_exponent,
        )
        right_reversed = (
            right_command_sign != 0
            and right_last_sign != 0
            and right_command_sign != right_last_sign
        )
        right_reversal_cost = (
            right_cfg.switch_fatigue_per_reversal if right_reversed else 0.0
        )
        right_fatigue = (
            right_fatigue
            + (
                right_cfg.fatigue_gain_s * right_load
                - right_cfg.fatigue_recovery_s * right_fatigue
            )
            * dt_s
            + right_reversal_cost
        )
        right_fatigue = _clamp(right_fatigue, 0.0, 1.0)
        if right_command_sign != 0:
            right_last_sign = right_command_sign

        left_hand_scale = 1.0
        right_hand_scale = 1.0
        left_coord_scale = 1.0
        right_coord_scale = 1.0
        positive_left = max(0.0, left_activation)
        positive_right = max(0.0, right_activation)
        press_demand = positive_left + positive_right

        if body_cfg.same_hand:
            demand = abs(left_activation) + abs(right_activation)
            normalized_demand = min(
                1.0,
                demand / max(hand_cfg.capacity, 1e-9),
            )
            hand_load = _fatigue_load(
                normalized_demand,
                hand_cfg.fatigue_threshold,
                hand_cfg.fatigue_exponent,
            )
            hand_fatigue = hand_fatigue + (
                hand_cfg.fatigue_gain_s * hand_load
                - hand_cfg.fatigue_recovery_s * hand_fatigue
            ) * dt_s
            hand_fatigue = _clamp(hand_fatigue, 0.0, 1.0)
            usable_capacity = hand_cfg.capacity * (1.0 - hand_fatigue)
            if demand > usable_capacity and demand > 0.0:
                scale = usable_capacity / demand
                left_hand_scale = scale
                right_hand_scale = scale

            if press_demand > 1e-9:
                target = (positive_left - positive_right) / press_demand
                tau = max(hand_cfg.switch_tau_s, dt_s)
                hand_coordination += (
                    target - hand_coordination
                ) * dt_s / tau
                hand_coordination = _clamp(hand_coordination, -1.0, 1.0)
                floor = max(0.0, min(1.0, hand_cfg.coordination_floor))
                left_affinity = 0.5 * (1.0 + hand_coordination)
                right_affinity = 1.0 - left_affinity
                left_coord_scale = floor + (1.0 - floor) * left_affinity
                right_coord_scale = floor + (1.0 - floor) * right_affinity
        elif press_demand > 1e-9:
            target = (positive_left - positive_right) / press_demand
            tau = max(bilateral_cfg.switch_tau_s, dt_s)
            bilateral_coordination += (
                target - bilateral_coordination
            ) * dt_s / tau
            bilateral_coordination = _clamp(
                bilateral_coordination,
                -1.0,
                1.0,
            )
            floor = max(0.0, min(1.0, bilateral_cfg.coordination_floor))
            left_affinity = 0.5 * (1.0 + bilateral_coordination)
            right_affinity = 1.0 - left_affinity
            left_coord_scale = floor + (1.0 - floor) * left_affinity
            right_coord_scale = floor + (1.0 - floor) * right_affinity

        left_muscle_force = (
            left_cfg.max_force_n
            * (1.0 - left_fatigue)
            * left_activation
            * left_hand_scale
            * left_coord_scale
        )
        left_spring_force = -left_cfg.spring_n_m * (
            left_position - left_cfg.rest_position_m
        )
        left_damping_force = -left_cfg.damping_n_s_m * left_velocity
        left_acceleration = (
            left_muscle_force + left_spring_force + left_damping_force
        ) / left_cfg.mass_kg
        left_velocity = left_velocity + left_acceleration * dt_s
        left_position = left_position + left_velocity * dt_s
        if left_position <= left_cfg.min_position_m:
            left_position = left_cfg.min_position_m
            left_velocity = max(0.0, left_velocity)
        elif left_position >= left_cfg.max_position_m:
            left_position = left_cfg.max_position_m
            left_velocity = min(0.0, left_velocity)

        right_muscle_force = (
            right_cfg.max_force_n
            * (1.0 - right_fatigue)
            * right_activation
            * right_hand_scale
            * right_coord_scale
        )
        right_spring_force = -right_cfg.spring_n_m * (
            right_position - right_cfg.rest_position_m
        )
        right_damping_force = -right_cfg.damping_n_s_m * right_velocity
        right_acceleration = (
            right_muscle_force + right_spring_force + right_damping_force
        ) / right_cfg.mass_kg
        right_velocity = right_velocity + right_acceleration * dt_s
        right_position = right_position + right_velocity * dt_s
        if right_position <= right_cfg.min_position_m:
            right_position = right_cfg.min_position_m
            right_velocity = max(0.0, right_velocity)
        elif right_position >= right_cfg.max_position_m:
            right_position = right_cfg.max_position_m
            right_velocity = min(0.0, right_velocity)

        next_time_s = time_s + dt_s
        if not left_pressed and left_position >= actuation_m:
            left_pressed = True
            events.append((next_time_s, "left", KeyEvent.DOWN))
        elif left_pressed and left_position <= reset_m:
            left_pressed = False
            events.append((next_time_s, "left", KeyEvent.UP))

        if not right_pressed and right_position >= actuation_m:
            right_pressed = True
            events.append((next_time_s, "right", KeyEvent.DOWN))
        elif right_pressed and right_position <= reset_m:
            right_pressed = False
            events.append((next_time_s, "right", KeyEvent.UP))
        time_s = next_time_s

    left_state.position_m = left_position
    left_state.velocity_m_s = left_velocity
    left_state.activation = left_activation
    left_state.fatigue = left_fatigue
    left_state.last_command = left_command
    left_state.last_nonzero_command_sign = left_last_sign

    right_state.position_m = right_position
    right_state.velocity_m_s = right_velocity
    right_state.activation = right_activation
    right_state.fatigue = right_fatigue
    right_state.last_command = right_command
    right_state.last_nonzero_command_sign = right_last_sign

    body.shared_hand.fatigue = hand_fatigue
    body.shared_hand.coordination = hand_coordination
    body.bilateral_state.coordination = bilateral_coordination
    keyboard.left.pressed = left_pressed
    keyboard.right.pressed = right_pressed
    sim.time_s = time_s
    return tuple(events)
