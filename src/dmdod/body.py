from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FingerConfig:
    """Provisional per-finger parameters; calibration targets, not physiology claims."""

    mass_kg: float = 0.020
    damping_n_s_m: float = 0.35
    spring_n_m: float = 35.0
    rest_position_m: float = 0.0
    max_force_n: float = 4.0
    activation_tau_s: float = 0.035
    fatigue_gain_s: float = 0.020
    fatigue_recovery_s: float = 0.35
    fatigue_threshold: float = 0.45
    fatigue_exponent: float = 2.0
    min_position_m: float = -0.001
    max_position_m: float = 0.006


@dataclass(frozen=True)
class HandConfig:
    """Shared force, fatigue and coordination limits for one hand."""

    capacity: float = 1.35
    fatigue_gain_s: float = 0.030
    fatigue_recovery_s: float = 0.25
    fatigue_threshold: float = 0.55
    fatigue_exponent: float = 2.0
    switch_tau_s: float = 0.030
    coordination_floor: float = 0.20


@dataclass
class HandState:
    fatigue: float = 0.0
    coordination: float = 0.0


@dataclass(frozen=True)
class BodyConfig:
    left: FingerConfig = field(default_factory=FingerConfig)
    right: FingerConfig = field(default_factory=FingerConfig)
    hand: HandConfig = field(default_factory=HandConfig)
    same_hand: bool = True


@dataclass
class FingerState:
    position_m: float = 0.0
    velocity_m_s: float = 0.0
    activation: float = 0.0
    fatigue: float = 0.0

    def copy(self) -> "FingerState":
        return FingerState(self.position_m, self.velocity_m_s, self.activation, self.fatigue)


@dataclass
class TwoFingerBody:
    config: BodyConfig = field(default_factory=BodyConfig)
    left: FingerState = field(default_factory=FingerState)
    right: FingerState = field(default_factory=FingerState)
    shared_hand: HandState = field(default_factory=HandState)

    def reset(self) -> None:
        self.left = FingerState(position_m=self.config.left.rest_position_m)
        self.right = FingerState(position_m=self.config.right.rest_position_m)
        self.shared_hand = HandState()

    @staticmethod
    def _fatigue_load(effort: float, threshold: float, exponent: float) -> float:
        """Nonlinear fatigue load: low/submaximal effort can be nearly sustainable."""
        threshold = max(0.0, min(0.99, threshold))
        normalized = max(0.0, effort - threshold) / (1.0 - threshold)
        return normalized ** max(1.0, exponent)

    def step(self, left_command: float, right_command: float, dt_s: float) -> tuple[FingerState, FingerState]:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")

        commands = (self._clamp(left_command, -1.0, 1.0), self._clamp(right_command, -1.0, 1.0))
        configs = (self.config.left, self.config.right)
        old = (self.left.copy(), self.right.copy())

        updated: list[FingerState] = []
        for state, command, cfg in zip(old, commands, configs):
            activation = state.activation + (command - state.activation) * dt_s / cfg.activation_tau_s
            activation = self._clamp(activation, -1.0, 1.0)
            load = self._fatigue_load(abs(activation), cfg.fatigue_threshold, cfg.fatigue_exponent)
            fatigue = state.fatigue + (cfg.fatigue_gain_s * load - cfg.fatigue_recovery_s * state.fatigue) * dt_s
            fatigue = self._clamp(fatigue, 0.0, 1.0)
            updated.append(FingerState(state.position_m, state.velocity_m_s, activation, fatigue))

        hand_scale = [1.0, 1.0]
        coordination_scale = [1.0, 1.0]
        if self.config.same_hand:
            hand_cfg = self.config.hand
            demand = abs(updated[0].activation) + abs(updated[1].activation)
            normalized_demand = min(1.0, demand / max(hand_cfg.capacity, 1e-9))
            hand_load = self._fatigue_load(normalized_demand, hand_cfg.fatigue_threshold, hand_cfg.fatigue_exponent)
            hand_fatigue = self.shared_hand.fatigue + (
                hand_cfg.fatigue_gain_s * hand_load - hand_cfg.fatigue_recovery_s * self.shared_hand.fatigue
            ) * dt_s
            self.shared_hand.fatigue = self._clamp(hand_fatigue, 0.0, 1.0)
            usable_capacity = hand_cfg.capacity * (1.0 - self.shared_hand.fatigue)
            if demand > usable_capacity and demand > 0.0:
                scale = usable_capacity / demand
                hand_scale = [scale, scale]

            positive_left = max(0.0, updated[0].activation)
            positive_right = max(0.0, updated[1].activation)
            press_demand = positive_left + positive_right
            if press_demand > 1e-9:
                target = (positive_left - positive_right) / press_demand
                tau = max(hand_cfg.switch_tau_s, dt_s)
                self.shared_hand.coordination += (target - self.shared_hand.coordination) * dt_s / tau
                self.shared_hand.coordination = self._clamp(self.shared_hand.coordination, -1.0, 1.0)

                floor = self._clamp(hand_cfg.coordination_floor, 0.0, 1.0)
                left_affinity = 0.5 * (1.0 + self.shared_hand.coordination)
                right_affinity = 1.0 - left_affinity
                coordination_scale = [
                    floor + (1.0 - floor) * left_affinity,
                    floor + (1.0 - floor) * right_affinity,
                ]

        next_states: list[FingerState] = []
        for index, (state, cfg) in enumerate(zip(updated, configs)):
            muscle_force = cfg.max_force_n * (1.0 - state.fatigue) * state.activation * hand_scale[index] * coordination_scale[index]
            spring_force = -cfg.spring_n_m * (state.position_m - cfg.rest_position_m)
            damping_force = -cfg.damping_n_s_m * state.velocity_m_s
            acceleration = (muscle_force + spring_force + damping_force) / cfg.mass_kg
            velocity = state.velocity_m_s + acceleration * dt_s
            position = state.position_m + velocity * dt_s

            if position <= cfg.min_position_m:
                position = cfg.min_position_m
                velocity = max(0.0, velocity)
            elif position >= cfg.max_position_m:
                position = cfg.max_position_m
                velocity = min(0.0, velocity)

            next_states.append(FingerState(position, velocity, state.activation, state.fatigue))

        self.left, self.right = next_states
        return self.left.copy(), self.right.copy()

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))
