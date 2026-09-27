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
    min_position_m: float = -0.001
    max_position_m: float = 0.006


@dataclass(frozen=True)
class HandConfig:
    """Shared capacity for fingers driven by one hand."""

    capacity: float = 1.35
    fatigue_gain_s: float = 0.030
    fatigue_recovery_s: float = 0.25


@dataclass
class HandState:
    fatigue: float = 0.0


@dataclass(frozen=True)
class BodyConfig:
    left: FingerConfig = field(default_factory=FingerConfig)
    right: FingerConfig = field(default_factory=FingerConfig)
    hand: HandConfig = field(default_factory=HandConfig)
    # True means the two simulated fingers share one hand-level resource (RI/RM).
    # False models fingers on separate hands (RI/LI).
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

    def step(self, left_command: float, right_command: float, dt_s: float) -> tuple[FingerState, FingerState]:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")

        commands = (
            self._clamp(left_command, -1.0, 1.0),
            self._clamp(right_command, -1.0, 1.0),
        )
        configs = (self.config.left, self.config.right)
        old = (self.left.copy(), self.right.copy())

        updated: list[FingerState] = []
        for state, command, cfg in zip(old, commands, configs):
            activation = state.activation + (command - state.activation) * dt_s / cfg.activation_tau_s
            activation = self._clamp(activation, -1.0, 1.0)
            fatigue = state.fatigue + (
                cfg.fatigue_gain_s * abs(activation) - cfg.fatigue_recovery_s * state.fatigue
            ) * dt_s
            fatigue = self._clamp(fatigue, 0.0, 1.0)
            updated.append(FingerState(state.position_m, state.velocity_m_s, activation, fatigue))

        # Same-hand fingers compete for a shared movement budget.  Separate-hand
        # fingers retain independent hand budgets, so RI/LI can exceed RI/RM.
        hand_scale = [1.0, 1.0]
        if self.config.same_hand:
            demand = abs(updated[0].activation) + abs(updated[1].activation)
            hand_cfg = self.config.hand
            hand_fatigue = self.shared_hand.fatigue + (
                hand_cfg.fatigue_gain_s * min(demand, 2.0)
                - hand_cfg.fatigue_recovery_s * self.shared_hand.fatigue
            ) * dt_s
            self.shared_hand.fatigue = self._clamp(hand_fatigue, 0.0, 1.0)
            usable_capacity = hand_cfg.capacity * (1.0 - self.shared_hand.fatigue)
            if demand > usable_capacity and demand > 0.0:
                scale = usable_capacity / demand
                hand_scale = [scale, scale]

        next_states: list[FingerState] = []
        for index, (state, cfg) in enumerate(zip(updated, configs)):
            muscle_force = (
                cfg.max_force_n
                * (1.0 - state.fatigue)
                * state.activation
                * hand_scale[index]
            )
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
