from __future__ import annotations

"""Four-key motor body for higher-speed ADOFAI experiments.

The canonical physical order is left-to-right::

    left_outer, left_inner, right_inner, right_outer

The two inner keys are the default teacher pair.  Outer keys are overflow: a
center-first router only reaches for them while both inner keys are still held.
This keeps ordinary play close to the existing two-finger strategy while making
four-finger bursts possible without hard-wiring chart targets to specific keys.
"""

from dataclasses import dataclass, replace

from dmdod.motor.body import (
    BilateralConfig,
    BilateralState,
    FingerConfig,
    FingerState,
    HandConfig,
    HandState,
)
from dmdod.motor.keyboard import KeyConfig, KeyEvent, KeyState
from dmdod.motor.env import TimedKeyEvent
from dmdod.profiles import personal_blue_switch_v0_1


FOUR_KEY_NAMES = (
    "left_outer",
    "left_inner",
    "right_inner",
    "right_outer",
)
CENTER_KEY_NAMES = ("left_inner", "right_inner")
OUTER_KEY_NAMES = ("left_outer", "right_outer")
FOUR_KEY_OBSERVATION_DIM = 12
FOUR_KEY_ACTION_DIM = 4


@dataclass(frozen=True, slots=True)
class FourKeyAction:
    left_outer: float
    left_inner: float
    right_inner: float
    right_outer: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (
            float(self.left_outer),
            float(self.left_inner),
            float(self.right_inner),
            float(self.right_outer),
        )


@dataclass(frozen=True, slots=True)
class FourKeyObservation:
    left_outer_position_m: float
    left_inner_position_m: float
    right_inner_position_m: float
    right_outer_position_m: float
    left_outer_velocity_m_s: float
    left_inner_velocity_m_s: float
    right_inner_velocity_m_s: float
    right_outer_velocity_m_s: float
    left_outer_pressed: bool
    left_inner_pressed: bool
    right_inner_pressed: bool
    right_outer_pressed: bool

    def pressed(self, key: str) -> bool:
        if key not in FOUR_KEY_NAMES:
            raise KeyError(key)
        return bool(getattr(self, f"{key}_pressed"))


@dataclass(frozen=True, slots=True)
class FourKeyDiagnostics:
    time_s: float
    left_outer_activation: float
    left_inner_activation: float
    right_inner_activation: float
    right_outer_activation: float
    left_outer_fatigue: float
    left_inner_fatigue: float
    right_inner_fatigue: float
    right_outer_fatigue: float
    left_hand_fatigue: float
    right_hand_fatigue: float
    left_hand_coordination: float
    right_hand_coordination: float
    bilateral_coordination: float


@dataclass(frozen=True, slots=True)
class FourKeyTransition:
    observation: FourKeyObservation
    evaluator_events: tuple[TimedKeyEvent, ...]
    diagnostics: FourKeyDiagnostics


@dataclass(frozen=True, slots=True)
class FourFingerConfig:
    left_outer: FingerConfig
    left_inner: FingerConfig
    right_inner: FingerConfig
    right_outer: FingerConfig
    left_hand: HandConfig
    right_hand: HandConfig
    bilateral: BilateralConfig


def personal_blue_switch_four_key_v0_1() -> FourFingerConfig:
    """Lift the frozen two-finger profile into two two-finger hands.

    This deliberately does not invent new finger constants yet.  All four
    fingers begin with the already-calibrated v0.1 switch/body parameters, each
    hand gets the existing shared-hand limits, and the existing bilateral
    transfer model remains active between the two hands.
    """

    base = personal_blue_switch_v0_1(same_hand=True)
    return FourFingerConfig(
        left_outer=base.left,
        left_inner=replace(base.right),
        right_inner=replace(base.left),
        right_outer=replace(base.right),
        left_hand=base.hand,
        right_hand=replace(base.hand),
        bilateral=base.bilateral,
    )


class FourFingerBody:
    """Four fingers arranged as two constrained two-finger hands."""

    def __init__(self, config: FourFingerConfig | None = None) -> None:
        self.config = config or personal_blue_switch_four_key_v0_1()
        self.reset()

    def reset(self) -> None:
        cfg = self.config
        self.left_outer = FingerState(position_m=cfg.left_outer.rest_position_m)
        self.left_inner = FingerState(position_m=cfg.left_inner.rest_position_m)
        self.right_inner = FingerState(position_m=cfg.right_inner.rest_position_m)
        self.right_outer = FingerState(position_m=cfg.right_outer.rest_position_m)
        self.left_hand_state = HandState()
        self.right_hand_state = HandState()
        self.bilateral_state = BilateralState()

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _fatigue_load(effort: float, threshold: float, exponent: float) -> float:
        threshold = max(0.0, min(0.99, threshold))
        normalized = max(0.0, effort - threshold) / (1.0 - threshold)
        return normalized ** max(1.0, exponent)

    @staticmethod
    def _coordination_scales(coordination: float, floor: float) -> tuple[float, float]:
        floor = max(0.0, min(1.0, floor))
        first_affinity = 0.5 * (1.0 + coordination)
        second_affinity = 1.0 - first_affinity
        return (
            floor + (1.0 - floor) * first_affinity,
            floor + (1.0 - floor) * second_affinity,
        )

    @staticmethod
    def _command_sign(command: float, deadzone: float = 1e-9) -> int:
        if command > deadzone:
            return 1
        if command < -deadzone:
            return -1
        return 0

    def _update_finger(
        self,
        state: FingerState,
        command: float,
        cfg: FingerConfig,
        dt_s: float,
    ) -> FingerState:
        command = self._clamp(command, -1.0, 1.0)
        activation = state.activation + (command - state.activation) * dt_s / cfg.activation_tau_s
        activation = self._clamp(activation, -1.0, 1.0)

        load = self._fatigue_load(abs(activation), cfg.fatigue_threshold, cfg.fatigue_exponent)
        command_sign = self._command_sign(command)
        reversed_direction = (
            command_sign != 0
            and state.last_nonzero_command_sign != 0
            and command_sign != state.last_nonzero_command_sign
        )
        reversal_cost = cfg.switch_fatigue_per_reversal if reversed_direction else 0.0
        fatigue = (
            state.fatigue
            + (cfg.fatigue_gain_s * load - cfg.fatigue_recovery_s * state.fatigue) * dt_s
            + reversal_cost
        )
        fatigue = self._clamp(fatigue, 0.0, 1.0)
        last_nonzero_sign = state.last_nonzero_command_sign if command_sign == 0 else command_sign
        return FingerState(
            state.position_m,
            state.velocity_m_s,
            activation,
            fatigue,
            command,
            last_nonzero_sign,
        )

    def _hand_scales(
        self,
        first: FingerState,
        second: FingerState,
        hand_state: HandState,
        hand_cfg: HandConfig,
        dt_s: float,
    ) -> tuple[float, float]:
        demand = abs(first.activation) + abs(second.activation)
        normalized_demand = min(1.0, demand / max(hand_cfg.capacity, 1e-9))
        load = self._fatigue_load(
            normalized_demand,
            hand_cfg.fatigue_threshold,
            hand_cfg.fatigue_exponent,
        )
        hand_state.fatigue = self._clamp(
            hand_state.fatigue
            + (
                hand_cfg.fatigue_gain_s * load
                - hand_cfg.fatigue_recovery_s * hand_state.fatigue
            )
            * dt_s,
            0.0,
            1.0,
        )
        usable_capacity = hand_cfg.capacity * (1.0 - hand_state.fatigue)
        capacity_scale = 1.0
        if demand > usable_capacity and demand > 0.0:
            capacity_scale = usable_capacity / demand

        positive_first = max(0.0, first.activation)
        positive_second = max(0.0, second.activation)
        press_demand = positive_first + positive_second
        coordination_scale = (1.0, 1.0)
        if press_demand > 1e-9:
            target = (positive_first - positive_second) / press_demand
            tau = max(hand_cfg.switch_tau_s, dt_s)
            hand_state.coordination += (target - hand_state.coordination) * dt_s / tau
            hand_state.coordination = self._clamp(hand_state.coordination, -1.0, 1.0)
            coordination_scale = self._coordination_scales(
                hand_state.coordination,
                hand_cfg.coordination_floor,
            )
        return (
            capacity_scale * coordination_scale[0],
            capacity_scale * coordination_scale[1],
        )

    def _bilateral_scales(
        self,
        left_first: FingerState,
        left_second: FingerState,
        right_first: FingerState,
        right_second: FingerState,
        dt_s: float,
    ) -> tuple[float, float]:
        left_press = max(0.0, left_first.activation) + max(0.0, left_second.activation)
        right_press = max(0.0, right_first.activation) + max(0.0, right_second.activation)
        demand = left_press + right_press
        if demand <= 1e-9:
            return 1.0, 1.0

        target = (left_press - right_press) / demand
        cfg = self.config.bilateral
        tau = max(cfg.switch_tau_s, dt_s)
        self.bilateral_state.coordination += (
            target - self.bilateral_state.coordination
        ) * dt_s / tau
        self.bilateral_state.coordination = self._clamp(
            self.bilateral_state.coordination,
            -1.0,
            1.0,
        )
        return self._coordination_scales(
            self.bilateral_state.coordination,
            cfg.coordination_floor,
        )

    def _integrate(
        self,
        state: FingerState,
        cfg: FingerConfig,
        scale: float,
        dt_s: float,
    ) -> FingerState:
        muscle_force = cfg.max_force_n * (1.0 - state.fatigue) * state.activation * scale
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

        return FingerState(
            position,
            velocity,
            state.activation,
            state.fatigue,
            state.last_command,
            state.last_nonzero_command_sign,
        )

    def step(
        self,
        action: FourKeyAction,
        dt_s: float,
    ) -> tuple[FingerState, FingerState, FingerState, FingerState]:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")

        old = (
            self.left_outer.copy(),
            self.left_inner.copy(),
            self.right_inner.copy(),
            self.right_outer.copy(),
        )
        cfgs = (
            self.config.left_outer,
            self.config.left_inner,
            self.config.right_inner,
            self.config.right_outer,
        )
        updated = tuple(
            self._update_finger(state, command, cfg, dt_s)
            for state, command, cfg in zip(old, action.as_tuple(), cfgs)
        )

        left_scales = self._hand_scales(
            updated[0],
            updated[1],
            self.left_hand_state,
            self.config.left_hand,
            dt_s,
        )
        right_scales = self._hand_scales(
            updated[2],
            updated[3],
            self.right_hand_state,
            self.config.right_hand,
            dt_s,
        )
        bilateral_left, bilateral_right = self._bilateral_scales(
            updated[0], updated[1], updated[2], updated[3], dt_s
        )
        scales = (
            left_scales[0] * bilateral_left,
            left_scales[1] * bilateral_left,
            right_scales[0] * bilateral_right,
            right_scales[1] * bilateral_right,
        )
        next_states = tuple(
            self._integrate(state, cfg, scale, dt_s)
            for state, cfg, scale in zip(updated, cfgs, scales)
        )
        (
            self.left_outer,
            self.left_inner,
            self.right_inner,
            self.right_outer,
        ) = next_states
        return tuple(state.copy() for state in next_states)  # type: ignore[return-value]


class FourKeyKeyboard:
    def __init__(self, config: KeyConfig | None = None) -> None:
        self.config = config or KeyConfig()
        self.reset()

    def reset(self) -> None:
        self.left_outer = KeyState()
        self.left_inner = KeyState()
        self.right_inner = KeyState()
        self.right_outer = KeyState()

    def step(
        self,
        positions: tuple[float, float, float, float],
    ) -> list[tuple[str, KeyEvent]]:
        events: list[tuple[str, KeyEvent]] = []
        for name, position in zip(FOUR_KEY_NAMES, positions):
            state: KeyState = getattr(self, name)
            if not state.pressed and position >= self.config.actuation_m:
                state.pressed = True
                events.append((name, KeyEvent.DOWN))
            elif state.pressed and position <= self.config.reset_m:
                state.pressed = False
                events.append((name, KeyEvent.UP))
        return events


class FourKeyMotorEnv:
    """1 kHz four-finger body behind the existing 100 Hz motor contract."""

    def __init__(
        self,
        *,
        control_dt_s: float = 0.010,
        physics_dt_s: float = 0.001,
        body_config: FourFingerConfig | None = None,
        key_config: KeyConfig | None = None,
    ) -> None:
        if physics_dt_s <= 0.0:
            raise ValueError("physics_dt_s must be positive")
        if control_dt_s < physics_dt_s:
            raise ValueError("control_dt_s must be at least one physics step")
        substeps = round(control_dt_s / physics_dt_s)
        if abs(substeps * physics_dt_s - control_dt_s) > 1e-12:
            raise ValueError("control_dt_s must be an integer multiple of physics_dt_s")

        self.body = FourFingerBody(body_config)
        self.keyboard = FourKeyKeyboard(key_config)
        self.control_dt_s = float(control_dt_s)
        self.physics_dt_s = float(physics_dt_s)
        self.physics_substeps = int(substeps)
        self.time_s = 0.0

    def reset(self) -> FourKeyObservation:
        self.body.reset()
        self.keyboard.reset()
        self.time_s = 0.0
        return self.observe()

    def observe(self) -> FourKeyObservation:
        body = self.body
        keyboard = self.keyboard
        return FourKeyObservation(
            left_outer_position_m=body.left_outer.position_m,
            left_inner_position_m=body.left_inner.position_m,
            right_inner_position_m=body.right_inner.position_m,
            right_outer_position_m=body.right_outer.position_m,
            left_outer_velocity_m_s=body.left_outer.velocity_m_s,
            left_inner_velocity_m_s=body.left_inner.velocity_m_s,
            right_inner_velocity_m_s=body.right_inner.velocity_m_s,
            right_outer_velocity_m_s=body.right_outer.velocity_m_s,
            left_outer_pressed=keyboard.left_outer.pressed,
            left_inner_pressed=keyboard.left_inner.pressed,
            right_inner_pressed=keyboard.right_inner.pressed,
            right_outer_pressed=keyboard.right_outer.pressed,
        )

    def diagnostics(self) -> FourKeyDiagnostics:
        b = self.body
        return FourKeyDiagnostics(
            time_s=self.time_s,
            left_outer_activation=b.left_outer.activation,
            left_inner_activation=b.left_inner.activation,
            right_inner_activation=b.right_inner.activation,
            right_outer_activation=b.right_outer.activation,
            left_outer_fatigue=b.left_outer.fatigue,
            left_inner_fatigue=b.left_inner.fatigue,
            right_inner_fatigue=b.right_inner.fatigue,
            right_outer_fatigue=b.right_outer.fatigue,
            left_hand_fatigue=b.left_hand_state.fatigue,
            right_hand_fatigue=b.right_hand_state.fatigue,
            left_hand_coordination=b.left_hand_state.coordination,
            right_hand_coordination=b.right_hand_state.coordination,
            bilateral_coordination=b.bilateral_state.coordination,
        )

    def step(self, action: FourKeyAction) -> FourKeyTransition:
        clamped = FourKeyAction(
            *(max(-1.0, min(1.0, value)) for value in action.as_tuple())
        )
        events: list[TimedKeyEvent] = []
        for _ in range(self.physics_substeps):
            states = self.body.step(clamped, self.physics_dt_s)
            self.time_s += self.physics_dt_s
            raw = self.keyboard.step(tuple(state.position_m for state in states))
            events.extend(
                TimedKeyEvent(self.time_s, key, event)
                for key, event in raw
            )
        return FourKeyTransition(self.observe(), tuple(events), self.diagnostics())


class CenterFirstFourKeyRouter:
    """Teacher-side key allocator that treats outer fingers as overflow.

    With all keys available it alternates ``left_inner`` and ``right_inner``.
    It uses an outer key only while both inner keys are still physically held.
    This routing state is privileged teacher machinery and is never part of the
    policy observation.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # Seed the opposite side so the very first choice is left-inner/left-outer.
        self._last_center = "right_inner"
        self._last_outer = "right_outer"

    @staticmethod
    def _ordered_pair(pair: tuple[str, str], last: str) -> tuple[str, str]:
        return (pair[1], pair[0]) if last == pair[0] else pair

    def choose_key(self, observation: FourKeyObservation) -> str | None:
        for key in self._ordered_pair(CENTER_KEY_NAMES, self._last_center):
            if not observation.pressed(key):
                self._last_center = key
                return key
        for key in self._ordered_pair(OUTER_KEY_NAMES, self._last_outer):
            if not observation.pressed(key):
                self._last_outer = key
                return key
        return None

    def teacher_action(
        self,
        observation: FourKeyObservation,
        *,
        press_now: bool,
    ) -> FourKeyAction:
        commands = {
            key: (-1.0 if observation.pressed(key) else 0.0)
            for key in FOUR_KEY_NAMES
        }
        if press_now:
            chosen = self.choose_key(observation)
            if chosen is not None:
                commands[chosen] = 1.0
        return FourKeyAction(*(commands[key] for key in FOUR_KEY_NAMES))
