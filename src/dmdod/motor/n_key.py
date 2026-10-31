from __future__ import annotations

"""Configurable even-N keyboard body for teacher-capacity experiments.

The physical layout is always symmetric around the keyboard center.  ``depth=1``
is the innermost key on a hand, with larger depths moving outward.  For example::

    4K: left_2 left_1 | right_1 right_2
    6K: left_3 left_2 left_1 | right_1 right_2 right_3
    8K: left_4 left_3 left_2 left_1 | right_1 right_2 right_3 right_4

The frozen v0.1 per-finger, shared-hand, and bilateral parameters are reused.
Per-hand capacity is deliberately *not* increased with finger count: adding keys
adds recovery lanes, not free hand strength.  The generalized hand-coordination
formula is defined so the 4K case is numerically equivalent to FourFingerBody.
"""

from dataclasses import dataclass, replace
from math import isfinite

from dmdod.motor.body import BilateralState, FingerConfig, FingerState, HandConfig, HandState
from dmdod.motor.keyboard import KeyConfig, KeyEvent, KeyState
from dmdod.motor.env import TimedKeyEvent
from dmdod.profiles import personal_blue_switch_v0_1


def _validated_key_count(key_count: int) -> int:
    """Reject fractional, boolean and string counts without silent coercion."""
    if type(key_count) is not int or key_count < 2 or key_count % 2 != 0:
        raise ValueError("key_count must be an even integer >= 2")
    return key_count


def n_key_names(key_count: int) -> tuple[str, ...]:
    key_count = _validated_key_count(key_count)
    per_hand = key_count // 2
    return tuple(
        [f"left_{depth}" for depth in range(per_hand, 0, -1)]
        + [f"right_{depth}" for depth in range(1, per_hand + 1)]
    )


def n_key_tiers(key_count: int) -> tuple[tuple[str, str], ...]:
    key_count = _validated_key_count(key_count)
    return tuple(
        (f"left_{depth}", f"right_{depth}")
        for depth in range(1, key_count // 2 + 1)
    )


@dataclass(frozen=True, slots=True)
class NKeyAction:
    values: tuple[float, ...]

    def as_tuple(self) -> tuple[float, ...]:
        # A boolean or numeric-looking string is not a physical command.
        # float(True) and float("1") otherwise turn malformed policy output
        # into a genuine full-strength key press. Convert once at the edge.
        commands: list[float] = []
        for value in self.values:
            if isinstance(value, (bool, str, bytes)):
                raise ValueError("N-key motor commands must be numeric")
            try:
                commands.append(float(value))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("N-key motor commands must be numeric") from exc
        return tuple(commands)


@dataclass(frozen=True, slots=True)
class NKeyObservation:
    key_names: tuple[str, ...]
    positions_m: tuple[float, ...]
    velocities_m_s: tuple[float, ...]
    pressed_flags: tuple[bool, ...]

    def __post_init__(self) -> None:
        size = len(self.key_names)
        if not (
            len(self.positions_m) == size
            and len(self.velocities_m_s) == size
            and len(self.pressed_flags) == size
        ):
            raise ValueError("N-key observation fields must have matching lengths")

    def index(self, key: str) -> int:
        try:
            return self.key_names.index(key)
        except ValueError as exc:
            raise KeyError(key) from exc

    def pressed(self, key: str) -> bool:
        return bool(self.pressed_flags[self.index(key)])


@dataclass(frozen=True, slots=True)
class NKeyDiagnostics:
    time_s: float
    activations: tuple[float, ...]
    finger_fatigue: tuple[float, ...]
    left_hand_fatigue: float
    right_hand_fatigue: float
    left_hand_coordination: float
    right_hand_coordination: float
    bilateral_coordination: float


@dataclass(frozen=True, slots=True)
class NKeyPhysicsSample:
    """Optional evaluator-side 1 kHz motor trace for diagnostics only."""

    time_s: float
    positions_m: tuple[float, ...]
    events: tuple[tuple[str, KeyEvent], ...]


@dataclass(frozen=True, slots=True)
class NKeyTransition:
    observation: NKeyObservation
    evaluator_events: tuple[TimedKeyEvent, ...]
    diagnostics: NKeyDiagnostics
    physics_samples: tuple[NKeyPhysicsSample, ...] = ()


class NKeyBody:
    """Two-hand body with an arbitrary equal number of fingers per hand."""

    def __init__(self, key_count: int) -> None:
        self.key_names = n_key_names(key_count)
        self.key_count = key_count
        self.per_hand = self.key_count // 2

        base = personal_blue_switch_v0_1(same_hand=True)
        # FourKeyBody uses [base.left, base.right] in physical order on each
        # hand.  Repeat that frozen pair outward/inward for larger hands.
        local_cfgs: list[FingerConfig] = []
        for index in range(self.per_hand):
            local_cfgs.append(base.left if index % 2 == 0 else replace(base.right))
        self.configs = tuple([*local_cfgs, *local_cfgs])
        self.left_hand_config: HandConfig = base.hand
        self.right_hand_config: HandConfig = replace(base.hand)
        self.bilateral_config = base.bilateral
        self.reset()

    def reset(self) -> None:
        self.states = [
            FingerState(position_m=cfg.rest_position_m)
            for cfg in self.configs
        ]
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

    @staticmethod
    def _coordination_positions(count: int) -> tuple[float, ...]:
        if count <= 1:
            return (0.0,)
        return tuple(1.0 - 2.0 * index / (count - 1) for index in range(count))

    def _hand_scales(
        self,
        states: tuple[FingerState, ...],
        hand_state: HandState,
        hand_cfg: HandConfig,
        dt_s: float,
    ) -> tuple[float, ...]:
        demand = sum(abs(state.activation) for state in states)
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

        positive = tuple(max(0.0, state.activation) for state in states)
        press_demand = sum(positive)
        coordination = [1.0] * len(states)
        if press_demand > 1e-9 and len(states) > 1:
            positions = self._coordination_positions(len(states))
            target = sum(p * value for p, value in zip(positions, positive)) / press_demand
            tau = max(hand_cfg.switch_tau_s, dt_s)
            hand_state.coordination += (target - hand_state.coordination) * dt_s / tau
            hand_state.coordination = self._clamp(hand_state.coordination, -1.0, 1.0)
            floor = self._clamp(hand_cfg.coordination_floor, 0.0, 1.0)
            coordination = [
                floor
                + (1.0 - floor)
                * 0.5
                * (1.0 + hand_state.coordination * position)
                for position in positions
            ]

        return tuple(capacity_scale * scale for scale in coordination)

    def _bilateral_scales(
        self,
        left: tuple[FingerState, ...],
        right: tuple[FingerState, ...],
        dt_s: float,
    ) -> tuple[float, float]:
        left_press = sum(max(0.0, state.activation) for state in left)
        right_press = sum(max(0.0, state.activation) for state in right)
        demand = left_press + right_press
        if demand <= 1e-9:
            return 1.0, 1.0

        target = (left_press - right_press) / demand
        cfg = self.bilateral_config
        tau = max(cfg.switch_tau_s, dt_s)
        self.bilateral_state.coordination += (
            target - self.bilateral_state.coordination
        ) * dt_s / tau
        self.bilateral_state.coordination = self._clamp(
            self.bilateral_state.coordination,
            -1.0,
            1.0,
        )
        floor = self._clamp(cfg.coordination_floor, 0.0, 1.0)
        left_affinity = 0.5 * (1.0 + self.bilateral_state.coordination)
        right_affinity = 1.0 - left_affinity
        return (
            floor + (1.0 - floor) * left_affinity,
            floor + (1.0 - floor) * right_affinity,
        )

    @staticmethod
    def _integrate(
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

    def step(self, action: NKeyAction, dt_s: float) -> tuple[FingerState, ...]:
        if not isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        values = action.as_tuple()
        if len(values) != self.key_count:
            raise ValueError(f"expected {self.key_count} action values, got {len(values)}")
        # NKeyMotorEnv validates this at its public boundary, but the body is
        # also used directly by research code. Reject before mutating fatigue
        # or coordination; NaN would otherwise be silently clamped to +1.
        if any(not isfinite(value) for value in values):
            raise ValueError("N-key motor commands must be finite")

        old = tuple(state.copy() for state in self.states)
        updated = tuple(
            self._update_finger(state, command, cfg, dt_s)
            for state, command, cfg in zip(old, values, self.configs)
        )
        split = self.per_hand
        left = updated[:split]
        right = updated[split:]
        left_scales = self._hand_scales(
            left,
            self.left_hand_state,
            self.left_hand_config,
            dt_s,
        )
        right_scales = self._hand_scales(
            right,
            self.right_hand_state,
            self.right_hand_config,
            dt_s,
        )
        bilateral_left, bilateral_right = self._bilateral_scales(left, right, dt_s)
        scales = tuple(
            [scale * bilateral_left for scale in left_scales]
            + [scale * bilateral_right for scale in right_scales]
        )
        self.states = [
            self._integrate(state, cfg, scale, dt_s)
            for state, cfg, scale in zip(updated, self.configs, scales)
        ]
        return tuple(state.copy() for state in self.states)


class NKeyKeyboard:
    def __init__(self, key_names: tuple[str, ...], config: KeyConfig | None = None) -> None:
        self.key_names = key_names
        self.config = config or KeyConfig()
        self.reset()

    def reset(self) -> None:
        self.states = [KeyState() for _ in self.key_names]

    def step(self, positions: tuple[float, ...]) -> list[tuple[str, KeyEvent]]:
        if len(positions) != len(self.key_names):
            raise ValueError("position count must match key count")
        events: list[tuple[str, KeyEvent]] = []
        for index, (name, position) in enumerate(zip(self.key_names, positions)):
            state = self.states[index]
            if not state.pressed and position >= self.config.actuation_m:
                state.pressed = True
                events.append((name, KeyEvent.DOWN))
            elif state.pressed and position <= self.config.reset_m:
                state.pressed = False
                events.append((name, KeyEvent.UP))
        return events


class NKeyMotorEnv:
    """1 kHz configurable-finger body behind a 100 Hz control contract."""

    def __init__(
        self,
        key_count: int,
        *,
        control_dt_s: float = 0.010,
        physics_dt_s: float = 0.001,
        key_config: KeyConfig | None = None,
    ) -> None:
        if not isfinite(physics_dt_s):
            raise ValueError("physics_dt_s must be finite")
        if not isfinite(control_dt_s):
            raise ValueError("control_dt_s must be finite")
        if physics_dt_s <= 0.0:
            raise ValueError("physics_dt_s must be positive")
        if control_dt_s < physics_dt_s:
            raise ValueError("control_dt_s must be at least one physics step")
        substeps = round(control_dt_s / physics_dt_s)
        if abs(substeps * physics_dt_s - control_dt_s) > 1e-12:
            raise ValueError("control_dt_s must be an integer multiple of physics_dt_s")

        self.key_names = n_key_names(key_count)
        self.key_count = key_count
        self.body = NKeyBody(self.key_count)
        self.keyboard = NKeyKeyboard(self.key_names, key_config)
        self.control_dt_s = float(control_dt_s)
        self.physics_dt_s = float(physics_dt_s)
        self.physics_substeps = int(substeps)
        self.time_s = 0.0
        self.capture_physics_trace = False
        self.last_transition: NKeyTransition | None = None

    def reset(self) -> NKeyObservation:
        self.body.reset()
        self.keyboard.reset()
        self.time_s = 0.0
        self.last_transition = None
        return self.observe()

    def observe(self) -> NKeyObservation:
        return NKeyObservation(
            key_names=self.key_names,
            positions_m=tuple(state.position_m for state in self.body.states),
            velocities_m_s=tuple(state.velocity_m_s for state in self.body.states),
            pressed_flags=tuple(state.pressed for state in self.keyboard.states),
        )

    def diagnostics(self) -> NKeyDiagnostics:
        return NKeyDiagnostics(
            time_s=self.time_s,
            activations=tuple(state.activation for state in self.body.states),
            finger_fatigue=tuple(state.fatigue for state in self.body.states),
            left_hand_fatigue=self.body.left_hand_state.fatigue,
            right_hand_fatigue=self.body.right_hand_state.fatigue,
            left_hand_coordination=self.body.left_hand_state.coordination,
            right_hand_coordination=self.body.right_hand_state.coordination,
            bilateral_coordination=self.body.bilateral_state.coordination,
        )

    def step(self, action: NKeyAction) -> NKeyTransition:
        values = action.as_tuple()
        if len(values) != self.key_count:
            raise ValueError(f"expected {self.key_count} action values, got {len(values)}")
        if any(not isfinite(value) for value in values):
            raise ValueError("N-key motor commands must be finite")
        clamped = NKeyAction(tuple(max(-1.0, min(1.0, value)) for value in values))
        if not self.capture_physics_trace:
            return self._step_held_exact(clamped)
        events: list[TimedKeyEvent] = []
        physics_samples: list[NKeyPhysicsSample] = []
        for _ in range(self.physics_substeps):
            states = self.body.step(clamped, self.physics_dt_s)
            self.time_s += self.physics_dt_s
            positions = tuple(state.position_m for state in states)
            raw = self.keyboard.step(positions)
            events.extend(TimedKeyEvent(self.time_s, key, event) for key, event in raw)
            if self.capture_physics_trace:
                physics_samples.append(
                    NKeyPhysicsSample(
                        time_s=float(self.time_s),
                        positions_m=positions,
                        events=tuple(raw),
                    )
                )
        transition = NKeyTransition(
            self.observe(),
            tuple(events),
            self.diagnostics(),
            tuple(physics_samples),
        )
        self.last_transition = transition
        return transition

    def _step_held_exact(self, action: NKeyAction) -> NKeyTransition:
        """Integrate one held control command without 1 kHz object churn.

        This is the N-key equivalent of ``motor.fast.step_held_exact``.  It
        deliberately retains the scalar equation and reduction order from
        ``NKeyBody.step`` and samples every keyboard threshold at 1 ms.  Only
        intermediate ``FingerState``/tuple/list materialization is removed;
        the public state is materialized once after the held command.
        """
        body = self.body
        dt_s = self.physics_dt_s
        commands = action.values
        configs = body.configs
        count = self.key_count
        split = body.per_hand

        positions = [state.position_m for state in body.states]
        velocities = [state.velocity_m_s for state in body.states]
        activations = [state.activation for state in body.states]
        fatigues = [state.fatigue for state in body.states]
        last_signs = [state.last_nonzero_command_sign for state in body.states]
        pressed = [state.pressed for state in self.keyboard.states]

        left_hand_fatigue = body.left_hand_state.fatigue
        right_hand_fatigue = body.right_hand_state.fatigue
        left_hand_coordination = body.left_hand_state.coordination
        right_hand_coordination = body.right_hand_state.coordination
        bilateral_coordination = body.bilateral_state.coordination
        left_cfg = body.left_hand_config
        right_cfg = body.right_hand_config
        bilateral_cfg = body.bilateral_config
        coordination_positions = body._coordination_positions(split)
        actuation_m = self.keyboard.config.actuation_m
        reset_m = self.keyboard.config.reset_m
        events: list[TimedKeyEvent] = []

        for _ in range(self.physics_substeps):
            for index in range(count):
                cfg = configs[index]
                command = commands[index]
                activation = (
                    activations[index]
                    + (command - activations[index]) * dt_s / cfg.activation_tau_s
                )
                activation = max(-1.0, min(1.0, activation))
                activations[index] = activation

                threshold = max(0.0, min(0.99, cfg.fatigue_threshold))
                normalized = max(0.0, abs(activation) - threshold) / (1.0 - threshold)
                load = normalized ** max(1.0, cfg.fatigue_exponent)
                command_sign = body._command_sign(command)
                reversed_direction = (
                    command_sign != 0
                    and last_signs[index] != 0
                    and command_sign != last_signs[index]
                )
                reversal_cost = (
                    cfg.switch_fatigue_per_reversal if reversed_direction else 0.0
                )
                fatigue = (
                    fatigues[index]
                    + (cfg.fatigue_gain_s * load
                       - cfg.fatigue_recovery_s * fatigues[index]) * dt_s
                    + reversal_cost
                )
                fatigues[index] = max(0.0, min(1.0, fatigue))
                if command_sign != 0:
                    last_signs[index] = command_sign

            hand_scales: list[float] = []
            for start, hand_cfg, hand_fatigue, hand_coordination in (
                (0, left_cfg, left_hand_fatigue, left_hand_coordination),
                (split, right_cfg, right_hand_fatigue, right_hand_coordination),
            ):
                stop = start + split
                demand = sum(abs(activations[index]) for index in range(start, stop))
                normalized_demand = min(1.0, demand / max(hand_cfg.capacity, 1e-9))
                threshold = max(0.0, min(0.99, hand_cfg.fatigue_threshold))
                normalized = max(0.0, normalized_demand - threshold) / (1.0 - threshold)
                load = normalized ** max(1.0, hand_cfg.fatigue_exponent)
                hand_fatigue = max(
                    0.0,
                    min(
                        1.0,
                        hand_fatigue
                        + (hand_cfg.fatigue_gain_s * load
                           - hand_cfg.fatigue_recovery_s * hand_fatigue) * dt_s,
                    ),
                )
                usable_capacity = hand_cfg.capacity * (1.0 - hand_fatigue)
                capacity_scale = 1.0
                if demand > usable_capacity and demand > 0.0:
                    capacity_scale = usable_capacity / demand

                positive = tuple(
                    max(0.0, activations[index]) for index in range(start, stop)
                )
                press_demand = sum(positive)
                coordination = [1.0] * split
                if press_demand > 1e-9 and split > 1:
                    target = sum(
                        position * value
                        for position, value in zip(coordination_positions, positive)
                    ) / press_demand
                    tau = max(hand_cfg.switch_tau_s, dt_s)
                    hand_coordination += (
                        target - hand_coordination
                    ) * dt_s / tau
                    hand_coordination = max(-1.0, min(1.0, hand_coordination))
                    floor = max(0.0, min(1.0, hand_cfg.coordination_floor))
                    coordination = [
                        floor
                        + (1.0 - floor)
                        * 0.5
                        * (1.0 + hand_coordination * position)
                        for position in coordination_positions
                    ]
                hand_scales.extend(
                    capacity_scale * scale for scale in coordination
                )
                if start == 0:
                    left_hand_fatigue = hand_fatigue
                    left_hand_coordination = hand_coordination
                else:
                    right_hand_fatigue = hand_fatigue
                    right_hand_coordination = hand_coordination

            left_press = sum(max(0.0, activations[index]) for index in range(split))
            right_press = sum(
                max(0.0, activations[index]) for index in range(split, count)
            )
            bilateral_demand = left_press + right_press
            bilateral_left = 1.0
            bilateral_right = 1.0
            if bilateral_demand > 1e-9:
                target = (left_press - right_press) / bilateral_demand
                tau = max(bilateral_cfg.switch_tau_s, dt_s)
                bilateral_coordination += (
                    target - bilateral_coordination
                ) * dt_s / tau
                bilateral_coordination = max(
                    -1.0, min(1.0, bilateral_coordination)
                )
                floor = max(0.0, min(1.0, bilateral_cfg.coordination_floor))
                left_affinity = 0.5 * (1.0 + bilateral_coordination)
                right_affinity = 1.0 - left_affinity
                bilateral_left = floor + (1.0 - floor) * left_affinity
                bilateral_right = floor + (1.0 - floor) * right_affinity

            for index in range(count):
                cfg = configs[index]
                scale = hand_scales[index] * (
                    bilateral_left if index < split else bilateral_right
                )
                muscle_force = (
                    cfg.max_force_n
                    * (1.0 - fatigues[index])
                    * activations[index]
                    * scale
                )
                spring_force = -cfg.spring_n_m * (
                    positions[index] - cfg.rest_position_m
                )
                damping_force = -cfg.damping_n_s_m * velocities[index]
                acceleration = (
                    muscle_force + spring_force + damping_force
                ) / cfg.mass_kg
                velocity = velocities[index] + acceleration * dt_s
                position = positions[index] + velocity * dt_s
                if position <= cfg.min_position_m:
                    position = cfg.min_position_m
                    velocity = max(0.0, velocity)
                elif position >= cfg.max_position_m:
                    position = cfg.max_position_m
                    velocity = min(0.0, velocity)
                positions[index] = position
                velocities[index] = velocity

            self.time_s += dt_s
            for index, (name, position) in enumerate(zip(self.key_names, positions)):
                if not pressed[index] and position >= actuation_m:
                    pressed[index] = True
                    events.append(TimedKeyEvent(self.time_s, name, KeyEvent.DOWN))
                elif pressed[index] and position <= reset_m:
                    pressed[index] = False
                    events.append(TimedKeyEvent(self.time_s, name, KeyEvent.UP))

        body.states = [
            FingerState(
                positions[index], velocities[index], activations[index],
                fatigues[index], commands[index], last_signs[index]
            )
            for index in range(count)
        ]
        body.left_hand_state.fatigue = left_hand_fatigue
        body.right_hand_state.fatigue = right_hand_fatigue
        body.left_hand_state.coordination = left_hand_coordination
        body.right_hand_state.coordination = right_hand_coordination
        body.bilateral_state.coordination = bilateral_coordination
        for index, value in enumerate(pressed):
            self.keyboard.states[index].pressed = value

        transition = NKeyTransition(
            self.observe(), tuple(events), self.diagnostics(), ()
        )
        self.last_transition = transition
        return transition
