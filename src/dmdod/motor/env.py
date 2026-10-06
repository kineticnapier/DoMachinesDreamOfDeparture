from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from dmdod.motor.fast import step_held_exact
from dmdod.motor.keyboard import KeyEvent
from dmdod.profiles import personal_blue_switch_v0_1
from dmdod.envs.simulator import Simulation
from dmdod.motor.body import TwoFingerBody


@dataclass(frozen=True)
class MotorAction:
    """Continuous motor commands issued by a policy.

    Values are clamped to [-1, 1] before reaching the body.  A policy never
    emits digital key presses directly.
    """

    left: float
    right: float


@dataclass(frozen=True)
class MotorObservation:
    """Agent-visible state.

    Exact chart timestamps, simulator time, fatigue variables, coordination
    internals, and evaluator state are deliberately absent.  This is the base
    interface that later perception modules can extend without leaking perfect
    chart timing.
    """

    left_position_m: float
    right_position_m: float
    left_velocity_m_s: float
    right_velocity_m_s: float
    left_pressed: bool
    right_pressed: bool


@dataclass(frozen=True)
class TimedKeyEvent:
    """Privileged evaluator event; do not feed this object to the policy."""

    time_s: float
    key: str
    event: KeyEvent


@dataclass(frozen=True)
class MotorDiagnostics:
    """Privileged diagnostics for experiments and plots, not policy input."""

    time_s: float
    left_activation: float
    right_activation: float
    left_fatigue: float
    right_fatigue: float
    hand_fatigue: float
    hand_coordination: float
    bilateral_coordination: float


@dataclass(frozen=True)
class MotorTransition:
    observation: MotorObservation
    evaluator_events: tuple[TimedKeyEvent, ...]
    diagnostics: MotorDiagnostics


class MotorPolicy(Protocol):
    def act(self, observation: MotorObservation) -> MotorAction: ...


class MotorEnv:
    """RL-facing motor wrapper around the 1 kHz physical simulator.

    Policies act at ``control_dt_s`` while physics remains at the simulator's
    fixed 1 ms step.  The command is held between policy decisions.  The exact
    fast path integrates all held-command substeps without allocating public
    simulator result objects for every 1 ms tick.
    """

    def __init__(self, *, same_hand: bool = True, control_dt_s: float = 0.010) -> None:
        body_config = personal_blue_switch_v0_1(same_hand=same_hand)
        self.sim = Simulation(body=TwoFingerBody(config=body_config))
        physics_dt = self.sim.config.dt_s
        if control_dt_s < physics_dt:
            raise ValueError("control_dt_s must be at least one physics step")
        substeps = round(control_dt_s / physics_dt)
        if abs(substeps * physics_dt - control_dt_s) > 1e-12:
            raise ValueError("control_dt_s must be an integer multiple of physics dt")
        self.control_dt_s = control_dt_s
        self.physics_substeps = substeps

    def reset(self) -> MotorObservation:
        self.sim.reset()
        return self.observe()

    def observe(self) -> MotorObservation:
        return MotorObservation(
            left_position_m=self.sim.body.left.position_m,
            right_position_m=self.sim.body.right.position_m,
            left_velocity_m_s=self.sim.body.left.velocity_m_s,
            right_velocity_m_s=self.sim.body.right.velocity_m_s,
            left_pressed=self.sim.keyboard.left.pressed,
            right_pressed=self.sim.keyboard.right.pressed,
        )

    def diagnostics(self) -> MotorDiagnostics:
        return MotorDiagnostics(
            time_s=self.sim.time_s,
            left_activation=self.sim.body.left.activation,
            right_activation=self.sim.body.right.activation,
            left_fatigue=self.sim.body.left.fatigue,
            right_fatigue=self.sim.body.right.fatigue,
            hand_fatigue=self.sim.body.shared_hand.fatigue,
            hand_coordination=self.sim.body.shared_hand.coordination,
            bilateral_coordination=self.sim.body.bilateral_state.coordination,
        )

    def step(self, action: MotorAction) -> MotorTransition:
        left = max(-1.0, min(1.0, action.left))
        right = max(-1.0, min(1.0, action.right))
        raw_events = step_held_exact(
            self.sim,
            left,
            right,
            self.physics_substeps,
        )
        events = tuple(
            TimedKeyEvent(time_s, key, event)
            for time_s, key, event in raw_events
        )
        return MotorTransition(self.observe(), events, self.diagnostics())


class LeftThresholdReflexPolicy:
    """Smoke-test policy using only digital key state from MotorObservation."""

    def act(self, observation: MotorObservation) -> MotorAction:
        return MotorAction(-1.0 if observation.left_pressed else 1.0, 0.0)
