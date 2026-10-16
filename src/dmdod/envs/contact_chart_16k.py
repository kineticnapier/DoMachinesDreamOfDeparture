from __future__ import annotations

"""Experimental 12-actuator/16-logical-contact ADOFAI chart environment.

An ordinary RealChartMotorEnv scorer receives *physical* coupled contact
KeyDown/KeyUp events. The frozen 8K HUD, student and evaluator are untouched.
No new same-frame Multipress rules are assumed.
"""

from dataclasses import dataclass

from dmdod.adofai_playable import PlayableChartSegment
from dmdod.envs.real_chart import RealChartMotorEnv, RealChartStep, RelativeVisibleFloor
from dmdod.motor.contact_16k import (
    ContactAction, ContactObservation, Coupled16ContactMotor, LOGICAL_KEYS,
)
from dmdod.motor.keyboard import KeyEvent


@dataclass(frozen=True, slots=True)
class ContactChartObservation:
    motor: ContactObservation
    orbiting_x: float
    orbiting_y: float
    floors: tuple[RelativeVisibleFloor, ...]


class CoupledContactRealChartEnv(RealChartMotorEnv):
    """Real chart with 16 switches, 12 physical Motor actuators.

    The evaluator is the existing RealChartMotorEnv scorer, including its
    known incomplete same-frame Multipress semantics. `last_events` and
    privileged target methods are diagnostics only, not policy inputs.
    """

    def __init__(self, segment: PlayableChartSegment, *,
                 control_dt_s: float = .010, physics_dt_s: float = .001,
                 **kwargs) -> None:
        super().__init__(segment, control_dt_s=control_dt_s, **kwargs)
        self.motor = Coupled16ContactMotor(control_dt_s=control_dt_s,
                                            physics_dt_s=physics_dt_s)
        self.key_names = LOGICAL_KEYS
        self.physical_keydowns = 0
        self.last_events = ()
        self.judgement_trace: list[dict] = []

    def reset(self) -> ContactChartObservation:
        self.physical_keydowns = 0
        self.last_events = ()
        self.judgement_trace = []
        return super().reset()

    def _observation(self, motor: ContactObservation,
                     episode_time_s: float) -> ContactChartObservation:
        geom = super()._observation(motor, episode_time_s)
        return ContactChartObservation(motor=motor, orbiting_x=geom.orbiting_x,
                                       orbiting_y=geom.orbiting_y,
                                       floors=geom.floors)

    def observe(self) -> ContactChartObservation:
        return self._observation(self.motor.observe(), self.motor.motor.time_s)

    def privileged_episode_time_s(self) -> float:
        return self.motor.motor.time_s

    def step(self, action: ContactAction) -> RealChartStep:
        if self._done:
            raise RuntimeError('episode is done; call reset() before step()')
        transition = self.motor.step(action)
        self.last_events = transition.logical_events
        reward = 0.
        for event in transition.logical_events:
            self._advance_overload(event.time_s)
            reward += self._expire_misses(event.time_s)
            if self._failed_on_miss:
                break
            if event.event is KeyEvent.DOWN:
                self.physical_keydowns += 1
                target_index = self._next_target_index()
                before = len(self._hit_margins)
                reward += self._score_event(event)
                self.judgement_trace.append({
                    'time_s': event.time_s, 'key': event.key,
                    'target_index': target_index,
                    'margin': (self._hit_margins[-1].value
                               if len(self._hit_margins) > before else None),
                })
            if self._overload.overloaded:
                break

        now = transition.diagnostics.time_s
        if not self._failed_on_miss and not self._overload.overloaded:
            self._advance_overload(now)
            reward += self._expire_misses(now)
        reward -= self.reward_config.effort_penalty * sum(abs(x) for x in action.commands)
        all_resolved = all(used or missed for used, missed in zip(self._used, self._missed))
        self._done = (self._failed_on_miss or self._overload.overloaded
                      or now >= self._episode_end_s or all_resolved)
        if self._done and now >= self._episode_end_s and not all_resolved:
            reward += self._expire_misses(float('inf'))
        self._total_reward += reward
        return RealChartStep(self._observation(transition.observation, now),
                             reward, self._done)
