from __future__ import annotations

"""Prestige I v2.4: evaluator/Teacher-only actuator-aware safety candidate.

No Student inference or Stage A timing label imports this module.  The forecast
copies the *full motor state* (activation, velocity, fatigue, key hysteresis)
and steps the unchanged 1ms physics.  It never alters the live environment.
Future Teacher decisions are unknown: holding a candidate command across the
lookahead is an approximation, not a guarantee of physical safety.
"""

from copy import deepcopy
from dataclasses import dataclass
from math import ceil

from dmdod.adofai_rules import TimingJudgement
from dmdod.motor.keyboard import KeyEvent
from dmdod.motor.n_key import NKeyAction


@dataclass(frozen=True)
class MotorForecast:
    """Counterfactual KeyDown outcomes, in physical event order."""

    keydowns: tuple[tuple[float, str, int | None, str], ...]
    too_early_keys: tuple[str, ...]
    hits: int
    misses: int


def forecast_held_command(env, action: NKeyAction, *, lead_s: float) -> MotorForecast:
    """Project one constant action over lead_s + control_dt, evaluator-only.

    This deliberately uses the true target resolution and judgement contract,
    but the only side effect is on a copied motor.  In particular, a second
    KeyDown at the same physics timestamp observes any preceding valid Hit.
    """
    if lead_s <= 0:
        raise ValueError("positive Teacher lead required")
    if len(action.values) != len(env.key_names):
        raise ValueError("Motor action width mismatch")
    simulated = deepcopy(env.motor)
    targets = env.segment.targets
    used = list(env._used)
    missed = list(env._missed)
    if len(used) != len(targets) or len(missed) != len(targets):
        raise ValueError("Physics target state mismatch")
    steps = ceil((lead_s + simulated.control_dt_s) / simulated.control_dt_s)
    downs: list[tuple[float, str, int | None, str]] = []
    early: list[str] = []
    hit_count = 0
    miss_count = 0
    for _ in range(steps):
        transition = simulated.step(action)
        for event in transition.evaluator_events:
            if event.event is not KeyEvent.DOWN:
                continue
            now = float(event.time_s)
            # Mirror _expire_misses before the event: an overdue unresolved
            # target cannot consume a later KeyDown.
            for idx, target in enumerate(targets):
                if not used[idx] and not missed[idx] and (
                    float(target.episode_time_s) + env._windows(target).pass_s < now
                ):
                    missed[idx] = True
                    miss_count += 1
            idx = next((i for i in range(len(targets))
                        if not used[i] and not missed[i]), None)
            if idx is None:
                downs.append((now, event.key, None, "all_resolved"))
                continue
            target = targets[idx]
            judgement = env._classify(now - float(target.episode_time_s), target)
            if judgement is TimingJudgement.TOO_EARLY:
                early.append(event.key)
            elif judgement is TimingJudgement.TOO_LATE:
                missed[idx] = True
                miss_count += 1
            else:
                used[idx] = True
                hit_count += 1
            downs.append((now, event.key, idx, judgement.value))
    return MotorForecast(tuple(downs), tuple(early), hit_count, miss_count)


class ActuatorAwareTeacherSafety:
    """Minimal intervention to the existing reserved-target Teacher.

    The original Teacher proposes a full 8-Key command.  Only an individual
    finger predicted to press too early may be braked (-1), and only when
    resimulation removes at least one forecast TooEarly.  This protects
    pre-acceleration whenever its *predicted KeyDown* is inside the pass window.
    No hyperparameter sweep, privileged observation, or Student checkpoint.
    """

    def __init__(self, *, lead_s: float):
        if lead_s <= 0:
            raise ValueError("lead_s must be positive")
        self.lead_s = float(lead_s)
        self.frames = 0
        self.changed_frames = 0
        self.braked_keys = 0
        self.forecast_early_before = 0
        self.forecast_early_after = 0
        self.unavoidable_frames = 0

    def __call__(self, env, original: NKeyAction) -> NKeyAction:
        self.frames += 1
        current = original
        forecast = forecast_held_command(env, current, lead_s=self.lead_s)
        self.forecast_early_before += len(forecast.too_early_keys)
        altered = set()
        # Each iteration permanently brakes one additional key; at most K.
        while forecast.too_early_keys:
            best = None
            for key in dict.fromkeys(forecast.too_early_keys):
                index = env.key_names.index(key)
                if current.values[index] <= -1.0:
                    continue
                proposed = list(current.values)
                proposed[index] = -1.0
                candidate = NKeyAction(tuple(proposed))
                predicted = forecast_held_command(env, candidate, lead_s=self.lead_s)
                # Do not trade a reduced predicted early count for a larger
                # predicted miss count when another option may exist.
                rank = (len(predicted.too_early_keys),
                        predicted.misses, -predicted.hits)
                if len(predicted.too_early_keys) >= len(forecast.too_early_keys):
                    continue
                if best is None or rank < best[0]:
                    best = (rank, candidate, predicted, index)
            if best is None:
                break
            _, current, forecast, index = best
            altered.add(index)
        if altered:
            self.changed_frames += 1
            self.braked_keys += len(altered)
        self.forecast_early_after += len(forecast.too_early_keys)
        if forecast.too_early_keys:
            self.unavoidable_frames += 1
        return current

    def statistics(self) -> dict:
        return {
            "control_frames": self.frames,
            "changed_frames": self.changed_frames,
            "braked_key_commands": self.braked_keys,
            "predicted_too_early_before": self.forecast_early_before,
            "predicted_too_early_after": self.forecast_early_after,
            "remaining_risk_frames": self.unavoidable_frames,
        }
