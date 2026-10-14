from __future__ import annotations

"""Stage C v2.6: fixed, Train-only timing-centered actuator Teacher proposal.

The original v2.4 safety filter is always executed first and left unchanged.
For each positive motor command, consider at most one 10ms defer action.
A deep-copied 1ms motor forecasts the candidate 10ms frame followed by the
unchanged safety action held to the frozen lead horizon. A defer is admitted
ONLY if every predicted Hit target is preserved, the number of missed/early
targets and stray KeyDowns does not increase, and total absolute Hit error
strictly improves without materially worsening any individual Hit.

This is evaluator-only privileged planning, not a Student feature or label.
Unseen future commands are unknown, so a forecast is not a guarantee.
"""

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from math import ceil, isfinite

from dmdod.adofai_rules import TimingJudgement
from dmdod.motor.keyboard import KeyEvent
from dmdod.motor.n_key import NKeyAction

from .prestige1_actuator_teacher import ActuatorAwareTeacherSafety
from .prestige1_teacher_quality import CommandTransitions

PHYSICS_TOLERANCE_MS = 1.01  # one 1ms simulation tick + numeric tolerance
MIN_IMPROVEMENT_MS = 1.0     # timing must improve at least one physics tick


@dataclass(frozen=True)
class ScheduledForecast:
    keydowns: tuple[tuple[float, str, int | None, str], ...]
    hit_errors_ms: tuple[tuple[int, float], ...]
    too_early: int
    misses: int
    too_late: int
    no_target: int


def forecast_first_frame(
    env, first: NKeyAction, follow: NKeyAction, *, lead_s: float,
) -> ScheduledForecast:
    """Copy the complete motor and chart-resolution state. No live mutation."""
    if not isfinite(float(lead_s)) or lead_s <= 0:
        raise ValueError("finite positive lead required")
    if len(first.values) != len(env.key_names) or len(follow.values) != len(env.key_names):
        raise ValueError("Motor command width mismatch")
    simulated = deepcopy(env.motor)
    targets = env.segment.targets
    used, missed = list(env._used), list(env._missed)
    if len(used) != len(targets) or len(missed) != len(targets):
        raise ValueError("Invalid target resolution state")

    steps = ceil((lead_s + simulated.control_dt_s) / simulated.control_dt_s)
    downs = []
    hits = []
    too_early = late = stray = new_misses = 0

    def expire(now):
        nonlocal new_misses
        for idx, target in enumerate(targets):
            if not used[idx] and not missed[idx] and (
                float(target.episode_time_s) + env._windows(target).pass_s < now
            ):
                missed[idx] = True
                new_misses += 1

    for frame in range(steps):
        transition = simulated.step(first if frame == 0 else follow)
        for event in transition.evaluator_events:
            expire(float(event.time_s))
            if event.event is not KeyEvent.DOWN:
                continue
            time_s = float(event.time_s)
            idx = next((i for i, (u, m) in enumerate(zip(used, missed))
                        if not u and not m), None)
            if idx is None:
                stray += 1
                downs.append((time_s, event.key, None, "all_resolved"))
                continue
            target = targets[idx]
            error_s = time_s - float(target.episode_time_s)
            judgement = env._classify(error_s, target)
            if judgement is TimingJudgement.TOO_EARLY:
                too_early += 1
            elif judgement is TimingJudgement.TOO_LATE:
                missed[idx] = True
                new_misses += 1
                late += 1
            else:
                used[idx] = True
                hits.append((idx, error_s * 1000.0))
            downs.append((time_s, event.key, idx, judgement.value))
        expire(float(transition.diagnostics.time_s))
    return ScheduledForecast(
        tuple(downs), tuple(hits), too_early, new_misses, late, stray
    )


def eligible_timing_defer(
    reference: ScheduledForecast, alternative: ScheduledForecast,
    *, physics_tolerance_ms: float = PHYSICS_TOLERANCE_MS,
    min_improvement_ms: float = MIN_IMPROVEMENT_MS,
) -> float | None:
    """Return decrease in aggregate |Hit timing error|, or reject completely.

    Forecasts are compared by *target identity*, not KeyDown order/key.
    A disappearance of a predicted Hit is a rejected defer, not 'progress'.
    """
    if not reference.hit_errors_ms:
        return None
    if (alternative.too_early > reference.too_early
            or alternative.misses > reference.misses
            or alternative.too_late > reference.too_late
            or alternative.no_target > reference.no_target):
        return None
    old = dict(reference.hit_errors_ms)
    new = dict(alternative.hit_errors_ms)
    if len(old) != len(reference.hit_errors_ms) or len(new) != len(alternative.hit_errors_ms):
        raise RuntimeError("Duplicate targets in physical Hit forecast")
    if set(old) != set(new):
        return None
    if any(abs(new[i]) > abs(old[i]) + physics_tolerance_ms for i in old):
        return None
    improvement = sum(abs(error) for error in old.values()) - sum(
        abs(error) for error in new.values())
    return improvement if improvement >= min_improvement_ms else None


class TimingCenteredActuatorTeacher:
    """Greedy, frozen one-step defer after the v2.4 safety shield.

    One-step delayed proposals are evaluated against the *same* v2.4 command
    sequence assumption. Timing delays use neutral (0) only; reverse (-1)
    commands remain exclusively the responsibility of the v2.4 safety shield.
    """

    def __init__(self, *, lead_s: float):
        if not isfinite(float(lead_s)) or lead_s <= 0:
            raise ValueError("finite positive lead required")
        self.lead_s = float(lead_s)
        self.safety = ActuatorAwareTeacherSafety(lead_s=lead_s)
        self.counts = Counter()
        self.safety_transitions = CommandTransitions(8)
        self.timing_transitions = CommandTransitions(8)
        self.changed_examples = []
        self.env = None

    def __call__(self, env, original: NKeyAction) -> NKeyAction:
        if self.env is not None and self.env is not env:
            raise RuntimeError("Timing controller cannot span different episodes")
        self.env = env
        safe = self.safety(env, original)
        self.counts["frames"] += 1
        self.safety_transitions.observe(safe.values)
        selected = safe
        # The v2.4 action is immutable for the entire counterfactual horizon.
        # Only the very next 10ms motor command can change.
        reference = forecast_first_frame(
            env, safe, safe, lead_s=self.lead_s)
        if reference.too_early or reference.misses or reference.too_late:
            self.counts["unsafe_reference_forecast_frames"] += 1
            self.timing_transitions.observe(selected.values)
            return selected
        self.counts["frames_with_forecast_hit"] += bool(reference.hit_errors_ms)
        changed_keys = set()

        while len(changed_keys) < len(env.key_names):
            best = None
            for index, key in enumerate(env.key_names):
                if index in changed_keys or selected.values[index] <= 0:
                    continue
                # Only defer fingers projected to strike *early* themselves.
                key_hits = {
                    target_idx for _, physical_key, target_idx, status
                    in reference.keydowns
                    if physical_key == key and target_idx is not None
                    and status not in (
                        TimingJudgement.TOO_EARLY.value,
                        TimingJudgement.TOO_LATE.value)
                }
                if not any(error < -MIN_IMPROVEMENT_MS
                           for target_idx, error in reference.hit_errors_ms
                           if target_idx in key_hits):
                    continue
                # Neutral delay only; do not introduce extra +1 -> -1
                # reversals beyond the unchanged v2.4 safety shield.
                for proposed_value in (0.0,):
                    values = list(selected.values)
                    values[index] = proposed_value
                    candidate = NKeyAction(tuple(values))
                    forecast = forecast_first_frame(
                        env, candidate, safe, lead_s=self.lead_s)
                    gain = eligible_timing_defer(reference, forecast)
                    if gain is None:
                        continue
                    # Maximize timing gain. If tied, neutral is preferable.
                    rank = (gain, int(proposed_value == 0.), -index)
                    if best is None or rank > best[0]:
                        best = (rank, candidate, forecast, index, proposed_value)
            if best is None:
                break
            _, selected, reference, index, value = best
            changed_keys.add(index)
            self.counts["deferred_key_commands"] += 1
            self.counts["neutral_defer_commands"] += int(value == 0.)
            self.counts["reverse_defer_commands"] += int(value == -1.)
        if changed_keys:
            self.counts["changed_frames"] += 1
            if len(self.changed_examples) < 30:
                self.changed_examples.append({
                    "time_s": float(env.privileged_episode_time_s()),
                    "keys": [env.key_names[index] for index in sorted(changed_keys)],
                    "safe_action_8k": list(safe.values),
                    "timing_action_8k": list(selected.values),
                })
        self.timing_transitions.observe(selected.values)
        return selected

    def statistics(self):
        keys = (
            "frames", "frames_with_forecast_hit", "unsafe_reference_forecast_frames",
            "changed_frames", "deferred_key_commands", "neutral_defer_commands",
            "reverse_defer_commands",
        )
        return {
            "timing": {key: int(self.counts[key]) for key in keys},
            "v24_shield": self.safety.statistics(),
            "safety_action_transitions": self.safety_transitions.summary(),
            "timing_action_transitions": self.timing_transitions.summary(),
            "examples": list(self.changed_examples),
        }
