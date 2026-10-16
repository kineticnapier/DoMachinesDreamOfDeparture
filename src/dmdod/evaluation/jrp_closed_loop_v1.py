from __future__ import annotations

"""Actual CPU Motor feedback loop with *read-only* JRP observable inputs.

This harness evaluates a caller-supplied policy. It does not secretly load or
pretend to run trained Student weights. The policy receives only the 263D HUD,
currently displayed (possibly stale) JRP text and two-stream Timing forecast.
True target labels and real Motor hit counters are read only AFTER its action.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from dmdod.envs.jrp_overlay_diagnostic import JrpOverlayDiagnosticNKeyEnv
from dmdod.envs.n_key import encode_n_key_hud_real_chart_observation
from dmdod.motor.n_key import NKeyAction
from dmdod.planning.jrp_cbpm_anchor_v1 import CausalJrpTimingBridge, JrpJointTimingUpdate


@dataclass(frozen=True, slots=True)
class CausalJrpPolicyInput:
    frame: int
    hud: tuple[float, ...]
    jrp_text: tuple[str, str, str] | None
    timing: JrpJointTimingUpdate


class CausalJrpPolicy(Protocol):
    def reset(self) -> None: ...
    def act(self, observation: CausalJrpPolicyInput) -> NKeyAction: ...


@dataclass(frozen=True, slots=True)
class ClosedLoopJrpResult:
    policy_label: str
    jrp_mode: str
    visible_delay_frames: int
    frames: int
    actual_hits: int
    actual_misses: int
    actual_too_early: int
    actual_overload: bool
    physical_keydowns: int
    actual_display_hit_hooks: int
    observed_text_changes: int
    anchored_frames: int
    jrp_added_frames: int
    unique_jrp_candidates: int
    matched_jrp_candidates: int
    false_jrp_candidates: int
    unique_jrp_forecasts: int
    matched_jrp_forecasts: int
    false_jrp_forecasts: int
    unique_baseline_elapsed: int
    matched_baseline_elapsed: int
    cbpm_statuses: tuple[tuple[str, int], ...]
    first_jrp_due_s: tuple[float, ...]
    # These fields summarize evaluator-only truth. Never pass result to policy.


def _match_unique(forecasts: list[float], targets: tuple[float, ...], *, tol=50e-6):
    unmatched = set(range(len(targets)))
    matches = 0
    for due in sorted(forecasts):
        if not unmatched:
            break
        near = min(unmatched, key=lambda i: abs(targets[i] - due))
        if abs(targets[near] - due) <= tol:
            matches += 1
            unmatched.remove(near)
    return matches


def run_jrp_closed_loop(
    segment, policy: CausalJrpPolicy, *, policy_label: str,
    jrp_mode: str = 'normal', visible_delay_frames: int = 0,
    max_frames: int | None = None,
) -> ClosedLoopJrpResult:
    """Evaluate one actual physically stepped environment, no oracle actions.

    Delay means JRP display reaches Student later; the game-side renderer still
    receives actual Motor-hit hooks at the original time. HUD is never delayed.
    The evaluator has target times, but the policy is only passed immutable
    CausalJrpPolicyInput, so it cannot use them via the intended interface.
    """
    if type(visible_delay_frames) is not int or not 0 <= visible_delay_frames <= 10:
        raise ValueError('visible_delay_frames must be integer 0..10')
    if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
        raise ValueError('invalid frame limit')

    env = JrpOverlayDiagnosticNKeyEnv(
        segment, key_count=8, jrp_mode=jrp_mode, control_dt_s=.010,
        physics_dt_s=.001, capture_overload_trace=True,
    )
    obs = env.reset()
    bridge = CausalJrpTimingBridge()
    policy.reset()
    text_history: list[tuple[str, str, str] | None] = []
    previous_seen_text = None
    observations = 0
    display_changes = 0
    jrp_adds = 0
    anchored = 0
    cbpm_statuses: dict[str, int] = {}
    seen_jrp: list[float] = []
    seen_candidates: list[float] = []
    seen_base: list[float] = []
    frames = 0
    limit = max_frames or 100_000
    for f in range(limit):
        frames = f + 1
        # Renderer state may know the true chart; its *visible numbers* are
        # the only values copied across the policy/renderer boundary.
        text_history.append(env.overlay_visible)
        read_at = max(0, f - visible_delay_frames)
        seen_text = text_history[read_at]
        if f > 0 and seen_text != previous_seen_text:
            display_changes += 1
        previous_seen_text = seen_text
        hud = tuple(encode_n_key_hud_real_chart_observation(obs))
        update = bridge.observe(f, hud, seen_text)
        cbpm_statuses[update.cbpm_update.status] = cbpm_statuses.get(update.cbpm_update.status, 0) + 1
        anchored += update.cbpm_update.eligible
        jrp_adds += update.jrp_added
        for e in update.base_update.elapsed:
            seen_base.append(e.due_s)
        if update.cbpm_update.eligible and update.cbpm_update.due_s is not None:
            due = update.cbpm_update.due_s
            if not any(abs(x-due) <= .00075 for x in seen_candidates):
                seen_candidates.append(due)
        if update.jrp_added and update.cbpm_update.due_s is not None:
            due = update.cbpm_update.due_s
            if not any(abs(x - due) <= .00075 for x in seen_jrp):
                seen_jrp.append(due)
        action = policy.act(CausalJrpPolicyInput(f, hud, seen_text, update))
        if not isinstance(action, NKeyAction) or len(action.values) != 8 or not all(
            isfinite(v) and -1. <= v <= 1. for v in action.values
        ):
            raise ValueError('Student must return a finite 8-key action')
        transition = env.step(action)
        observations += 1
        obs = transition.observation
        if transition.done:
            break
    else:
        raise RuntimeError('episode truncated without done; increase max_frames')

    # Truth access begins here. The policy did not receive target timestamps,
    # source floor identities, hit-counter changes or raw renderer metadata.
    stats = env.stats
    target_times = tuple(t.episode_time_s for t in segment.targets)
    actual_reported = [t for t in seen_jrp if 0 < t <= (frames*.01+1e-7)]
    candidate_reported = [t for t in seen_candidates if 0 < t <= (frames*.01+1e-7)]
    baseline_reported = list(seen_base)
    return ClosedLoopJrpResult(
        policy_label=policy_label, jrp_mode=jrp_mode,
        visible_delay_frames=visible_delay_frames, frames=frames,
        actual_hits=stats.hits, actual_misses=stats.misses,
        actual_too_early=stats.too_early_presses, actual_overload=stats.overloaded,
        physical_keydowns=env.physical_keydowns,
        actual_display_hit_hooks=env.overlay_hit_hook_count,
        observed_text_changes=display_changes,
        anchored_frames=anchored, jrp_added_frames=jrp_adds,
        unique_jrp_candidates=len(candidate_reported),
        matched_jrp_candidates=_match_unique(candidate_reported, target_times),
        false_jrp_candidates=len(candidate_reported)-_match_unique(candidate_reported, target_times),
        unique_jrp_forecasts=len(actual_reported),
        matched_jrp_forecasts=_match_unique(actual_reported, target_times),
        false_jrp_forecasts=len(actual_reported)-_match_unique(actual_reported, target_times),
        unique_baseline_elapsed=len(baseline_reported),
        matched_baseline_elapsed=_match_unique(baseline_reported, target_times),
        cbpm_statuses=tuple(sorted(cbpm_statuses.items())),
        first_jrp_due_s=tuple(actual_reported[:12]),
    )


class IdlePolicy:
    """Causal never-press baseline. Expect no JRP Hit hooks."""

    def reset(self):
        pass

    def act(self, observation: CausalJrpPolicyInput) -> NKeyAction:
        return NKeyAction((-1.,)*8)


class ScheduledMotorFixture:
    """TEST ONLY: preselected pulse windows; NOT an autonomous Student.

    The windows can be deliberately chosen using chart truth before creation;
    never use this controller to claim HUD-only Motor competence.
    """

    def __init__(self, windows: tuple[tuple[float, float], ...]):
        self.windows = tuple(windows)

    def reset(self):
        pass

    def act(self, observation: CausalJrpPolicyInput) -> NKeyAction:
        t = observation.frame*.010
        down = any(start <= t < stop for start, stop in self.windows)
        return NKeyAction(((1.,) if down else (-1.,)) + (-1.,)*7)
