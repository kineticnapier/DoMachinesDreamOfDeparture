from __future__ import annotations

"""Actual CPU Motor feedback loop with *read-only* JRP observable inputs.

This harness evaluates a caller-supplied policy. It does not secretly load or
pretend to run trained Student weights. The policy receives only the 263D HUD,
currently displayed (possibly stale) JRP text and two-stream Timing forecast.
True target labels and real Motor hit counters are read only AFTER its action.
"""

from dataclasses import dataclass
from pathlib import Path
from math import isfinite
from typing import Protocol

import torch

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
class StudentTimingEstimate:
    """A checkpoint Student's causal next-event estimate at one HUD frame."""

    frame: int
    predicted_due_s: float
    confidence: float
    chord_size: int


class FrozenPrestigeMotorPolicy:
    """Strict adapter from a frozen Prestige v2 checkpoint to the Motor loop.

    The neural network still receives exactly the historical 263D HUD.  JRP
    text and evaluator truth are deliberately not appended to that tensor.
    The separate ``observation.timing`` stream can be evaluated alongside the
    checkpoint output without silently changing its trained input contract.
    """

    CHECKPOINT_FORMAT = 'prestige1-bc-pretrained-transfer-v2'

    def __init__(self, model, *, device: torch.device | str = 'cpu'):
        self.model = model
        self.device = torch.device(device)
        if getattr(model, 'input_dim', None) != 263 or getattr(model, 'key_count', None) != 8:
            raise ValueError('Prestige policy requires the frozen 263D / 8-key contract')
        self.model.to(self.device)
        self.model.eval()
        self.state = None
        self.estimates: list[StudentTimingEstimate] = []

    @classmethod
    def load(cls, checkpoint: str | Path, core_path: str | Path, *,
             device: torch.device | str = 'cpu', expected_arm: str | None = None):
        """Load only a provenance-bearing Stage C v2 Motor checkpoint."""
        from dmdod.training.prestige1 import PrestigeMotorStudent, sha256

        checkpoint = Path(checkpoint)
        core_path = Path(core_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if not core_path.is_file():
            raise FileNotFoundError(core_path)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        required = {'format', 'stage', 'arm', 'model', 'seed', 'config_sha256',
                    'pretrained_sha256', 'core_sha256', 'updates'}
        if not isinstance(payload, dict) or required.difference(payload):
            raise RuntimeError('incomplete Prestige v2 checkpoint')
        if payload['format'] != cls.CHECKPOINT_FORMAT or payload['stage'] != 'C':
            raise RuntimeError('unsupported Prestige checkpoint provenance')
        if payload['arm'] not in ('control', 'prestige'):
            raise RuntimeError('invalid Prestige checkpoint arm')
        if expected_arm is not None and payload['arm'] != expected_arm:
            raise RuntimeError('Prestige checkpoint arm mismatch')
        if not isinstance(payload['model'], dict) or not payload['model']:
            raise RuntimeError('empty Prestige checkpoint model state')
        if payload['core_sha256'] != sha256(core_path):
            raise RuntimeError('Prestige checkpoint MaleCNS core mismatch')
        model = PrestigeMotorStudent(core_path)
        model.load_state_dict(payload['model'], strict=True)
        policy = cls(model, device=device)
        policy.checkpoint_arm = payload['arm']
        policy.checkpoint_path = checkpoint
        return policy

    def reset(self) -> None:
        self.state = self.model.initial_state(self.device)
        self.estimates.clear()

    @torch.no_grad()
    def act(self, observation: CausalJrpPolicyInput) -> NKeyAction:
        if self.state is None:
            raise RuntimeError('policy must be reset before act')
        if len(observation.hud) != 263:
            raise ValueError('Prestige Student visible HUD must remain 263D')
        x = torch.tensor(observation.hud, dtype=torch.float32, device=self.device)
        mean, _, _, self.state, raw = self.model.forward_step_with_timing(x, self.state)
        predicted_s = float(torch.sigmoid(raw[0]).item() * .200)
        confidence = float(torch.sigmoid(raw[1]).item())
        chord_size = int(raw[2:].argmax().item()) + 1
        estimate = StudentTimingEstimate(
            frame=observation.frame,
            predicted_due_s=observation.frame * .010 + predicted_s,
            confidence=confidence,
            chord_size=chord_size,
        )
        if not (isfinite(estimate.predicted_due_s) and
                isfinite(estimate.confidence) and 0. <= estimate.confidence <= 1.):
            raise RuntimeError('non-finite Prestige timing estimate')
        self.estimates.append(estimate)
        values = tuple(float(v) for v in torch.tanh(mean).detach().cpu())
        return NKeyAction(values)


@dataclass(frozen=True, slots=True)
class ClosedLoopJrpResult:
    policy_label: str
    jrp_mode: str
    jrp_visible: bool
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
    target_count: int
    matched_combined: int
    false_combined: int
    baseline_mae_ms: float | None
    combined_mae_ms: float | None
    mean_jrp_confidence: float | None
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


def _matched_errors_ms(forecasts: list[float], targets: tuple[float, ...], *, tol=50e-6):
    unmatched = set(range(len(targets)))
    errors = []
    for due in sorted(forecasts):
        if not unmatched:
            break
        near = min(unmatched, key=lambda i: abs(targets[i] - due))
        error = abs(targets[near] - due)
        if error <= tol:
            errors.append(error * 1000.)
            unmatched.remove(near)
    return (sum(errors) / len(errors)) if errors else None


def run_jrp_closed_loop(
    segment, policy: CausalJrpPolicy, *, policy_label: str,
    jrp_mode: str = 'normal', visible_delay_frames: int = 0,
    jrp_visible: bool = True,
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
    seen_jrp_confidence: list[float] = []
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
        seen_text = text_history[read_at] if jrp_visible else None
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
                seen_jrp_confidence.append(update.cbpm_update.confidence)
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
    combined_reported = list(baseline_reported)
    for due in actual_reported:
        if not any(abs(x - due) <= .00075 for x in combined_reported):
            combined_reported.append(due)
    matched_combined = _match_unique(combined_reported, target_times)
    return ClosedLoopJrpResult(
        policy_label=policy_label, jrp_mode=jrp_mode,
        jrp_visible=jrp_visible,
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
        target_count=len(target_times), matched_combined=matched_combined,
        false_combined=len(combined_reported)-matched_combined,
        baseline_mae_ms=_matched_errors_ms(baseline_reported, target_times),
        combined_mae_ms=_matched_errors_ms(combined_reported, target_times),
        mean_jrp_confidence=(sum(seen_jrp_confidence)/len(seen_jrp_confidence)
                             if seen_jrp_confidence else None),
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
