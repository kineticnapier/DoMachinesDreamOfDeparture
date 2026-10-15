from __future__ import annotations

"""Stage C v2.8: full-pulse, globally evaluated, feasible SA with repair.

Optimize *complete* positive-press-plus-release motor waveforms, not only the
left edge of positive press frames. The v2.7 verified SAFE850 action tape
remains the immutable initial feasible solution and the mandatory fallback.
All comparisons use the original 1ms physics to complete termination.
"""

from dataclasses import dataclass
from math import exp, isfinite
from random import Random

from . import prestige1 as p
from . import prestige1_global_annealing as v27

# Single declared exploratory protocol. No hyperparameter search.
PROPOSALS_PER_ANCHOR = 96
MAX_SHIFT_TICKS = 8
MAX_GROUP_SIZE = 3
REPAIR_TRIALS_PER_UNSAFE = 3
START_TEMPERATURE_MS = 3.0
END_TEMPERATURE_MS = 0.15
EPS = 1e-9


@dataclass(frozen=True)
class Pulse:
    key_index: int
    start: int
    positive_end: int
    tail_end: int
    max_shift: int


def pulse_windows(
    baseline: tuple[tuple[float, ...], ...],
) -> tuple[Pulse, ...]:
    """Identify disjoint positive+immediate-negative waveforms per finger.

    A shifted pulse retains BOTH the number of +1 frames and the original
    following consecutive -1 frames. Rightward shift is bounded by the
    contiguous zero-command gap following the release tail for that finger.
    """
    if not baseline:
        raise ValueError("Empty source action tape")
    if any(len(frame) != p.KEY_COUNT
           or any(value not in (-1., 0., 1.) for value in frame)
           for frame in baseline):
        raise ValueError("Expected 8-key ternary action tape")
    n = len(baseline)
    windows = []
    for key in range(p.KEY_COUNT):
        cursor = 0
        while cursor < n:
            if baseline[cursor][key] != 1.:
                cursor += 1
                continue
            start = cursor
            while cursor < n and baseline[cursor][key] == 1.:
                cursor += 1
            positive_end = cursor
            while cursor < n and baseline[cursor][key] == -1.:
                cursor += 1
            tail_end = cursor
            gap_end = cursor
            while gap_end < n and baseline[gap_end][key] == 0.:
                gap_end += 1
            max_shift = min(MAX_SHIFT_TICKS, gap_end - tail_end)
            if max_shift:
                windows.append(Pulse(
                    key, start, positive_end, tail_end, max_shift))
    return tuple(sorted(windows, key=lambda item: (item.start, item.key_index)))


def shifted_tape(
    baseline: tuple[tuple[float, ...], ...],
    windows: tuple[Pulse, ...],
    shifts: tuple[int, ...],
) -> tuple[tuple[float, ...], ...]:
    """Re-synthesize simultaneous pulse shifts without overlapping commands."""
    if len(windows) != len(shifts):
        raise ValueError("Pulse schedule length mismatch")
    actions = [list(frame) for frame in baseline]
    # Remove all moved source waveforms before inserting the shifted copies.
    for window, delta in zip(windows, shifts):
        if type(delta) is not int or not 0 <= delta <= window.max_shift:
            raise ValueError("Pulse shift exceeds safe structural gap")
        if not 0 <= window.start < window.positive_end <= window.tail_end <= len(actions):
            raise ValueError("Invalid pulse waveform boundary")
        if delta:
            for frame in range(window.start, window.tail_end):
                actions[frame][window.key_index] = 0.
    for window, delta in zip(windows, shifts):
        if not delta:
            continue
        for old_frame in range(window.start, window.tail_end):
            new_frame = old_frame + delta
            if actions[new_frame][window.key_index] != 0.:
                raise ValueError("Pulse shift collides with another motor command")
            actions[new_frame][window.key_index] = baseline[old_frame][window.key_index]
    return tuple(tuple(frame) for frame in actions)


def changed_first(
    old_actions: tuple[tuple[float, ...], ...],
    new_actions: tuple[tuple[float, ...], ...],
) -> int | None:
    if len(old_actions) != len(new_actions):
        raise ValueError("Cannot compare unequal motor episode durations")
    for index, (left, right) in enumerate(zip(old_actions, new_actions)):
        if left != right:
            return index
    return None


def mutate(
    shifts: tuple[int, ...],
    windows: tuple[Pulse, ...],
    rng: Random,
) -> tuple[int, ...] | None:
    if len(shifts) != len(windows):
        raise ValueError("Pulse state has incorrect dimension")
    if not windows:
        return None
    position = rng.randrange(len(windows))
    group = rng.choice((1, 1, 2, 3))
    chosen = range(position, min(position + group, len(windows)))
    proposal = list(shifts)
    for index in chosen:
        max_shift = windows[index].max_shift
        delta = rng.choice((-2, -1, 1, 2))
        candidate = max(0, min(max_shift, proposal[index] + delta))
        if candidate == proposal[index]:
            candidate = (proposal[index] + 1) % (max_shift + 1)
        proposal[index] = candidate
    result = tuple(proposal)
    return result if result != shifts else None


def temperature(step: int, budget: int) -> float:
    if budget < 1 or not 0 <= step < budget:
        raise ValueError("Invalid cooling step")
    proportion = step / max(1, budget - 1)
    return START_TEMPERATURE_MS * (
        END_TEMPERATURE_MS / START_TEMPERATURE_MS) ** proportion


def accepts(current_mae: float, candidate_mae: float,
            temp: float, rng: Random) -> bool:
    if not all(isfinite(value) for value in (current_mae, candidate_mae, temp)):
        raise ValueError("Non-finite SA energy or temperature")
    if temp <= 0.:
        raise ValueError("Temperature must be positive")
    delta = candidate_mae - current_mae
    return delta <= 0. or rng.random() < exp(-delta / temp)


def missed_target_index(trial: v27.PhysicalReplay) -> int | None:
    missed = getattr(trial.env, "_missed", ())
    for i, flag in enumerate(missed):
        if flag:
            return i
    return None


def repair_neighbors(
    proposal: tuple[int, ...],
    accepted: tuple[int, ...],
    windows: tuple[Pulse, ...],
    trial: v27.PhysicalReplay,
    *, control_dt_s: float = p.CONTROL_DT,
) -> tuple[tuple[int, ...], ...]:
    """Deterministic local physical-miss repair using nearby full pulses.

    Unsafe states are NEVER put in the SA chain. Only a fully physically
    verified repaired trajectory may be considered for ordinary SA acceptance.
    """
    index = missed_target_index(trial)
    if index is None:
        return ()
    target = trial.env.segment.targets[index]
    deadline_frame = float(target.episode_time_s) / control_dt_s
    ranked = sorted(
        range(len(windows)),
        key=lambda j: (abs(
            windows[j].start - (deadline_frame - 12.)), j),
    )
    result = []
    seen = set()
    for j in ranked:
        original = proposal[j]
        for delta in (-1, 1):
            amended = original + delta
            if not 0 <= amended <= windows[j].max_shift:
                continue
            candidate = list(proposal)
            candidate[j] = amended
            candidate = tuple(candidate)
            if candidate == accepted or candidate in seen:
                continue
            seen.add(candidate)
            result.append(candidate)
            if len(result) >= REPAIR_TRIALS_PER_UNSAFE:
                return tuple(result)
    return tuple(result)


def evaluate_from_accepted(
    named,
    baseline: tuple[tuple[float, ...], ...],
    windows: tuple[Pulse, ...],
    shifts: tuple[int, ...],
    *,
    accepted_tape: tuple[tuple[float, ...], ...],
    accepted_replay: v27.PhysicalReplay,
):
    actions = shifted_tape(baseline, windows, shifts)
    changed_at = changed_first(accepted_tape, actions)
    if changed_at is None:
        return None
    result = v27.replay_commands(
        named, actions, parent=accepted_replay,
        first_changed_frame=changed_at)
    return actions, result


def search_chart(
    named,
    baseline: tuple[tuple[float, ...], ...],
    *, seed: int, proposals: int = PROPOSALS_PER_ANCHOR,
):
    if proposals < 0:
        raise ValueError("Negative proposal budget")
    windows = pulse_windows(baseline)
    zero = tuple(0 for _ in windows)
    initial = v27.replay_commands(named, baseline)
    if not initial.safe:
        raise RuntimeError("v2.7 physical starting plan is not SAFE")

    current = best = initial
    current_shifts = best_shifts = zero
    current_tape = best_tape = baseline
    rng = Random(seed)
    counts = {
        "main_proposals": 0, "physically_safe_main": 0,
        "rejected_unsafe_main": 0, "rejected_metropolis": 0,
        "accepted_uphill": 0, "accepted_non_uphill": 0,
        "best_updates": 0, "no_op": 0,
        "repair_attempts": 0, "repair_safe": 0,
        "repair_selected": 0, "unsafe_miss_main": 0,
        "unsafe_too_early_main": 0,
        "unsafe_overload_main": 0,
    }
    for step in range(proposals):
        proposal = mutate(current_shifts, windows, rng)
        if proposal is None:
            counts["no_op"] += 1
            continue
        candidate = evaluate_from_accepted(
            named, baseline, windows, proposal,
            accepted_tape=current_tape, accepted_replay=current)
        if candidate is None:
            counts["no_op"] += 1
            continue
        actions, trial = candidate
        counts["main_proposals"] += 1
        if not trial.safe:
            counts["rejected_unsafe_main"] += 1
            counts["unsafe_miss_main"] += int(trial.stats["misses"] > 0)
            counts["unsafe_too_early_main"] += int(trial.stats["too_early"] > 0)
            counts["unsafe_overload_main"] += int(trial.stats["overload"])
            # Attempt repair, but do not use unsafe states as a parent.
            for repaired in repair_neighbors(
                    proposal, current_shifts, windows, trial):
                evaluation = evaluate_from_accepted(
                    named, baseline, windows, repaired,
                    accepted_tape=current_tape, accepted_replay=current)
                if evaluation is None:
                    continue
                counts["repair_attempts"] += 1
                repaired_actions, repaired_trial = evaluation
                if repaired_trial.safe:
                    counts["repair_safe"] += 1
                    proposal, actions, trial = (
                        repaired, repaired_actions, repaired_trial)
                    counts["repair_selected"] += 1
                    break
            else:
                continue
        else:
            counts["physically_safe_main"] += 1

        assert current.objective_ms is not None and trial.objective_ms is not None
        if not accepts(
                current.objective_ms, trial.objective_ms,
                temperature(step, proposals), rng):
            counts["rejected_metropolis"] += 1
            continue

        if trial.objective_ms > current.objective_ms + EPS:
            counts["accepted_uphill"] += 1
        else:
            counts["accepted_non_uphill"] += 1
        current = trial
        current_tape = actions
        current_shifts = proposal
        assert best.objective_ms is not None
        if trial.objective_ms < best.objective_ms - EPS:
            best = trial
            best_tape = actions
            best_shifts = proposal
            counts["best_updates"] += 1

    verified = v27.replay_commands(named, best_tape)
    if (not verified.safe or verified.stats != best.stats
            or verified.hit_events != best.hit_events
            or verified.timing != best.timing
            or abs(verified.objective_ms - best.objective_ms) > EPS):
        raise RuntimeError("v2.8 cached trajectory diverged from full independent replay")
    if (verified.objective_ms is None or initial.objective_ms is None
            or verified.objective_ms > initial.objective_ms + EPS):
        raise RuntimeError("v2.8 SA best regressed below v2.7 safe starting plan")

    return {
        "baseline": initial, "best": verified,
        "best_commands": best_tape,
        "pulse_windows": windows, "best_shifts": best_shifts,
        "counts": counts, "seed": seed, "proposal_budget": proposals,
    }
