from __future__ import annotations

"""Prestige I Stage C v2.7: global, physically verified Teacher annealing.

The frozen v2.4 all-Hit command trace is the only initial schedule. Each
schedule variable delays the *onset* of one finger's continuous +1 command
run while retaining its original ending and release commands. Multiple
neighboring press runs may be altered together. Every candidate is scored
with true 1ms physics through episode termination (not a forecast). Only
full-Hit, zero-TooEarly, zero-Overload candidates can enter the SA chain.
Neither the Student nor Validation/FINAL is used.
"""

from copy import deepcopy
from dataclasses import dataclass
from math import ceil, exp, isfinite
from random import Random

from dmdod.envs.n_key import DiagnosticHudNKeyRealChartMotorEnv
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG
from dmdod.motor.n_key import NKeyAction

from . import prestige1 as p
from . import prestige1_teacher_quality as quality

# Fixed, single exploratory protocol. These are NOT swept or optimized.
PROPOSALS_PER_ANCHOR = 96
MAX_DELAY_TICKS = 9
SNAPSHOT_STRIDE = 100
INITIAL_TEMPERATURE_MS = 2.0
FINAL_TEMPERATURE_MS = 0.15
NEIGHBOR_BLOCK_MAX = 3
EPS = 1e-9


@dataclass(frozen=True)
class PressRun:
    key_index: int
    start: int
    end: int  # exclusive
    max_delay: int


@dataclass
class PhysicalReplay:
    env: object
    checkpoints: dict[int, object]
    stats: dict
    hit_events: list[dict]
    timing: dict
    termination: dict
    safe: bool
    objective_ms: float | None


def press_runs(commands: tuple[tuple[float, ...], ...]) -> tuple[PressRun, ...]:
    """Independent scheduling variables for frozen +1 press intervals."""
    if not commands:
        raise ValueError("Empty baseline control trajectory")
    width = len(commands[0])
    if width != p.KEY_COUNT:
        raise ValueError("Expected original 8-key actions")
    if any(len(frame) != width or any(v not in (-1., 0., 1.) for v in frame)
           for frame in commands):
        raise ValueError("Baseline action alphabet/shape changed")
    result = []
    for key in range(width):
        start = None
        for i in range(len(commands) + 1):
            active = i < len(commands) and commands[i][key] > 0.
            if active and start is None:
                start = i
            elif not active and start is not None:
                length = i - start
                maximum = min(MAX_DELAY_TICKS, length - 1)
                if maximum:
                    result.append(PressRun(key, start, i, maximum))
                start = None
    return tuple(sorted(result, key=lambda run: (run.start, run.key_index)))


def realize_schedule(
    baseline: tuple[tuple[float, ...], ...],
    runs: tuple[PressRun, ...],
    delays: tuple[int, ...],
) -> tuple[tuple[float, ...], ...]:
    """Render bounded per-finger onset delays into 10ms motor commands."""
    if len(runs) != len(delays):
        raise ValueError("Timing schedule width mismatch")
    actions = [list(frame) for frame in baseline]
    for run, delay in zip(runs, delays):
        if not isinstance(delay, int) or not 0 <= delay <= run.max_delay:
            raise ValueError("Timing delay outside frozen schedule bounds")
        if not 0 <= run.start < run.end <= len(actions):
            raise ValueError("Out-of-range press scheduling interval")
        for frame in range(run.start, run.start + delay):
            if actions[frame][run.key_index] != 1.:
                raise RuntimeError("Overlapping/changed original positive press run")
            actions[frame][run.key_index] = 0.
    return tuple(tuple(frame) for frame in actions)


def earliest_difference(
    runs: tuple[PressRun, ...], before: tuple[int, ...], after: tuple[int, ...]
) -> int:
    """Earliest motor frame that changes when the timing schedule changes."""
    if not (len(before) == len(after) == len(runs)):
        raise ValueError("Timing schedule comparison length mismatch")
    locations = [
        run.start + min(previous, candidate)
        for run, previous, candidate in zip(runs, before, after)
        if previous != candidate
    ]
    if not locations:
        raise ValueError("No timing changes to re-evaluate")
    return min(locations)


def _new_env(named):
    env = DiagnosticHudNKeyRealChartMotorEnv(
        named.segment, key_count=p.KEY_COUNT, capture_overload_trace=True,
        control_dt_s=p.CONTROL_DT, physics_dt_s=p.PHYSICS_DT,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )
    env.reset()
    return env


def safe_physics(stats: dict, *, targets: int) -> bool:
    return (
        stats["hits"] == targets
        and stats["targets"] == targets
        and stats["misses"] == 0
        and stats["too_early"] == 0
        and stats["keydowns"] == targets
        and stats["unjudged"] == 0
        and stats["overload"] is False
    )


def replay_commands(
    named, commands: tuple[tuple[float, ...], ...], *, 
    parent: PhysicalReplay | None = None, first_changed_frame: int = 0,
) -> PhysicalReplay:
    """Whole-chart replay with exact copied prefix state (not truncated scoring).

    Once a schedule is accepted, cached simulator snapshots at 100-frame
    boundaries allow recomputation from the nearest *unchanged* prefix.
    The altered suffix always advances to true simulator termination.
    """
    if first_changed_frame < 0 or first_changed_frame > len(commands):
        raise ValueError("Invalid first affected frame")
    if parent is None:
        if first_changed_frame != 0:
            raise ValueError("Prefix replay requires an accepted parent")
        env = _new_env(named)
        start = 0
        checkpoints = {0: deepcopy(env)}
    else:
        eligible = [frame for frame in parent.checkpoints
                    if frame <= first_changed_frame]
        if not eligible:
            raise RuntimeError("Missing a prefix checkpoint")
        start = max(eligible)
        env = deepcopy(parent.checkpoints[start])
        checkpoints = {i: snapshot for i, snapshot in parent.checkpoints.items()
                       if i <= start}

    limit = ceil((named.segment.duration_s + 2.0) / p.CONTROL_DT) + 200
    neutral = NKeyAction((0.,) * p.KEY_COUNT)
    for frame in range(start, limit):
        action = NKeyAction(commands[frame]) if frame < len(commands) else neutral
        step = env.step(action)
        if step.done:
            break
        if (frame + 1) % SNAPSHOT_STRIDE == 0:
            checkpoints[frame + 1] = deepcopy(env)
    else:
        raise RuntimeError("Full-chart motor replay exceeded frame budget")

    trace = env.overload_trace
    if trace is None:
        raise RuntimeError("Missing true physical KeyDown trace")
    stats_obj = env.stats
    stats = {
        "hits": int(stats_obj.hits), "misses": int(stats_obj.misses),
        "targets": int(stats_obj.targets),
        "unjudged": int(stats_obj.targets)-int(stats_obj.hits)-int(stats_obj.misses),
        "too_early": int(stats_obj.too_early_presses),
        "keydowns": int(env.physical_keydowns),
        "overload": bool(stats_obj.overloaded),
    }
    # Never compare quality on a partial/invalid episode to a full-safe one.
    safe = safe_physics(stats, targets=len(named.segment.targets))
    events = [
        {
            "time_s": float(event.time_s),
            "target_index": event.target_index,
            "target_time_s": event.target_time_s,
            "key": event.key,
        }
        for event in trace.physical_keydowns
    ]
    classified = quality.classify_physical_keydowns(events, env, stats)
    hits = classified["hit_events"]
    objective = classified["quality"]["mae_ms"] if safe else None
    return PhysicalReplay(
        env=env, checkpoints=checkpoints, stats=stats,
        hit_events=hits, timing=classified["quality"],
        termination={
            "time_s": float(trace.termination.time_s),
            "reason": trace.termination.reason,
            "hits": trace.termination.hits,
            "keydowns": trace.termination.keydowns,
        },
        safe=safe, objective_ms=objective,
    )


def mutate_schedule(
    delays: tuple[int, ...], runs: tuple[PressRun, ...], rng: Random,
) -> tuple[int, ...] | None:
    """Occasionally change a consecutive 2-3-pulse block to cross couplings."""
    adjustable = [i for i, run in enumerate(runs) if run.max_delay > 0]
    if not adjustable:
        return None
    center = rng.choice(adjustable)
    size = rng.choice((1, 1, 2, 3))
    chosen = range(center, min(len(runs), center + min(size, NEIGHBOR_BLOCK_MAX)))
    proposal = list(delays)
    for i in chosen:
        run = runs[i]
        amount = rng.choice((-2, -1, 1, 2))
        proposed = max(0, min(run.max_delay, proposal[i] + amount))
        if proposed == proposal[i]:
            proposed = (proposal[i] + 1) % (run.max_delay + 1)
        proposal[i] = proposed
    proposed = tuple(proposal)
    return proposed if proposed != delays else None


def accept_sa(
    current_energy_ms: float, candidate_energy_ms: float,
    temperature_ms: float, rng: Random,
) -> bool:
    if not all(map(isfinite, (current_energy_ms, candidate_energy_ms, temperature_ms))):
        raise ValueError("SA energies and temperature must be finite")
    if temperature_ms <= 0.:
        raise ValueError("SA temperature must be positive")
    delta = candidate_energy_ms - current_energy_ms
    return delta <= 0. or rng.random() < exp(-delta / temperature_ms)


def _temperature(step: int, budget: int) -> float:
    if budget < 1 or not 0 <= step < budget:
        raise ValueError("Invalid SA step")
    fraction = step / max(1, budget - 1)
    return INITIAL_TEMPERATURE_MS * (
        FINAL_TEMPERATURE_MS / INITIAL_TEMPERATURE_MS) ** fraction


def anneal_chart(
    named,
    baseline_commands: tuple[tuple[float, ...], ...],
    *, seed: int, proposals: int = PROPOSALS_PER_ANCHOR,
):
    if proposals < 0:
        raise ValueError("Negative SA proposal budget")
    runs = press_runs(baseline_commands)
    zero = tuple(0 for _ in runs)
    initial = replay_commands(named, baseline_commands)
    if not initial.safe:
        raise RuntimeError("Frozen v2.4 open-loop baseline is not all-Hit SAFE")

    current_delays = best_delays = zero
    current = best = initial
    rng = Random(seed)
    counts = {
        "proposals": 0, "physically_safe": 0,
        "rejected_unsafe": 0, "accepted_uphill": 0,
        "accepted_not_uphill": 0, "rejected_sa": 0,
        "new_bests": 0, "no_op": 0,
        "unsafe_miss_candidates": 0, "unsafe_early_candidates": 0,
        "unsafe_overload_candidates": 0, "unsafe_extra_down_candidates": 0,
    }
    for step in range(proposals):
        proposal = mutate_schedule(current_delays, runs, rng)
        if proposal is None:
            counts["no_op"] += 1
            continue
        first = earliest_difference(runs, current_delays, proposal)
        trial_actions = realize_schedule(baseline_commands, runs, proposal)
        trial = replay_commands(
            named, trial_actions, parent=current, first_changed_frame=first)
        counts["proposals"] += 1
        if not trial.safe:
            counts["rejected_unsafe"] += 1
            counts["unsafe_miss_candidates"] += int(trial.stats["misses"] > 0)
            counts["unsafe_early_candidates"] += int(trial.stats["too_early"] > 0)
            counts["unsafe_overload_candidates"] += int(trial.stats["overload"])
            counts["unsafe_extra_down_candidates"] += int(
                trial.stats["keydowns"] > trial.stats["targets"])
            continue
        counts["physically_safe"] += 1
        previous = current.objective_ms
        assert previous is not None and trial.objective_ms is not None
        temperature = _temperature(step, proposals)
        if not accept_sa(previous, trial.objective_ms, temperature, rng):
            counts["rejected_sa"] += 1
            continue
        if trial.objective_ms > previous + EPS:
            counts["accepted_uphill"] += 1
        else:
            counts["accepted_not_uphill"] += 1
        current_delays, current = proposal, trial
        if trial.objective_ms < best.objective_ms - EPS:
            best_delays, best = proposal, trial
            counts["new_bests"] += 1
    # Independently verify the *best* complete 10ms action tape from frame 0.
    best_commands = realize_schedule(baseline_commands, runs, best_delays)
    confirmed = replay_commands(named, best_commands)
    if (not confirmed.safe
            or confirmed.stats != best.stats
            or confirmed.hit_events != best.hit_events
            or abs(confirmed.objective_ms - best.objective_ms) > EPS):
        raise RuntimeError("SA prefix checkpoint diverged from independent full replay")
    return {
        "baseline": initial, "best": confirmed,
        "best_commands": best_commands,
        "runs": runs, "best_delays": best_delays,
        "counts": counts, "seed": seed,
        "proposals_budget": proposals,
    }
