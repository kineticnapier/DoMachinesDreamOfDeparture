from __future__ import annotations

"""Stage C v2.9: protected whole-chart phase alignment + full-pulse SA.

Global phase is a *decision variable*, not a hyperparameter. Every phase
candidate is physically evaluated on the entire chart. The most accurate
feasible plan seeds an unchanged v2.8 pulse-level annealer. This retains
the original verified v2.8 solution as an irrevocable safe fallback.
"""

from dataclasses import dataclass

from . import prestige1 as p
from . import prestige1_global_annealing as physical
from . import prestige1_pulse_annealing as local

# One fixed exploratory protocol; this does not sweep temperature/budget.
MAX_GLOBAL_PHASE_TICKS = 8
LOCAL_PROPOSALS = local.PROPOSALS_PER_ANCHOR
EPS = 1e-9


def global_phase_tape(
    tape: tuple[tuple[float, ...], ...], ticks: int
) -> tuple[tuple[float, ...], ...]:
    """Translate the complete original 8-Key 10ms waveform to the right.

    Insert neutral controls at time zero and keep every existing +1/-1/0
    command, including press duration, release and off-target braking.
    Appending all source commands also preserves the original ending.
    """
    if not tape or any(
        len(frame) != p.KEY_COUNT or
        any(v not in (-1., 0., 1.) for v in frame)
        for frame in tape
    ):
        raise ValueError("Expected nonempty 8-Key ternary action tape")
    if type(ticks) is not int or not 0 <= ticks <= MAX_GLOBAL_PHASE_TICKS:
        raise ValueError("Global timing phase out of allowed interval")
    neutral = (0.,) * p.KEY_COUNT
    return (neutral,) * ticks + tape


@dataclass
class PhaseSelection:
    original: physical.PhysicalReplay
    phase_best: physical.PhysicalReplay
    best_tape: tuple[tuple[float, ...], ...]
    ticks: int
    candidates: int
    safe_candidates: int
    unsafe_miss_candidates: int
    unsafe_overload_candidates: int
    unsafe_early_candidates: int


def select_global_phase(named, original_tape, *, max_ticks=MAX_GLOBAL_PHASE_TICKS):
    """Discrete exhaustive *one-variable* optimization, full safety gated."""
    if type(max_ticks) is not int or not 0 <= max_ticks <= MAX_GLOBAL_PHASE_TICKS:
        raise ValueError("Invalid phase search bound")
    original = physical.replay_commands(named, original_tape)
    if not original.safe or original.objective_ms is None:
        raise RuntimeError("v2.8 phase starting tape not physically SAFE")
    best = original
    best_tape = original_tape
    best_ticks = 0
    safe = 1
    unsafe_miss = unsafe_early = unsafe_overload = 0
    for ticks in range(1, max_ticks + 1):
        shifted = global_phase_tape(original_tape, ticks)
        candidate = physical.replay_commands(named, shifted)
        if not candidate.safe:
            unsafe_miss += int(candidate.stats["misses"] > 0)
            unsafe_early += int(candidate.stats["too_early"] > 0)
            unsafe_overload += int(candidate.stats["overload"])
            continue
        safe += 1
        if candidate.objective_ms is None:
            raise RuntimeError("SAFE full-chart replay lacks timing objective")
        if candidate.objective_ms < best.objective_ms - EPS:
            best = candidate
            best_tape = shifted
            best_ticks = ticks
    return PhaseSelection(
        original=original, phase_best=best, best_tape=best_tape,
        ticks=best_ticks, candidates=max_ticks + 1,
        safe_candidates=safe, unsafe_miss_candidates=unsafe_miss,
        unsafe_early_candidates=unsafe_early,
        unsafe_overload_candidates=unsafe_overload,
    )


def optimize_phase_and_pulses(
    named, original_tape, *, seed: int, max_ticks=MAX_GLOBAL_PHASE_TICKS,
    local_proposals=LOCAL_PROPOSALS,
):
    phase = select_global_phase(named, original_tape, max_ticks=max_ticks)
    # The local optimizer independently replays and validates its starting tape.
    detail = local.search_chart(
        named, phase.best_tape, seed=seed, proposals=local_proposals)
    if (detail["baseline"].stats != phase.phase_best.stats
            or detail["baseline"].hit_events != phase.phase_best.hit_events
            or detail["baseline"].timing != phase.phase_best.timing):
        raise RuntimeError("Local SA start diverged from best SAFE global phase")
    best = detail["best"]
    if (not best.safe or best.objective_ms is None
            or phase.original.objective_ms is None
            or best.objective_ms > phase.original.objective_ms + EPS):
        raise RuntimeError("Phase + SA must preserve original SAFE trajectory")
    # Local SA performs its own independent-from-zero full replay. This
    # additionally confirms the complete final tape against the initial
    # physically verified v2.8 plan, regardless of cache snapshots.
    return {
        "initial": phase.original, "phase_best": phase.phase_best,
        "best": best, "best_commands": detail["best_commands"],
        "phase_ticks": phase.ticks,
        "phase_candidates": phase.candidates,
        "phase_safe_candidates": phase.safe_candidates,
        "phase_unsafe_miss_candidates": phase.unsafe_miss_candidates,
        "phase_unsafe_early_candidates": phase.unsafe_early_candidates,
        "phase_unsafe_overload_candidates": phase.unsafe_overload_candidates,
        "pulse_annealing": detail, "seed": seed,
    }
