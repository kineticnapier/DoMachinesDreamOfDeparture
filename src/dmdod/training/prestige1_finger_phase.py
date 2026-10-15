from __future__ import annotations

"""Stage C v3.0: whole-finger phase alignment with true 1ms physics.

Search one *per-finger phase vector* by conservative coordinate descent.
Translate each finger's complete +/-1/0 waveform, preserving its shape
and press/release durations. Only fully SAFE 850/850 candidates are kept.
Then reuse the unchanged v2.8 coupled full-pulse SA/repair on the feasible
phase-aligned starting tape. The v2.9 source tape is never modified.
"""

from dataclasses import dataclass

from . import prestige1 as p
from . import prestige1_global_annealing as physical
from . import prestige1_pulse_annealing as local

FINGER_PHASE_LIMIT = 2                 # +/- 20ms at 10ms control
PHASE_CHOICES = (-2, -1, 1, 2)        # single, fixed protocol
LOCAL_PROPOSALS = local.PROPOSALS_PER_ANCHOR
EPS = 1e-9


def translate_finger_waveforms(
    baseline: tuple[tuple[float, ...], ...],
    offsets: tuple[int, ...],
) -> tuple[tuple[float, ...], ...]:
    """Rebuild independent key waveforms from an immutable source tape.

    All eight fingers are translated relative to the SAME original tape,
    avoiding compounded/intermediate resampling. Entire waveform columns
    (including independent release/brake commands) are copied intact.
    Early non-neutral commands can never be truncated. Positive shifts
    always fit into an additional 2 neutral trailing frames. Source
    command times are integral 10ms frames.
    """
    if not baseline or any(
        len(frame) != p.KEY_COUNT
        or any(v not in (-1., 0., 1.) for v in frame)
        for frame in baseline
    ):
        raise ValueError("Expected nonempty 8-Key ternary baseline tape")
    if len(offsets) != p.KEY_COUNT or any(
        type(value) is not int or abs(value) > FINGER_PHASE_LIMIT
        for value in offsets
    ):
        raise ValueError("Invalid per-finger phase offset vector")
    if all(value == 0 for value in offsets):
        return baseline
    n = len(baseline)
    out = [[0.] * p.KEY_COUNT for _ in range(n + FINGER_PHASE_LIMIT)]
    for key, shift in enumerate(offsets):
        for i, original in enumerate(baseline):
            command = original[key]
            if command == 0.:
                continue
            new_i = i + shift
            if not 0 <= new_i < len(out):
                raise ValueError("Nonzero finger command would be truncated")
            out[new_i][key] = command
    return tuple(tuple(row) for row in out)


@dataclass
class FingerPhaseSelection:
    original: physical.PhysicalReplay
    best: physical.PhysicalReplay
    commands: tuple[tuple[float, ...], ...]
    offsets: tuple[int, ...]
    checked: int
    physically_safe: int
    unsafe_miss: int
    unsafe_too_early: int
    unsafe_overload: int
    invalid_boundary: int
    improved_keys: int
    active_fingers: int


def select_finger_phases(named, source_tape):
    """One fixed full-safety-gated coordinate sweep (no HPO)."""
    source = tuple(tuple(frame) for frame in source_tape)
    initial = physical.replay_commands(named, source)
    if not initial.safe or initial.objective_ms is None:
        raise RuntimeError("Frozen v2.9 original action tape is not physically SAFE")

    nkeys = p.KEY_COUNT
    offsets = (0,) * nkeys
    current = initial
    current_tape = source
    counters = dict(checked=0, physically_safe=0, unsafe_miss=0,
                    unsafe_too_early=0, unsafe_overload=0,
                    invalid_boundary=0, improved_keys=0, active_fingers=0)

    # Descending number of source positive frames prioritizes keys carrying
    # most chart notes; ties resolve deterministically by finger index.
    key_order = sorted(
        range(nkeys),
        key=lambda key: (
            -sum(frame[key] == 1. for frame in source), key),
    )
    for key in key_order:
        if not any(frame[key] != 0. for frame in source):
            continue
        counters["active_fingers"] += 1
        best_for_key = current
        tape_for_key = current_tape
        offset_for_key = offsets[key]
        for trial_offset in PHASE_CHOICES:
            proposal = list(offsets)
            proposal[key] = trial_offset
            try:
                tape = translate_finger_waveforms(source, tuple(proposal))
            except ValueError as error:
                if str(error) != "Nonzero finger command would be truncated":
                    raise
                counters["invalid_boundary"] += 1
                continue
            candidate = physical.replay_commands(named, tape)
            counters["checked"] += 1
            if not candidate.safe:
                counters["unsafe_miss"] += int(candidate.stats["misses"] > 0)
                counters["unsafe_too_early"] += int(
                    candidate.stats["too_early"] > 0)
                counters["unsafe_overload"] += int(
                    candidate.stats["overload"])
                continue
            counters["physically_safe"] += 1
            if candidate.objective_ms is None:
                raise RuntimeError("SAFE replay must have a valid hit MAE")
            if candidate.objective_ms < best_for_key.objective_ms - EPS:
                best_for_key = candidate
                tape_for_key = tape
                offset_for_key = trial_offset
        if offset_for_key != offsets[key]:
            counters["improved_keys"] += 1
            offsets = offsets[:key] + (offset_for_key,) + offsets[key+1:]
            current = best_for_key
            current_tape = tape_for_key
    if current.objective_ms > initial.objective_ms + EPS:
        raise RuntimeError("Motor phase coordinate selection regressed")
    replay = physical.replay_commands(named, current_tape)
    if (not replay.safe or replay.stats != current.stats
            or replay.hit_events != current.hit_events
            or replay.timing != current.timing
            or replay.objective_ms != current.objective_ms):
        raise RuntimeError("Motor phase cached result diverged from exact full replay")
    return FingerPhaseSelection(
        initial, replay, current_tape, offsets, **counters)


def optimize_finger_phases_and_pulses(
    named, source_tape, *, seed: int, local_proposals=LOCAL_PROPOSALS,
):
    phase = select_finger_phases(named, source_tape)
    local_result = local.search_chart(
        named, phase.commands, seed=seed, proposals=local_proposals)
    local_start, best = local_result["baseline"], local_result["best"]
    if (local_start.stats != phase.best.stats
            or local_start.hit_events != phase.best.hit_events
            or local_start.timing != phase.best.timing
            or local_start.objective_ms != phase.best.objective_ms):
        raise RuntimeError("v3.0 pulse annealer start diverged from phase solution")
    if (not best.safe or best.objective_ms is None
            or best.objective_ms > phase.original.objective_ms + EPS
            or best.stats != phase.original.stats):
        raise RuntimeError("v3.0 final plan worse or less safe than v2.9")
    return {
        "initial": phase.original,
        "phase_best": phase.best,
        "best": best,
        "best_commands": local_result["best_commands"],
        "finger_phase": phase,
        "pulse_annealing": local_result,
        "seed": seed,
    }
