from __future__ import annotations

import random
import sys
from pathlib import Path

try:
    from . import train_geometry_ppo as base
except ImportError:  # direct script execution
    import train_geometry_ppo as base


# v0.3.4: cheap probes may request a full verification before they satisfy the
# strict gate, and rollout focus follows the actual per-BPM gate deficit.
NEAR_VERIFY_MIN_BPM_XACC = 85.0
NEAR_VERIFY_OVERALL_XACC = 93.0
NEAR_VERIFY_MIN_BPM_PP = 0.50
NEAR_VERIFY_OVERALL_PP = 0.75
SINGLE_WEAK_BPM_FOCUS = 0.75
MULTI_WEAK_BPM_FOCUS = 0.60

_BASE_PASSES = base.passes
_BASE_SAVE_CHECKPOINT = base.save_checkpoint
_BASE_PRINT_GATE = base._print_gate

_FOCUS_WEIGHTS: dict[float, float] = {}


def near_precision_passes(probe: base.core.Probe) -> bool:
    return (
        base._min_xacc(probe) >= NEAR_VERIFY_MIN_BPM_XACC
        and base._xacc(probe) >= NEAR_VERIFY_OVERALL_XACC
        and base._min_pp(probe) >= NEAR_VERIFY_MIN_BPM_PP
        and base._pp(probe) >= NEAR_VERIFY_OVERALL_PP
    )


def passes(probe: base.core.Probe, retention: base.core.Retention, args, notes: int) -> bool:
    """Use relaxed precision only to trigger an expensive full verification."""

    if probe.episodes <= base.QUICK_EVAL_EPISODES:
        return base.completion_passes(probe, retention, args, notes) and near_precision_passes(probe)
    return _BASE_PASSES(probe, retention, args, notes)


def weak_bpms(probe: base.core.Probe, count: int | None = None) -> tuple[float, ...]:
    """Return only BPMs with a real gate deficit, strongest deficit first."""

    global _FOCUS_WEIGHTS
    count = base.WORST_BPM_COUNT if count is None else count
    deficits = [
        (item.bpm, base._slice_deficit(item))
        for item in probe.bpm_slices
        if base._slice_deficit(item) > 1e-12
    ]
    deficits.sort(key=lambda pair: pair[1], reverse=True)
    selected = deficits[: max(1, count)]
    _FOCUS_WEIGHTS = {bpm: deficit for bpm, deficit in selected}
    return tuple(bpm for bpm, _ in selected)


def focused_training_bpm_schedule(
    phase: base.core.CurriculumPhase,
    *,
    episodes: int,
    points: int,
    rng: random.Random,
    focus_bpms: tuple[float, ...] = (),
    focus_fraction: float | None = None,
) -> list[float]:
    """Cover every anchor once, then spend most extra rollouts on deficient BPMs."""

    if episodes <= 0:
        return []
    anchors = list(base.core.bpm_points(phase.bpm_min, phase.bpm_max, points))
    if not focus_bpms:
        return base.core.training_bpm_schedule(
            phase, episodes=episodes, points=points, rng=rng
        )

    # Preserve broad BPM coverage: every anchor gets at least one rollout when
    # the rollout budget permits it.
    schedule = anchors[:episodes]
    remaining = episodes - len(schedule)
    if remaining <= 0:
        rng.shuffle(schedule)
        return schedule

    fraction = (
        focus_fraction
        if focus_fraction is not None
        else (SINGLE_WEAK_BPM_FOCUS if len(focus_bpms) == 1 else MULTI_WEAK_BPM_FOCUS)
    )
    target_focus = max(1, min(episodes, int(round(episodes * fraction))))
    already_focus = sum(1 for bpm in schedule if bpm in focus_bpms)
    extra_focus = min(remaining, max(0, target_focus - already_focus))

    weights = [max(1e-9, _FOCUS_WEIGHTS.get(bpm, 1.0)) for bpm in focus_bpms]
    if extra_focus:
        schedule.extend(rng.choices(list(focus_bpms), weights=weights, k=extra_focus))
        remaining -= extra_focus

    if remaining:
        schedule.extend(
            base.core.training_bpm_schedule(
                phase, episodes=remaining, points=points, rng=rng
            )
        )
    rng.shuffle(schedule)
    return schedule


def _print_gate(probe: base.core.Probe, retention: base.core.Retention, args, notes: int) -> None:
    _BASE_PRINT_GATE(probe, retention, args, notes)
    if (
        probe.episodes <= base.QUICK_EVAL_EPISODES
        and base.completion_passes(probe, retention, args, notes)
        and near_precision_passes(probe)
        and not base.precision_passes(probe, args, notes)
    ):
        base._write("  verify-candidate=YES (near accuracy gate)")


def _print_header(args, phases: tuple[base.core.CurriculumPhase, ...], mode: str) -> None:
    base._write("=== DMDOD / Planet Geometry PPO v0.3.4 ===")
    base._write(
        f"mode={mode}  device={args.device}  seed={args.seed}  checkpoint={args.checkpoint}"
    )
    base._write(
        f"task={args.notes} notes  BPM={args.bpm_min:g}..{args.bpm_max:g}  "
        f"control={args.control_dt*1000:.1f}ms  vision={args.vision_hz:g}Hz/{args.vision_latency_ms:g}ms"
    )
    base._write(
        f"accuracy gate: minX>={base.PRECISION_MIN_BPM_XACC:g}% X>={base.PRECISION_OVERALL_XACC:g}% "
        f"minPP>={base.PRECISION_MIN_BPM_PP:.0%} PP>={base.PRECISION_OVERALL_PP:.0%}"
    )
    base._write(
        f"near-verify: minX>={NEAR_VERIFY_MIN_BPM_XACC:g}% X>={NEAR_VERIFY_OVERALL_XACC:g}% "
        f"minPP>={NEAR_VERIFY_MIN_BPM_PP:.0%} PP>={NEAR_VERIFY_OVERALL_PP:.0%}"
    )
    base._write(
        f"screen={base.QUICK_EVAL_EPISODES}ep; verify={args.eval_episodes}ep; "
        f"single-focus={SINGLE_WEAK_BPM_FOCUS:.0%} multi-focus={MULTI_WEAK_BPM_FOCUS:.0%}; "
        f"phases={len(phases)}"
    )
    base._write("visible=motor+planet geometry | hidden=time/BPM/target-angle/error/direction")
    base._write()


def save_checkpoint(
    path: Path,
    model: base.core.RecurrentActorCritic,
    optimizer: base.core.torch.optim.Optimizer,
    *,
    args,
    phase_index: int,
    phase: base.core.CurriculumPhase,
    global_update: int,
    probe: base.core.Probe,
) -> None:
    _BASE_SAVE_CHECKPOINT(
        path,
        model,
        optimizer,
        args=args,
        phase_index=phase_index,
        phase=phase,
        global_update=global_update,
        probe=probe,
    )
    saved = base.core.torch.load(path, map_location="cpu")
    saved["format_version"] = 13
    saved["trainer_ui_version"] = "0.3.4-deficit-focus"
    saved["adaptive_focus_v034"] = {
        "near_verify_min_bpm_xacc": NEAR_VERIFY_MIN_BPM_XACC,
        "near_verify_overall_xacc": NEAR_VERIFY_OVERALL_XACC,
        "near_verify_min_bpm_perfect_rate": NEAR_VERIFY_MIN_BPM_PP,
        "near_verify_overall_perfect_rate": NEAR_VERIFY_OVERALL_PP,
        "single_weak_bpm_focus": SINGLE_WEAK_BPM_FOCUS,
        "multi_weak_bpm_focus": MULTI_WEAK_BPM_FOCUS,
        "minimum_anchor_coverage": True,
    }
    base.core.torch.save(saved, path)


def _pop_float_arg(name: str, default: float) -> float:
    prefix = name + "="
    for i, value in enumerate(tuple(sys.argv)):
        if value.startswith(prefix):
            sys.argv.remove(value)
            return float(value[len(prefix) :])
        if value == name:
            if i + 1 >= len(sys.argv):
                raise SystemExit(f"{name} requires a value")
            result = float(sys.argv[i + 1])
            del sys.argv[i : i + 2]
            return result
    return default


def _read_v034_args() -> None:
    global NEAR_VERIFY_MIN_BPM_XACC, NEAR_VERIFY_OVERALL_XACC
    global NEAR_VERIFY_MIN_BPM_PP, NEAR_VERIFY_OVERALL_PP
    global SINGLE_WEAK_BPM_FOCUS, MULTI_WEAK_BPM_FOCUS

    NEAR_VERIFY_MIN_BPM_XACC = _pop_float_arg(
        "--near-verify-min-bpm-xacc", NEAR_VERIFY_MIN_BPM_XACC
    )
    NEAR_VERIFY_OVERALL_XACC = _pop_float_arg(
        "--near-verify-overall-xacc", NEAR_VERIFY_OVERALL_XACC
    )
    NEAR_VERIFY_MIN_BPM_PP = _pop_float_arg(
        "--near-verify-min-bpm-pp", NEAR_VERIFY_MIN_BPM_PP
    )
    NEAR_VERIFY_OVERALL_PP = _pop_float_arg(
        "--near-verify-overall-pp", NEAR_VERIFY_OVERALL_PP
    )
    SINGLE_WEAK_BPM_FOCUS = _pop_float_arg(
        "--single-weak-bpm-focus", SINGLE_WEAK_BPM_FOCUS
    )
    MULTI_WEAK_BPM_FOCUS = _pop_float_arg(
        "--multi-weak-bpm-focus", MULTI_WEAK_BPM_FOCUS
    )

    for name, value in (
        ("--near-verify-min-bpm-xacc", NEAR_VERIFY_MIN_BPM_XACC / 100.0),
        ("--near-verify-overall-xacc", NEAR_VERIFY_OVERALL_XACC / 100.0),
        ("--near-verify-min-bpm-pp", NEAR_VERIFY_MIN_BPM_PP),
        ("--near-verify-overall-pp", NEAR_VERIFY_OVERALL_PP),
        ("--single-weak-bpm-focus", SINGLE_WEAK_BPM_FOCUS),
        ("--multi-weak-bpm-focus", MULTI_WEAK_BPM_FOCUS),
    ):
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{name} is outside its valid range")


def main() -> None:
    _read_v034_args()

    # Patch the v0.3.3 frontend without duplicating its training loop. Its
    # train_ui resolves these module globals at runtime.
    base.passes = passes
    base.weak_bpms = weak_bpms
    base.focused_training_bpm_schedule = focused_training_bpm_schedule
    base._print_gate = _print_gate
    base._print_header = _print_header
    base.save_checkpoint = save_checkpoint
    base.main()


if __name__ == "__main__":
    main()
