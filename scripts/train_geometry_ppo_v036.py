from __future__ import annotations

import random
from pathlib import Path

try:
    from . import train_geometry_ppo_v035 as v035
except ImportError:  # direct script execution
    import train_geometry_ppo_v035 as v035


v034 = v035.v034
base = v035.base

# v0.3.6: a single screen must not instantly move almost all rollouts from one
# BPM edge to the other. Keep a short memory of per-BPM gate deficit and retain
# recently-weak BPMs until they have been clean for several consecutive screens.
FOCUS_EMA_ALPHA = 0.50
FOCUS_CLEAR_SCREENS = 3
FOCUS_EPS = 1e-12

_ORIGINAL_EVALUATE = base._evaluate
_ORIGINAL_V035_SAVE_CHECKPOINT = v035.save_checkpoint

_PHASE_KEY: tuple[object, ...] | None = None
_FOCUS_EMA: dict[float, float] = {}
_FOCUS_CLEAR_STREAK: dict[float, int] = {}
_ACTIVE_FOCUS: set[float] = set()
_FOCUS_WEIGHTS: dict[float, float] = {}
_SEEN_PROBES: set[int] = set()


def _reset_focus_state() -> None:
    global _PHASE_KEY
    _PHASE_KEY = None
    _FOCUS_EMA.clear()
    _FOCUS_CLEAR_STREAK.clear()
    _ACTIVE_FOCUS.clear()
    _FOCUS_WEIGHTS.clear()
    _SEEN_PROBES.clear()


def _phase_identity(phase: base.core.CurriculumPhase) -> tuple[object, ...]:
    return (
        phase.name,
        phase.notes,
        phase.bpm_min,
        phase.bpm_max,
        phase.train_phase_jitter_ms,
        phase.eval_phase_jitter_ms,
        phase.vision.latency_s,
        phase.vision.latency_jitter_s,
        phase.vision.sample_period_s,
        phase.vision.position_noise_std,
        phase.vision.dropout_probability,
    )


def _ensure_phase(phase: base.core.CurriculumPhase) -> None:
    global _PHASE_KEY
    key = _phase_identity(phase)
    if key == _PHASE_KEY:
        return
    _reset_focus_state()
    _PHASE_KEY = key


def _evaluate(
    model: base.core.RecurrentActorCritic,
    device: base.core.torch.device,
    *,
    phase: base.core.CurriculumPhase,
    previous_notes: tuple[int, ...],
    args,
    episodes: int,
    retention_episodes: int,
    seed_base: int,
    label: str,
):
    # P6/P7/P8 share the same BPM range, so phase changes cannot be inferred
    # from BPM slices alone. Reset history from the actual curriculum phase.
    _ensure_phase(phase)
    return _ORIGINAL_EVALUATE(
        model,
        device,
        phase=phase,
        previous_notes=previous_notes,
        args=args,
        episodes=episodes,
        retention_episodes=retention_episodes,
        seed_base=seed_base,
        label=label,
    )


def weak_bpms(probe: base.core.Probe, count: int | None = None) -> tuple[float, ...]:
    """Select weak BPMs using EMA deficit plus a short clear hysteresis.

    A BPM that was focused remains eligible until it has shown zero gate deficit
    for ``FOCUS_CLEAR_SCREENS`` consecutive *new* probes. This prevents the
    145/240-style see-saw where one repaired edge immediately loses training
    weight and regresses while the other edge is being repaired.
    """

    global _ACTIVE_FOCUS
    if not probe.bpm_slices:
        return ()
    count = base.WORST_BPM_COUNT if count is None else count
    count = max(1, count)

    raw = {item.bpm: base._slice_deficit(item) for item in probe.bpm_slices}
    probe_id = id(probe)
    if probe_id not in _SEEN_PROBES:
        _SEEN_PROBES.add(probe_id)
        for bpm, deficit in raw.items():
            previous = _FOCUS_EMA.get(bpm, deficit)
            _FOCUS_EMA[bpm] = (
                deficit
                if bpm not in _FOCUS_EMA
                else FOCUS_EMA_ALPHA * deficit + (1.0 - FOCUS_EMA_ALPHA) * previous
            )
            if deficit > FOCUS_EPS:
                _FOCUS_CLEAR_STREAK[bpm] = 0
            else:
                _FOCUS_CLEAR_STREAK[bpm] = _FOCUS_CLEAR_STREAK.get(bpm, 0) + 1

    candidates = [
        bpm
        for bpm, deficit in raw.items()
        if deficit > FOCUS_EPS
        or (
            bpm in _ACTIVE_FOCUS
            and _FOCUS_CLEAR_STREAK.get(bpm, 0) < FOCUS_CLEAR_SCREENS
        )
    ]
    candidates.sort(
        key=lambda bpm: (
            _FOCUS_EMA.get(bpm, 0.0),
            raw.get(bpm, 0.0),
        ),
        reverse=True,
    )
    selected = tuple(candidates[:count])
    _ACTIVE_FOCUS = set(selected)

    _FOCUS_WEIGHTS.clear()
    for bpm in selected:
        _FOCUS_WEIGHTS[bpm] = max(
            FOCUS_EPS,
            _FOCUS_EMA.get(bpm, 0.0),
            raw.get(bpm, 0.0),
        )
    return selected


def focused_training_bpm_schedule(
    phase: base.core.CurriculumPhase,
    *,
    episodes: int,
    points: int,
    rng: random.Random,
    focus_bpms: tuple[float, ...] = (),
    focus_fraction: float | None = None,
) -> list[float]:
    """Keep anchor coverage while balancing persistent multi-edge focus.

    With the default 16 rollouts / 5 anchors and two focused edges this yields
    5 + 5 edge rollouts and 2 + 2 + 2 interior rollouts. A sole weak BPM still
    receives 12/16 rollouts, matching the v0.3.4 75% single-focus behaviour.
    """

    if episodes <= 0:
        return []
    anchors = list(base.core.bpm_points(phase.bpm_min, phase.bpm_max, points))
    if not focus_bpms:
        return base.core.training_bpm_schedule(
            phase, episodes=episodes, points=points, rng=rng
        )

    schedule = anchors[:episodes]
    remaining = episodes - len(schedule)
    if remaining <= 0:
        rng.shuffle(schedule)
        return schedule

    fraction = (
        focus_fraction
        if focus_fraction is not None
        else (
            v034.SINGLE_WEAK_BPM_FOCUS
            if len(focus_bpms) == 1
            else v034.MULTI_WEAK_BPM_FOCUS
        )
    )
    target_focus = max(1, min(episodes, int(round(episodes * fraction))))
    already_focus = sum(1 for bpm in schedule if bpm in focus_bpms)
    extra_focus = min(remaining, max(0, target_focus - already_focus))

    # Balance two hysteretically-active edges instead of letting the newest
    # larger deficit monopolize the batch. Order stronger EMA first only to
    # decide who receives an odd extra rollout.
    ordered_focus = sorted(
        focus_bpms,
        key=lambda bpm: _FOCUS_WEIGHTS.get(bpm, 0.0),
        reverse=True,
    )
    for i in range(extra_focus):
        schedule.append(ordered_focus[i % len(ordered_focus)])
    remaining -= extra_focus

    if remaining:
        nonfocus_anchors = [
            bpm
            for bpm in anchors
            if not any(abs(bpm - focused) < 1e-12 for focused in focus_bpms)
        ]
        fill = nonfocus_anchors or anchors
        schedule.extend(fill[i % len(fill)] for i in range(remaining))

    rng.shuffle(schedule)
    return schedule


def _print_header(args, phases: tuple[base.core.CurriculumPhase, ...], mode: str) -> None:
    base._write("=== DMDOD / Planet Geometry PPO v0.3.6 ===")
    base._write(
        f"mode={mode} device={args.device} seed={args.seed} checkpoint={args.checkpoint}"
    )
    base._write(
        f"task={args.notes} BPM={args.bpm_min:g}..{args.bpm_max:g} "
        f"vision={args.vision_hz:g}Hz/{args.vision_latency_ms:g}ms | "
        f"gate X={base.PRECISION_OVERALL_XACC:g}%/min{base.PRECISION_MIN_BPM_XACC:g}% "
        f"PP={base.PRECISION_OVERALL_PP:.0%}/min{base.PRECISION_MIN_BPM_PP:.0%}"
    )
    base._write(
        f"focus EMA={FOCUS_EMA_ALPHA:.2f} hold={FOCUS_CLEAR_SCREENS} screens; "
        "persistent multi-edge balance"
    )
    if v035.VERBOSE_OUTPUT:
        base._write(
            f"metric=DLL HitMargin denominator; TooEarly reward penalty={v035.TOO_EARLY_PENALTY:g}"
        )
        base._write(
            f"screen={base.QUICK_EVAL_EPISODES}ep verify={args.eval_episodes}ep; "
            f"single-focus={v034.SINGLE_WEAK_BPM_FOCUS:.0%} "
            f"multi-focus={v034.MULTI_WEAK_BPM_FOCUS:.0%}; phases={len(phases)}"
        )
        base._write("visible=motor+planet geometry | hidden=time/BPM/target-angle/error/direction")
    else:
        base._write("compact output; use --verbose for per-BPM details")
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
    _ORIGINAL_V035_SAVE_CHECKPOINT(
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
    saved["format_version"] = 15
    saved["trainer_ui_version"] = "0.3.6-hysteretic-focus"
    saved["adaptive_focus_v036"] = {
        "ema_alpha": FOCUS_EMA_ALPHA,
        "clear_screens": FOCUS_CLEAR_SCREENS,
        "balanced_multi_focus": True,
        "phase_scoped_history": True,
    }
    base.core.torch.save(saved, path)


def main() -> None:
    # v0.3.5 owns DLL-margin metrics/reward and compact output. Override only
    # phase-scoped evaluation/focus scheduling and checkpoint metadata.
    base._evaluate = _evaluate
    v034.weak_bpms = weak_bpms
    v034.focused_training_bpm_schedule = focused_training_bpm_schedule
    v035._print_header = _print_header
    v035.save_checkpoint = save_checkpoint
    v035.main()


if __name__ == "__main__":
    main()
