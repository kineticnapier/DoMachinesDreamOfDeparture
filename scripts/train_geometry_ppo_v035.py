from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

try:
    from . import train_geometry_ppo_v034 as v034
except ImportError:  # direct script execution
    import train_geometry_ppo_v034 as v034

from dmdod.geometry_rhythm_env import GeometryRhythmEnv
from dmdod.rhythm_env import RewardConfig, make_regular_targets


base = v034.base

# DLL-style accuracy counts HitMargins, not chart targets. v0.3.4 still
# aggregated probe PP/XAcc with target denominators, which can print impossible
# combinations such as X=60% / PP=100% after [TooEarly, Perfect].
TOO_EARLY_PENALTY = 1.0
_ORIGINAL_V034_SAVE_CHECKPOINT = v034.save_checkpoint


def _accuracy_terms(stats) -> tuple[float, int, int, int]:
    """Return XAcc points/denominator and Perfect count/denominator.

    New DLL-rule environments expose the exact HitMargin counters. The fallback
    keeps compatibility with checkpoints/tests produced before those fields
    existed.
    """

    xacc_denominator = int(getattr(stats, "x_accuracy_denominator", stats.targets))
    xacc_points = float(
        getattr(
            stats,
            "x_accuracy_points",
            stats.x_accuracy_percent * xacc_denominator / 100.0,
        )
    )
    perfect_denominator = int(getattr(stats, "hit_margin_count", stats.targets))
    perfects = int(stats.perfects)
    return xacc_points, xacc_denominator, perfects, perfect_denominator


def make_env(
    *,
    bpm: float,
    notes: int,
    start_s: float,
    control_dt: float,
    config,
    seed: int,
) -> GeometryRhythmEnv:
    """Training environment with a reward that actually discourages stray margins."""

    return GeometryRhythmEnv(
        make_regular_targets(bpm=bpm, count=notes, start_s=start_s, pattern="left"),
        bpm=bpm,
        same_hand=True,
        control_dt_s=control_dt,
        vision_config=config,
        perception_seed=seed,
        reward_config=RewardConfig(too_early_penalty=TOO_EARLY_PENALTY),
    )


def deterministic_probe(
    model: base.core.RecurrentActorCritic,
    device: base.core.torch.device,
    *,
    phase: base.core.CurriculumPhase,
    control_dt: float,
    episodes: int,
    eval_bpm_points: int,
    seed_base: int,
) -> base.AccuracyProbe:
    """Probe using HitMargin denominators for both XAcc and Perfect rate."""

    jitter_s = phase.eval_phase_jitter_ms / 1000.0
    offsets = (
        (0.0,)
        if episodes == 1
        else tuple(-jitter_s + 2.0 * jitter_s * i / (episodes - 1) for i in range(episodes))
    )
    bpms = base.core.bpm_points(phase.bpm_min, phase.bpm_max, eval_bpm_points)
    buckets: dict[float, dict[str, object]] = {
        bpm: {
            "episodes": 0,
            "hits": 0,
            "targets": 0,
            "full": 0,
            "clean": 0,
            "errors": [],
            "xacc_points": 0.0,
            "xacc_denominator": 0,
            "perfects": 0,
            "perfect_denominator": 0,
        }
        for bpm in bpms
    }

    hits = targets = full = clean = overloads = too_early = 0
    perfects = perfect_denominator = xacc_denominator = 0
    xacc_points = 0.0
    errors: list[float] = []
    was_training = model.training
    model.eval()

    iterator = base._bar(
        enumerate(offsets),
        total=len(offsets),
        desc="probe",
        unit="ep",
        leave=False,
        position=2,
    )
    for i, offset in iterator:
        bpm = bpms[i % len(bpms)]
        env = base.core.make_env(
            bpm=bpm,
            notes=phase.notes,
            start_s=max(0.050, base.core.curriculum_start_s(phase.notes) + offset),
            control_dt=control_dt,
            config=phase.vision,
            seed=seed_base + i * 1009,
        )
        observation = env.reset()
        state = model.initial_state(device)
        while True:
            action, state = model.deterministic_action(
                base.core.observation_tensor(observation, device), state
            )
            transition = env.step(action)
            observation = transition.observation
            if transition.done:
                break

        stats = env.stats
        is_full = stats.hits == stats.targets and not stats.overloaded
        is_clean = is_full and stats.too_early_presses == 0
        episode_errors = env.timing_errors_ms
        ep_xpoints, ep_xden, ep_perfects, ep_ppden = _accuracy_terms(stats)

        hits += stats.hits
        targets += stats.targets
        full += int(is_full)
        clean += int(is_clean)
        overloads += int(stats.overloaded)
        too_early += stats.too_early_presses
        xacc_points += ep_xpoints
        xacc_denominator += ep_xden
        perfects += ep_perfects
        perfect_denominator += ep_ppden
        errors.extend(episode_errors)

        bucket = buckets[bpm]
        bucket["episodes"] = int(bucket["episodes"]) + 1
        bucket["hits"] = int(bucket["hits"]) + stats.hits
        bucket["targets"] = int(bucket["targets"]) + stats.targets
        bucket["full"] = int(bucket["full"]) + int(is_full)
        bucket["clean"] = int(bucket["clean"]) + int(is_clean)
        bucket["xacc_points"] = float(bucket["xacc_points"]) + ep_xpoints
        bucket["xacc_denominator"] = int(bucket["xacc_denominator"]) + ep_xden
        bucket["perfects"] = int(bucket["perfects"]) + ep_perfects
        bucket["perfect_denominator"] = int(bucket["perfect_denominator"]) + ep_ppden
        bucket_errors = bucket["errors"]
        assert isinstance(bucket_errors, list)
        bucket_errors.extend(episode_errors)

    if was_training:
        model.train()

    slices = tuple(
        base.AccuracyBpmSlice(
            bpm=bpm,
            episodes=int(bucket["episodes"]),
            hits=int(bucket["hits"]),
            targets=int(bucket["targets"]),
            full=int(bucket["full"]),
            clean=int(bucket["clean"]),
            errors_ms=tuple(float(x) for x in bucket["errors"]),
            x_accuracy_percent=(
                100.0
                * float(bucket["xacc_points"])
                / max(1, int(bucket["xacc_denominator"]))
            ),
            perfect_rate=(
                int(bucket["perfects"])
                / max(1, int(bucket["perfect_denominator"]))
            ),
        )
        for bpm, bucket in buckets.items()
    )
    return base.AccuracyProbe(
        episodes=episodes,
        hits=hits,
        targets=targets,
        full=full,
        clean=clean,
        overloads=overloads,
        too_early=too_early,
        errors_ms=tuple(errors),
        bpm_slices=slices,
        x_accuracy_percent=100.0 * xacc_points / max(1, xacc_denominator),
        perfect_rate=perfects / max(1, perfect_denominator),
    )


def _print_header(args, phases: tuple[base.core.CurriculumPhase, ...], mode: str) -> None:
    base._write("=== DMDOD / Planet Geometry PPO v0.3.5 ===")
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
        f"metric=DLL HitMargin denominator; TooEarly reward penalty={TOO_EARLY_PENALTY:g}"
    )
    base._write(
        f"near-verify: minX>={v034.NEAR_VERIFY_MIN_BPM_XACC:g}% "
        f"X>={v034.NEAR_VERIFY_OVERALL_XACC:g}% "
        f"minPP>={v034.NEAR_VERIFY_MIN_BPM_PP:.0%} "
        f"PP>={v034.NEAR_VERIFY_OVERALL_PP:.0%}"
    )
    base._write(
        f"screen={base.QUICK_EVAL_EPISODES}ep; verify={args.eval_episodes}ep; "
        f"single-focus={v034.SINGLE_WEAK_BPM_FOCUS:.0%} "
        f"multi-focus={v034.MULTI_WEAK_BPM_FOCUS:.0%}; phases={len(phases)}"
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
    _ORIGINAL_V034_SAVE_CHECKPOINT(
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
    saved["format_version"] = 14
    saved["trainer_ui_version"] = "0.3.5-margin-aware"
    saved["margin_aware_accuracy"] = {
        "xacc_denominator": "HitMargin/dead-tile entries",
        "perfect_denominator": "HitMargin entries",
        "too_early_reward_penalty": TOO_EARLY_PENALTY,
    }
    base.core.torch.save(saved, path)


def main() -> None:
    # Patch the shared implementation before v0.3.4 installs its curriculum
    # hooks. This keeps checkpoint compatibility while fixing the metric/reward
    # mismatch that caused early phases to grind on stray TooEarly presses.
    base.core.make_env = make_env
    base.deterministic_probe = deterministic_probe
    v034._print_header = _print_header
    v034.save_checkpoint = save_checkpoint
    v034.main()


if __name__ == "__main__":
    main()
