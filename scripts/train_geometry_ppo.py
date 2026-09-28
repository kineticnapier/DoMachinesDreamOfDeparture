from __future__ import annotations

import copy
import random
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from tqdm.auto import tqdm
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "tqdm is required. Run: uv sync --extra dev --extra rl --inexact"
    ) from exc

import train_geometry_ppo_core as core


PROGRESS_ENABLED = True


@dataclass(frozen=True)
class AccuracyBpmSlice(core.BpmSlice):
    x_accuracy_percent: float = 0.0
    perfect_rate: float = 0.0


@dataclass(frozen=True)
class AccuracyProbe(core.Probe):
    x_accuracy_percent: float = 0.0
    perfect_rate: float = 0.0


def _bar(iterable=None, **kwargs):
    kwargs.setdefault("disable", not PROGRESS_ENABLED)
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("ascii", False)
    return tqdm(iterable, **kwargs)


def _write(message: str = "") -> None:
    if PROGRESS_ENABLED:
        tqdm.write(message)
    else:
        print(message)


def _fmt_ms(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "--"
    return f"{value:+.1f}" if signed else f"{value:.1f}"


def deterministic_probe(
    model: core.RecurrentActorCritic,
    device: core.torch.device,
    *,
    phase: core.CurriculumPhase,
    control_dt: float,
    episodes: int,
    eval_bpm_points: int,
    seed_base: int,
) -> AccuracyProbe:
    """Core deterministic probe plus ADOFAI X-Accuracy / Perfect metrics."""

    jitter_s = phase.eval_phase_jitter_ms / 1000.0
    offsets = (
        (0.0,)
        if episodes == 1
        else tuple(-jitter_s + 2.0 * jitter_s * i / (episodes - 1) for i in range(episodes))
    )
    bpms = core.bpm_points(phase.bpm_min, phase.bpm_max, eval_bpm_points)
    buckets: dict[float, dict[str, object]] = {
        bpm: {
            "episodes": 0,
            "hits": 0,
            "targets": 0,
            "full": 0,
            "clean": 0,
            "errors": [],
            "xacc_points": 0.0,
            "perfects": 0,
        }
        for bpm in bpms
    }

    hits = targets = full = clean = overloads = too_early = perfects = 0
    xacc_points = 0.0
    errors: list[float] = []
    was_training = model.training
    model.eval()

    iterator = _bar(
        enumerate(offsets),
        total=len(offsets),
        desc="probe",
        unit="ep",
        leave=False,
        position=2,
    )
    for i, offset in iterator:
        bpm = bpms[i % len(bpms)]
        env = core.make_env(
            bpm=bpm,
            notes=phase.notes,
            start_s=max(0.050, core.curriculum_start_s(phase.notes) + offset),
            control_dt=control_dt,
            config=phase.vision,
            seed=seed_base + i * 1009,
        )
        observation = env.reset()
        state = model.initial_state(device)
        while True:
            action, state = model.deterministic_action(
                core.observation_tensor(observation, device), state
            )
            transition = env.step(action)
            observation = transition.observation
            if transition.done:
                break

        stats = env.stats
        is_full = stats.hits == stats.targets and not stats.overloaded
        is_clean = is_full and stats.too_early_presses == 0
        episode_errors = env.timing_errors_ms
        episode_xacc_points = stats.x_accuracy_percent * stats.targets / 100.0

        hits += stats.hits
        targets += stats.targets
        full += int(is_full)
        clean += int(is_clean)
        overloads += int(stats.overloaded)
        too_early += stats.too_early_presses
        perfects += stats.perfects
        xacc_points += episode_xacc_points
        errors.extend(episode_errors)

        bucket = buckets[bpm]
        bucket["episodes"] = int(bucket["episodes"]) + 1
        bucket["hits"] = int(bucket["hits"]) + stats.hits
        bucket["targets"] = int(bucket["targets"]) + stats.targets
        bucket["full"] = int(bucket["full"]) + int(is_full)
        bucket["clean"] = int(bucket["clean"]) + int(is_clean)
        bucket["xacc_points"] = float(bucket["xacc_points"]) + episode_xacc_points
        bucket["perfects"] = int(bucket["perfects"]) + stats.perfects
        bucket_errors = bucket["errors"]
        assert isinstance(bucket_errors, list)
        bucket_errors.extend(episode_errors)

    if was_training:
        model.train()

    slices = tuple(
        AccuracyBpmSlice(
            bpm=bpm,
            episodes=int(bucket["episodes"]),
            hits=int(bucket["hits"]),
            targets=int(bucket["targets"]),
            full=int(bucket["full"]),
            clean=int(bucket["clean"]),
            errors_ms=tuple(float(x) for x in bucket["errors"]),
            x_accuracy_percent=(
                100.0 * float(bucket["xacc_points"]) / max(1, int(bucket["targets"]))
            ),
            perfect_rate=int(bucket["perfects"]) / max(1, int(bucket["targets"])),
        )
        for bpm, bucket in buckets.items()
    )
    return AccuracyProbe(
        episodes=episodes,
        hits=hits,
        targets=targets,
        full=full,
        clean=clean,
        overloads=overloads,
        too_early=too_early,
        errors_ms=tuple(errors),
        bpm_slices=slices,
        x_accuracy_percent=100.0 * xacc_points / max(1, targets),
        perfect_rate=perfects / max(1, targets),
    )


def _probe_summary(label: str, probe: core.Probe) -> str:
    xacc = getattr(probe, "x_accuracy_percent", float("nan"))
    perfect = getattr(probe, "perfect_rate", float("nan"))
    return (
        f"{label:<9} hit={probe.hit_rate:.3f} full={probe.full}/{probe.episodes} "
        f"clean={probe.clean}/{probe.episodes} XAcc={xacc:6.2f}% PP={perfect:5.1%} "
        f"err={_fmt_ms(probe.mean_error_ms, signed=True)}ms "
        f"MAE={_fmt_ms(probe.mae_ms)}ms ovl={probe.overloads}"
    )


def _print_probe(label: str, probe: core.Probe) -> None:
    _write(_probe_summary(label, probe))
    if not probe.bpm_slices:
        return
    pieces: list[str] = []
    for item in probe.bpm_slices:
        xacc = getattr(item, "x_accuracy_percent", float("nan"))
        perfect = getattr(item, "perfect_rate", float("nan"))
        pieces.append(
            f"{item.bpm:g}: H{item.hit_rate:.2f} F{item.full_rate:.2f} "
            f"X{xacc:.1f} P{perfect:.0%} E{_fmt_ms(item.mean_error_ms, signed=True)}"
        )
    _write("  BPM  " + " | ".join(pieces))


def _print_gate(probe: core.Probe, retention: core.Retention, args, notes: int) -> None:
    completion = core.completion_passes(probe, retention, args, notes)
    precision = core.precision_passes(probe, args, notes)
    if notes == 1:
        _write(
            "  gate "
            f"completion={'PASS' if completion else 'wait'} "
            f"precision={'PASS' if precision else 'wait'} "
            f"clean={probe.clean_rate:.3f}/{args.stage1_clean_rate:.3f} "
            f"bias={_fmt_ms(probe.max_abs_bpm_bias_ms)}/{args.precision_max_bpm_bias_ms:.0f}ms"
        )
    else:
        _write(f"  gate completion={'PASS' if completion else 'wait'}")


def _print_header(args, phases: tuple[core.CurriculumPhase, ...], mode: str) -> None:
    _write("=== DMDOD / Planet Geometry PPO v0.3.1 ===")
    _write(
        f"mode={mode}  device={args.device}  seed={args.seed}  "
        f"checkpoint={args.checkpoint}"
    )
    _write(
        f"task={args.notes} notes  BPM={args.bpm_min:g}..{args.bpm_max:g}  "
        f"control={args.control_dt*1000:.1f}ms  vision={args.vision_hz:g}Hz/"
        f"{args.vision_latency_ms:g}ms"
    )
    _write(
        f"PPO={args.rollout_episodes} ep x {args.ppo_epochs} epochs  "
        f"clip={args.ppo_clip:.2f}  KL={args.target_kl:.3f}  lr={args.lr:g}"
    )
    _write(
        f"probe={args.eval_episodes} ep x {args.eval_bpm_points} BPM  "
        f"rollback={args.rollback_drop:.2f}  phases={len(phases)}"
    )
    _write("visible=motor+planet geometry | hidden=time/BPM/target-angle/error/direction")
    _write()


def train_ui(args) -> None:
    core.torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = core.torch.device(args.device)
    phases = core.build_curriculum(args)
    model = core.RecurrentActorCritic(
        input_dim=core.GEOMETRY_INPUT_DIM,
        hidden_dim=64,
        initial_log_std=args.initial_log_std,
    ).to(device)
    optimizer = core.torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint = Path(args.checkpoint)
    resume_phase = 0
    global_update = 0
    mode = "fresh"

    if args.resume and args.warm_start:
        raise SystemExit("--resume and --warm-start are mutually exclusive")
    if args.warm_start:
        source = Path(args.warm_start)
        if not source.exists():
            raise SystemExit(f"warm-start checkpoint not found: {source}")
        copied = core.warm_start(model, source, device)
        mode = f"warm-start:{source.name}"
        _write(f"warm-start copied: {copied}")
    if args.resume:
        if not checkpoint.exists():
            raise SystemExit(f"resume checkpoint not found: {checkpoint}")
        saved = core.torch.load(checkpoint, map_location=device)
        if saved.get("experiment") != "planet-geometry-ppo-v0.3":
            raise SystemExit("checkpoint is not geometry PPO v0.3")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        resume_phase = int(saved.get("curriculum_phase_index", 1)) - 1
        global_update = int(saved.get("global_update", 0))
        mode = f"resume:P{resume_phase + 1}"

    _print_header(args, phases, mode)

    phase_bar = _bar(
        total=len(phases),
        initial=resume_phase,
        desc="curriculum",
        unit="phase",
        position=0,
    )
    try:
        for phase_pos in range(resume_phase, len(phases)):
            phase = phases[phase_pos]
            phase_index = phase_pos + 1
            previous_notes = (
                tuple(notes for notes in core.note_curriculum(phase.notes) if notes < phase.notes)
                if phase.notes > 1
                else ()
            )
            phase_bar.set_description_str(
                f"P{phase_index:02d}/{len(phases):02d} {phase.name}"
            )
            _write(
                f"\n[P{phase_index:02d}/{len(phases):02d}] {phase.name}  "
                f"notes={phase.notes} BPM={phase.bpm_min:g}..{phase.bpm_max:g} "
                f"jitter=+/-{phase.train_phase_jitter_ms:g}ms"
            )

            baseline = deterministic_probe(
                model,
                device,
                phase=phase,
                control_dt=args.control_dt,
                episodes=args.eval_episodes,
                eval_bpm_points=args.eval_bpm_points,
                seed_base=args.seed * 1000000 + phase_index * 10000,
            )
            retention = core.previous_note_probes(
                model, device, previous_notes=previous_notes, args=args
            )
            _print_probe("baseline", baseline)
            _print_gate(baseline, retention, args, phase.notes)

            if core.passes(baseline, retention, args, phase.notes):
                _write("  status: already passed")
                core.save_checkpoint(
                    checkpoint,
                    model,
                    optimizer,
                    args=args,
                    phase_index=phase_index,
                    phase=phase,
                    global_update=global_update,
                    probe=baseline,
                )
                phase_bar.update(1)
                continue

            best_probe = baseline
            best_retention = retention
            best_key = core.rank_key(baseline, retention)
            best_model = copy.deepcopy(model.state_dict())
            best_optimizer = copy.deepcopy(optimizer.state_dict())
            core.save_checkpoint(
                checkpoint,
                model,
                optimizer,
                args=args,
                phase_index=phase_index,
                phase=phase,
                global_update=global_update,
                probe=baseline,
            )
            phase_passed = False

            update_bar = _bar(
                range(1, args.updates_per_phase + 1),
                total=args.updates_per_phase,
                desc=f"P{phase_index:02d} updates",
                unit="upd",
                leave=False,
                position=1,
            )
            for phase_update in update_bar:
                global_update += 1
                rollouts: list[core.Rollout] = []
                bpm_schedule = core.training_bpm_schedule(
                    phase,
                    episodes=args.rollout_episodes,
                    points=args.train_bpm_points,
                    rng=rng,
                )
                rollout_iter = _bar(
                    range(args.rollout_episodes),
                    total=args.rollout_episodes,
                    desc="rollouts",
                    unit="ep",
                    leave=False,
                    position=2,
                )
                for rollout_index in rollout_iter:
                    rollout_phase = phase
                    bpm_override: float | None = bpm_schedule[rollout_index]
                    if previous_notes and rng.random() < args.previous_stage_replay:
                        rollout_phase = core.full_vision_phase(args, rng.choice(previous_notes))
                        bpm_override = None
                    rollouts.append(
                        core.collect_rollout(
                            model,
                            device,
                            phase=rollout_phase,
                            control_dt=args.control_dt,
                            gamma=args.gamma,
                            rng=rng,
                            bpm_override=bpm_override,
                        )
                    )

                policy_loss, value_loss, entropy, approx_kl, epochs_done = core.ppo_update(
                    model, optimizer, rollouts, args
                )
                rollout_hit = sum(r.hits for r in rollouts) / max(
                    1, sum(r.targets for r in rollouts)
                )
                rollout_reward = sum(r.reward for r in rollouts) / len(rollouts)
                rollout_overload = sum(int(r.overloaded) for r in rollouts)
                update_bar.set_postfix_str(
                    f"H={rollout_hit:.3f} R={rollout_reward:+.2f} "
                    f"O={rollout_overload} sig={model.log_std.detach().exp().mean().item():.3f} "
                    f"KL={approx_kl:.4f} e={epochs_done}",
                    refresh=True,
                )

                if (
                    phase_update % args.eval_every_updates != 0
                    and phase_update != args.updates_per_phase
                ):
                    continue

                probe = deterministic_probe(
                    model,
                    device,
                    phase=phase,
                    control_dt=args.control_dt,
                    episodes=args.eval_episodes,
                    eval_bpm_points=args.eval_bpm_points,
                    seed_base=args.seed * 1000000 + phase_index * 10000,
                )
                retention = core.previous_note_probes(
                    model, device, previous_notes=previous_notes, args=args
                )
                _print_probe(f"eval {phase_update:02d}", probe)
                _print_gate(probe, retention, args, phase.notes)
                key = core.rank_key(probe, retention)

                if key > best_key:
                    best_key = key
                    best_probe = probe
                    best_retention = retention
                    best_model = copy.deepcopy(model.state_dict())
                    best_optimizer = copy.deepcopy(optimizer.state_dict())
                    core.save_checkpoint(
                        checkpoint,
                        model,
                        optimizer,
                        args=args,
                        phase_index=phase_index,
                        phase=phase,
                        global_update=global_update,
                        probe=probe,
                    )
                    _write(f"  saved best -> {checkpoint}")

                catastrophic = (
                    probe.hit_rate < best_probe.hit_rate - args.rollback_drop
                    or probe.full_rate < best_probe.full_rate - args.rollback_drop
                    or (best_probe.overloads == 0 and probe.overloads > 0)
                )
                if catastrophic:
                    model.load_state_dict(best_model)
                    optimizer.load_state_dict(best_optimizer)
                    for group in optimizer.param_groups:
                        group["lr"] = max(
                            args.min_lr,
                            float(group["lr"]) * args.rollback_lr_factor,
                        )
                    _write(
                        f"  rollback -> H={best_probe.hit_rate:.3f} "
                        f"F={best_probe.full_rate:.3f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e}"
                    )
                    continue

                if core.passes(probe, retention, args, phase.notes):
                    phase_passed = True
                    _write("  status: PASS (completion + precision)")
                    break

            update_bar.close()
            if not phase_passed:
                model.load_state_dict(best_model)
                optimizer.load_state_dict(best_optimizer)
                if core.completion_passes(best_probe, best_retention, args, phase.notes):
                    _write(
                        "  status: advance on stable completion "
                        f"H={best_probe.hit_rate:.3f} F={best_probe.full_rate:.3f} "
                        f"X={getattr(best_probe, 'x_accuracy_percent', float('nan')):.2f}% "
                        f"bias={_fmt_ms(best_probe.max_abs_bpm_bias_ms)}ms"
                    )
                    phase_bar.update(1)
                    continue
                _write(
                    f"  status: STOP best H={best_probe.hit_rate:.3f} "
                    f"F={best_probe.full_rate:.3f}"
                )
                break

            phase_bar.update(1)
    finally:
        phase_bar.close()

    _write()
    _write(f"best checkpoint: {checkpoint}")


def main() -> None:
    global PROGRESS_ENABLED
    if "--no-progress" in sys.argv:
        sys.argv.remove("--no-progress")
        PROGRESS_ENABLED = False
    elif not sys.stderr.isatty():
        PROGRESS_ENABLED = False

    # Reuse the established CLI/parser and training helpers. Only the trainer
    # presentation and deterministic probe reporting are replaced here.
    core.deterministic_probe = deterministic_probe
    core.train = train_ui
    core.main()


if __name__ == "__main__":
    main()
