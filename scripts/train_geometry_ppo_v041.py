from __future__ import annotations

import copy
import multiprocessing as mp
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:
    from . import train_geometry_ppo_v040 as v040
except ImportError:  # direct script execution
    import train_geometry_ppo_v040 as v040

from dmdod.parallel_rollout import ParallelRolloutSpec, collect_rollout_batch


v039 = v040.v039
v038 = v040.v038
v036 = v040.v036
v035 = v040.v035
base = v040.base
core = base.core

ROLLOUT_WORKERS = max(1, min(6, (os.cpu_count() or 2) // 2))
_ORIGINAL_V040_HEADER = v040._print_header
_ORIGINAL_V040_SAVE_CHECKPOINT = v040.save_checkpoint


def _pop_workers() -> None:
    global ROLLOUT_WORKERS
    value = v038._pop_arg_value("--rollout-workers")
    if value is None:
        return
    ROLLOUT_WORKERS = int(value)
    if ROLLOUT_WORKERS <= 0:
        raise SystemExit("--rollout-workers must be positive")


def _vision_dict(config) -> dict[str, float]:
    return {
        "latency_s": float(config.latency_s),
        "latency_jitter_s": float(config.latency_jitter_s),
        "sample_period_s": float(config.sample_period_s),
        "position_noise_std": float(config.position_noise_std),
        "dropout_probability": float(config.dropout_probability),
    }


def _resolve_rollout_spec(
    *,
    index: int,
    phase,
    bpm_override: float | None,
    args,
    rng: random.Random,
) -> ParallelRolloutSpec:
    bpm = (
        float(bpm_override)
        if bpm_override is not None
        else (
            float(phase.bpm_min)
            if abs(float(phase.bpm_max) - float(phase.bpm_min)) < 1e-12
            else rng.uniform(float(phase.bpm_min), float(phase.bpm_max))
        )
    )
    start_s = max(
        0.050,
        core.curriculum_start_s(phase.notes)
        + rng.uniform(-phase.train_phase_jitter_ms, phase.train_phase_jitter_ms) / 1000.0,
    )
    env_seed = rng.randrange(0, 2**31)
    # Derive exploration randomness without consuming the curriculum RNG again.
    policy_seed = (env_seed ^ (index * 0x45D9F3B) ^ int(bpm * 1000.0)) & 0x7FFFFFFF
    return ParallelRolloutSpec(
        index=index,
        notes=int(phase.notes),
        bpm=bpm,
        start_s=start_s,
        control_dt=float(args.control_dt),
        vision=_vision_dict(phase.vision),
        env_seed=env_seed,
        policy_seed=policy_seed,
        gamma=float(args.gamma),
    )


def _chunks(specs: list[ParallelRolloutSpec], workers: int) -> list[list[ParallelRolloutSpec]]:
    workers = max(1, min(int(workers), len(specs)))
    result = [[] for _ in range(workers)]
    for i, spec in enumerate(specs):
        result[i % workers].append(spec)
    return [chunk for chunk in result if chunk]


def _payload_to_rollout(payload: dict[str, object], device):
    torch = core.torch
    return core.Rollout(
        observations=torch.tensor(payload["observations"], dtype=torch.float32, device=device),
        latents=torch.tensor(payload["latents"], dtype=torch.float32, device=device),
        old_log_probs=torch.tensor(payload["old_log_probs"], dtype=torch.float32, device=device),
        old_values=torch.tensor(payload["old_values"], dtype=torch.float32, device=device),
        returns=torch.tensor(payload["returns"], dtype=torch.float32, device=device),
        advantages=torch.tensor(payload["advantages"], dtype=torch.float32, device=device),
        hits=int(payload["hits"]),
        targets=int(payload["targets"]),
        too_early=int(payload["too_early"]),
        overloaded=bool(payload["overloaded"]),
        reward=float(payload["reward"]),
    )


def _parallel_rollouts(
    executor: ProcessPoolExecutor,
    model,
    device,
    specs: list[ParallelRolloutSpec],
) -> list:
    # Freeze one CPU snapshot per PPO update. Each worker receives it once and
    # reuses its reconstructed model for several episodes.
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    chunks = _chunks(specs, ROLLOUT_WORKERS)
    futures = [
        executor.submit(
            collect_rollout_batch,
            state,
            int(model.hidden_dim),
            chunk,
            too_early_penalty=v035.TOO_EARLY_PENALTY,
        )
        for chunk in chunks
    ]
    payloads = []
    for future in futures:
        payloads.extend(future.result())
    payloads.sort(key=lambda item: int(item["index"]))
    return [_payload_to_rollout(item, device) for item in payloads]


def _sequential_specs(model, device, specs: list[ParallelRolloutSpec]):
    # Keep a no-multiprocessing fallback for debugging and non-CPU devices.
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    payloads = collect_rollout_batch(
        state,
        int(model.hidden_dim),
        specs,
        too_early_penalty=v035.TOO_EARLY_PENALTY,
    )
    payloads.sort(key=lambda item: int(item["index"]))
    return [_payload_to_rollout(item, device) for item in payloads]


def parallel_train_ui(args) -> None:
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
        base._write(f"warm-start copied: {copied}")
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

    base._print_header(args, phases, mode)
    phase_bar = base._bar(
        total=len(phases), initial=resume_phase, desc="curriculum", unit="phase", position=0
    )

    use_parallel = device.type == "cpu" and ROLLOUT_WORKERS > 1
    executor = None
    if use_parallel:
        executor = ProcessPoolExecutor(
            max_workers=ROLLOUT_WORKERS,
            mp_context=mp.get_context("spawn"),
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
            phase_bar.set_description_str(f"P{phase_index:02d}/{len(phases):02d} {phase.name}")
            base._write(
                f"\n[P{phase_index:02d}/{len(phases):02d}] {phase.name}  "
                f"notes={phase.notes} BPM={phase.bpm_min:g}..{phase.bpm_max:g} "
                f"jitter=+/-{phase.train_phase_jitter_ms:g}ms"
            )

            quick_seed = args.seed * 1000000 + phase_index * 10000
            full_seed = quick_seed + 500000
            baseline, baseline_retention = base._evaluate(
                model,
                device,
                phase=phase,
                previous_notes=previous_notes,
                args=args,
                episodes=base.QUICK_EVAL_EPISODES,
                retention_episodes=base.QUICK_RETENTION_EPISODES,
                seed_base=quick_seed,
                label="screen 00",
            )

            if base.passes(baseline, baseline_retention, args, phase.notes):
                verified, verified_retention = base._evaluate(
                    model,
                    device,
                    phase=phase,
                    previous_notes=previous_notes,
                    args=args,
                    episodes=args.eval_episodes,
                    retention_episodes=args.retention_episodes,
                    seed_base=full_seed,
                    label="verify",
                )
                if base.passes(verified, verified_retention, args, phase.notes):
                    base._write("  status: already passed (verified)")
                    base.save_checkpoint(
                        checkpoint, model, optimizer, args=args,
                        phase_index=phase_index, phase=phase,
                        global_update=global_update, probe=verified,
                    )
                    phase_bar.update(1)
                    continue

            best_probe = baseline
            best_retention = baseline_retention
            best_key = base.rank_key(baseline, baseline_retention)
            best_model = copy.deepcopy(model.state_dict())
            best_optimizer = copy.deepcopy(optimizer.state_dict())
            focus = base.weak_bpms(baseline)
            phase_passed = False
            eval_every = base._eval_interval(args, phase_index)

            update_bar = base._bar(
                range(1, args.updates_per_phase + 1),
                total=args.updates_per_phase,
                desc=f"P{phase_index:02d} updates",
                unit="upd", leave=False, position=1,
            )
            for phase_update in update_bar:
                global_update += 1
                bpm_schedule = base.focused_training_bpm_schedule(
                    phase,
                    episodes=args.rollout_episodes,
                    points=args.train_bpm_points,
                    rng=rng,
                    focus_bpms=focus,
                )
                specs: list[ParallelRolloutSpec] = []
                for rollout_index in range(args.rollout_episodes):
                    rollout_phase = phase
                    bpm_override: float | None = bpm_schedule[rollout_index]
                    if previous_notes and rng.random() < args.previous_stage_replay:
                        rollout_phase = core.full_vision_phase(args, rng.choice(previous_notes))
                        bpm_override = None
                    specs.append(
                        _resolve_rollout_spec(
                            index=rollout_index,
                            phase=rollout_phase,
                            bpm_override=bpm_override,
                            args=args,
                            rng=rng,
                        )
                    )

                if executor is not None:
                    rollouts = _parallel_rollouts(executor, model, device, specs)
                else:
                    rollouts = _sequential_specs(model, device, specs)

                _, _, _, approx_kl, epochs_done = core.ppo_update(
                    model, optimizer, rollouts, args
                )
                rollout_hit = sum(r.hits for r in rollouts) / max(
                    1, sum(r.targets for r in rollouts)
                )
                rollout_reward = sum(r.reward for r in rollouts) / len(rollouts)
                rollout_overload = sum(int(r.overloaded) for r in rollouts)
                focus_text = "/".join(f"{b:g}" for b in focus) or "-"
                update_bar.set_postfix_str(
                    f"H={rollout_hit:.3f} R={rollout_reward:+.2f} O={rollout_overload} "
                    f"focus={focus_text} KL={approx_kl:.4f} e={epochs_done}",
                    refresh=True,
                )

                if phase_update % eval_every != 0 and phase_update != args.updates_per_phase:
                    continue

                probe, retention = base._evaluate(
                    model,
                    device,
                    phase=phase,
                    previous_notes=previous_notes,
                    args=args,
                    episodes=base.QUICK_EVAL_EPISODES,
                    retention_episodes=base.QUICK_RETENTION_EPISODES,
                    seed_base=quick_seed,
                    label=f"screen {phase_update:02d}",
                )
                focus = base.weak_bpms(probe)
                key = base.rank_key(probe, retention)

                if key > best_key:
                    best_key = key
                    best_probe = probe
                    best_retention = retention
                    best_model = copy.deepcopy(model.state_dict())
                    best_optimizer = copy.deepcopy(optimizer.state_dict())
                    base._write(
                        f"  best screen X={base._xacc(probe):.2f}% PP={base._pp(probe):.1%} "
                        f"minX={base._min_xacc(probe):.1f}% minPP={base._min_pp(probe):.0%}"
                    )

                catastrophic = (
                    probe.hit_rate < best_probe.hit_rate - args.rollback_drop
                    or probe.full_rate < best_probe.full_rate - args.rollback_drop
                    or base._min_hit(probe) < base._min_hit(best_probe) - args.rollback_drop
                    or base._min_full(probe) < base._min_full(best_probe) - args.rollback_drop
                    or (best_probe.overloads == 0 and probe.overloads > 0)
                )
                if catastrophic:
                    model.load_state_dict(best_model)
                    optimizer.load_state_dict(best_optimizer)
                    focus = base.weak_bpms(best_probe)
                    for group in optimizer.param_groups:
                        group["lr"] = max(
                            args.min_lr, float(group["lr"]) * args.rollback_lr_factor
                        )
                    base._write(
                        f"  rollback -> H={best_probe.hit_rate:.3f} "
                        f"F={best_probe.full_rate:.3f} lr={optimizer.param_groups[0]['lr']:.2e}"
                    )
                    continue

                if base.passes(probe, retention, args, phase.notes):
                    verified, verified_retention = base._evaluate(
                        model,
                        device,
                        phase=phase,
                        previous_notes=previous_notes,
                        args=args,
                        episodes=args.eval_episodes,
                        retention_episodes=args.retention_episodes,
                        seed_base=full_seed,
                        label="verify",
                    )
                    if base.passes(verified, verified_retention, args, phase.notes):
                        base.save_checkpoint(
                            checkpoint, model, optimizer, args=args,
                            phase_index=phase_index, phase=phase,
                            global_update=global_update, probe=verified,
                        )
                        phase_passed = True
                        base._write("  status: PASS (full verification)")
                        break
                    base._write("  verify failed; continue training")

            update_bar.close()

            if not phase_passed:
                model.load_state_dict(best_model)
                optimizer.load_state_dict(best_optimizer)
                verified, verified_retention = base._evaluate(
                    model,
                    device,
                    phase=phase,
                    previous_notes=previous_notes,
                    args=args,
                    episodes=args.eval_episodes,
                    retention_episodes=args.retention_episodes,
                    seed_base=full_seed,
                    label="final",
                )
                base.save_checkpoint(
                    checkpoint, model, optimizer, args=args,
                    phase_index=phase_index, phase=phase,
                    global_update=global_update, probe=verified,
                )

                if base.passes(verified, verified_retention, args, phase.notes):
                    base._write("  status: PASS at final verification")
                    phase_bar.update(1)
                    continue
                if base.ALLOW_PRECISION_SKIP and base.completion_passes(
                    verified, verified_retention, args, phase.notes
                ):
                    base._write(
                        "  status: precision skip "
                        f"X={base._xacc(verified):.2f}% PP={base._pp(verified):.1%} "
                        f"minX={base._min_xacc(verified):.1f}% minPP={base._min_pp(verified):.0%}"
                    )
                    phase_bar.update(1)
                    continue
                base._write(
                    "  status: STOP accuracy target unmet "
                    f"H={verified.hit_rate:.3f} F={verified.full_rate:.3f} "
                    f"X={base._xacc(verified):.2f}% PP={base._pp(verified):.1%} "
                    f"minX={base._min_xacc(verified):.1f}% minPP={base._min_pp(verified):.0%}"
                )
                break

            phase_bar.update(1)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        phase_bar.close()

    base._write()
    base._write(f"best checkpoint: {checkpoint}")


def _print_header(args, phases, mode: str) -> None:
    _ORIGINAL_V040_HEADER(args, phases, mode)
    mode_text = f"{ROLLOUT_WORKERS} process(es)" if args.device == "cpu" else "off (non-CPU device)"
    base._write(f"v0.4.1 rollout parallelism={mode_text}; worker torch threads=1")
    base._write()


def save_checkpoint(
    path,
    model,
    optimizer,
    *,
    args,
    phase_index: int,
    phase,
    global_update: int,
    probe,
) -> None:
    _ORIGINAL_V040_SAVE_CHECKPOINT(
        path,
        model,
        optimizer,
        args=args,
        phase_index=phase_index,
        phase=phase,
        global_update=global_update,
        probe=probe,
    )
    saved = core.torch.load(path, map_location="cpu")
    saved["format_version"] = 20
    saved["trainer_ui_version"] = "0.4.1-parallel-rollouts"
    saved["parallel_rollouts_v041"] = {
        "workers": ROLLOUT_WORKERS,
        "cpu_only": True,
        "worker_torch_threads": 1,
        "batch_per_worker": True,
    }
    core.torch.save(saved, path)


def main() -> None:
    _pop_workers()
    # v0.4.0 installs the precision guard and PPO update. Replace only the
    # inherited training loop so rollout episodes can be collected concurrently.
    base.train_ui = parallel_train_ui
    v040._print_header = _print_header
    v040.save_checkpoint = save_checkpoint
    v040.main()


if __name__ == "__main__":
    main()
