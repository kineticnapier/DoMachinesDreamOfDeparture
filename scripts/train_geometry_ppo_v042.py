from __future__ import annotations

import copy
from pathlib import Path

try:
    from . import train_geometry_ppo_v041 as v041
except ImportError:  # direct script execution
    import train_geometry_ppo_v041 as v041


v040 = v041.v040
v039 = v041.v039
v038 = v041.v038
v036 = v041.v036
v035 = v041.v035
base = v041.base
core = v041.core

# v0.4.2 fixes the v0.4.0/v0.4.1 rollback bookkeeping bug.  Evaluation no
# longer mutates the model.  A candidate is classified first, and a rejected
# candidate can therefore never be paired with an already-restored model and
# accidentally become the trainer's new "best" snapshot.
GATE_ROLLBACK_WORST_DELTA = 0.08
GATE_ROLLBACK_TOTAL_DELTA = 0.18

_ORIGINAL_V041_HEADER = v041._print_header
_ORIGINAL_V041_SAVE_CHECKPOINT = v041.save_checkpoint


def gate_deficit_score(probe, retention) -> tuple[float, float]:
    """Return worst and total normalized gate deficits for one probe."""

    deficits = v040._completion_deficits(probe, retention) + v040._precision_deficits(probe)
    return max(deficits, default=0.0), sum(deficits)


def material_gate_regression(probe, retention, best_probe, best_retention) -> bool:
    """Reject only a broad gate regression, not one noisy metric in isolation."""

    worst, total = gate_deficit_score(probe, retention)
    best_worst, best_total = gate_deficit_score(best_probe, best_retention)
    return (
        worst > best_worst + GATE_ROLLBACK_WORST_DELTA
        and total > best_total + GATE_ROLLBACK_TOTAL_DELTA
    )


def candidate_decision(probe, retention, best_probe, best_retention, args) -> str:
    """Classify a screen before any model/best-state mutation occurs."""

    if best_probe.overloads == 0 and probe.overloads > 0:
        return "rollback"

    completion_catastrophe = (
        probe.hit_rate < best_probe.hit_rate - args.rollback_drop
        or probe.full_rate < best_probe.full_rate - args.rollback_drop
        or base._min_hit(probe) < base._min_hit(best_probe) - args.rollback_drop
        or base._min_full(probe) < base._min_full(best_probe) - args.rollback_drop
    )
    if completion_catastrophe:
        return "rollback"
    if material_gate_regression(probe, retention, best_probe, best_retention):
        return "rollback"
    if base.rank_key(probe, retention) > base.rank_key(best_probe, best_retention):
        return "best"
    return "keep"


def _raw_evaluate(
    model,
    device,
    *,
    phase,
    previous_notes,
    args,
    episodes,
    retention_episodes,
    seed_base,
    label,
):
    """Use v0.3.6 evaluation without v0.4.0's in-evaluator rollback side effect."""

    return v040._ORIGINAL_V036_EVALUATE(
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


def _sync_guard_best(model, optimizer, probe) -> None:
    """Keep v0.4.0's behavior anchor consistent with the outer best snapshot."""

    v040._GUARD_BEST_PROBE = probe
    v040._GUARD_BEST_MODEL_STATE = copy.deepcopy(model.state_dict())
    v040._GUARD_BEST_OPTIMIZER_STATE = copy.deepcopy(optimizer.state_dict())
    v040._LAST_OPTIMIZER_STATE = copy.deepcopy(optimizer.state_dict())
    v040._ROLLBACK_PENDING = False
    v040._set_anchor_from_model(model)


def parallel_train_ui(args) -> None:
    core.torch.manual_seed(args.seed)
    rng = v041.random.Random(args.seed)
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

    use_parallel = device.type == "cpu" and v041.ROLLOUT_WORKERS > 1
    executor = None
    if use_parallel:
        executor = v041.ProcessPoolExecutor(
            max_workers=v041.ROLLOUT_WORKERS,
            mp_context=v041.mp.get_context("spawn"),
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

            # Reset the behavior anchor at the actual phase boundary, but do not
            # let evaluation itself restore/replace policy state.
            v040._reset_phase_guard(model, phase)

            quick_seed = args.seed * 1000000 + phase_index * 10000
            full_seed = quick_seed + 500000
            baseline, baseline_retention = _raw_evaluate(
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
                verified, verified_retention = _raw_evaluate(
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
            best_model = copy.deepcopy(model.state_dict())
            best_optimizer = copy.deepcopy(optimizer.state_dict())
            _sync_guard_best(model, optimizer, best_probe)
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
                specs: list[v041.ParallelRolloutSpec] = []
                for rollout_index in range(args.rollout_episodes):
                    rollout_phase = phase
                    bpm_override: float | None = bpm_schedule[rollout_index]
                    if previous_notes and rng.random() < args.previous_stage_replay:
                        rollout_phase = core.full_vision_phase(args, rng.choice(previous_notes))
                        bpm_override = None
                    specs.append(
                        v041._resolve_rollout_spec(
                            index=rollout_index,
                            phase=rollout_phase,
                            bpm_override=bpm_override,
                            args=args,
                            rng=rng,
                        )
                    )

                if executor is not None:
                    rollouts = v041._parallel_rollouts(executor, model, device, specs)
                else:
                    rollouts = v041._sequential_specs(model, device, specs)

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

                probe, retention = _raw_evaluate(
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

                decision = candidate_decision(
                    probe, retention, best_probe, best_retention, args
                )
                if decision == "rollback":
                    current_worst, current_total = gate_deficit_score(probe, retention)
                    best_worst, best_total = gate_deficit_score(best_probe, best_retention)
                    model.load_state_dict(best_model)
                    optimizer.load_state_dict(best_optimizer)
                    for group in optimizer.param_groups:
                        group["lr"] = max(
                            args.min_lr, float(group["lr"]) * args.rollback_lr_factor
                        )
                    # Persist the reduced LR as part of the restored best state;
                    # repeated rollbacks therefore continue reducing it instead
                    # of jumping back to the same pre-rollback LR each time.
                    best_optimizer = copy.deepcopy(optimizer.state_dict())
                    _sync_guard_best(model, optimizer, best_probe)
                    focus = base.weak_bpms(best_probe)
                    base._write(
                        "  gate rollback -> "
                        f"worst={current_worst:.3f}/{best_worst:.3f} "
                        f"total={current_total:.3f}/{best_total:.3f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e}"
                    )
                    # Crucial: a rejected probe is never considered for best
                    # selection after the model has already been restored.
                    continue

                focus = base.weak_bpms(probe)
                if decision == "best":
                    best_probe = probe
                    best_retention = retention
                    best_model = copy.deepcopy(model.state_dict())
                    best_optimizer = copy.deepcopy(optimizer.state_dict())
                    _sync_guard_best(model, optimizer, best_probe)
                    base._write(
                        f"  best screen X={base._xacc(probe):.2f}% PP={base._pp(probe):.1%} "
                        f"minX={base._min_xacc(probe):.1f}% minPP={base._min_pp(probe):.0%}"
                    )

                if base.passes(probe, retention, args, phase.notes):
                    verified, verified_retention = _raw_evaluate(
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
                _sync_guard_best(model, optimizer, best_probe)
                verified, verified_retention = _raw_evaluate(
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
    base._write("=== DMDOD / Planet Geometry PPO v0.4.2 ===")
    base._write(
        f"mode={mode} device={args.device} seed={args.seed} checkpoint={args.checkpoint}"
    )
    base._write(
        f"task={args.notes} BPM={args.bpm_min:g}..{args.bpm_max:g} "
        f"vision={args.vision_hz:g}Hz/{args.vision_latency_ms:g}ms | "
        f"gate X={base.PRECISION_OVERALL_XACC:g}%/min{base.PRECISION_MIN_BPM_XACC:g}% "
        f"PP={base.PRECISION_OVERALL_PP:.0%}/min{base.PRECISION_MIN_BPM_PP:.0%}"
    )
    lead = v039._IMITATION_STATE.get("teacher_lead_s")
    probe = v039._IMITATION_STATE.get("student_probe")
    probe145 = probe.get("edge_145_error_ms") if isinstance(probe, dict) else None
    probe240 = probe.get("edge_240_error_ms") if isinstance(probe, dict) else None
    base._write(
        f"input=14D visible motion | imitation={int(v039._IMITATION_STATE.get('episodes', 0))}ep "
        f"lead={(float(lead)*1000 if lead is not None else float('nan')):.1f}ms "
        f"probe145={v039._fmt_optional(probe145)}ms probe240={v039._fmt_optional(probe240)}ms"
    )
    mode_text = (
        f"{v041.ROLLOUT_WORKERS} process(es)"
        if args.device == "cpu"
        else "off (non-CPU device)"
    )
    base._write(
        f"parallel rollouts={mode_text} | PPO+next-delta={v038.PREDICTION_COEF:g} | "
        f"anchor={v040.ANCHOR_COEF_START:g}->0/{v040.ANCHOR_DECAY_UPDATES}upd"
    )
    base._write(
        f"consistent gate guard: rollback only if worst +{GATE_ROLLBACK_WORST_DELTA:.2f} "
        f"AND total +{GATE_ROLLBACK_TOTAL_DELTA:.2f}; rejected screens cannot become best"
    )
    if v035.VERBOSE_OUTPUT:
        base._write(
            "teacher target time is BC-only; policy input remains motor+visible geometry/delta"
        )
    else:
        base._write("compact output; use --verbose for per-BPM details")
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
    _ORIGINAL_V041_SAVE_CHECKPOINT(
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
    saved["format_version"] = 21
    saved["trainer_ui_version"] = "0.4.2-consistent-gate-guard"
    saved["gate_guard_v042"] = {
        "rollback_worst_deficit_delta": GATE_ROLLBACK_WORST_DELTA,
        "rollback_total_deficit_delta": GATE_ROLLBACK_TOTAL_DELTA,
        "requires_both_deficit_regressions": True,
        "rejected_probe_cannot_update_best": True,
        "model_probe_snapshot_consistent": True,
        "rollback_lr_persists": True,
    }
    core.torch.save(saved, path)


def main() -> None:
    v041._pop_workers()
    # Reuse v0.4.1's worker implementation and v0.4.0's PPO/behavior anchor,
    # replacing only the outer training/evaluation state machine.
    v041.parallel_train_ui = parallel_train_ui
    v041._print_header = _print_header
    v041.save_checkpoint = save_checkpoint
    v041.main()


if __name__ == "__main__":
    main()
