from __future__ import annotations

import copy
import random
from pathlib import Path

try:
    from . import train_geometry_ppo_v042 as v042
except ImportError:  # direct script execution
    import train_geometry_ppo_v042 as v042


v041 = v042.v041
v040 = v042.v040
v039 = v042.v039
v038 = v042.v038
v036 = v042.v036
v035 = v042.v035
base = v042.base
core = v042.core

# v0.4.3: clean one-note speed expansion is primarily a supervised motor-skill
# curriculum. P3..P6 get a short privileged-teacher BC adaptation before PPO.
# If BC already satisfies the phase gate, the inherited trainer verifies and
# advances immediately, avoiding 40 PPO updates for a skill the teacher already
# knows how to demonstrate. P7+ remains perception/RL work: no target-time
# teacher is used once latency/sampling is introduced.
PHASE_BC_NAMES = frozenset(
    {
        "bpm-145-240-clean",
        "bpm-135-260-clean",
        "bpm-125-280-clean",
        "full-bpm-clean",
    }
)
PHASE_BC_EPISODES = 128
PHASE_BC_EPOCHS = 6
PHASE_BC_LR = 3e-4
PHASE_BC_ACTIVE_WEIGHT = v039.IMITATION_ACTIVE_WEIGHT
PHASE_BC_SEED = 4301

_ORIGINAL_V042_RAW_EVALUATE = v042._raw_evaluate
_ORIGINAL_V040_PPO_UPDATE = v040.ppo_update
_ORIGINAL_V042_SAVE_CHECKPOINT = v042.save_checkpoint

_PHASE_BC_HISTORY: list[dict[str, object]] = []
_BC_OPTIMIZER_RESET_PENDING = False
_TEACHER_CACHE: dict[float, tuple[object, object]] = {}


def _is_clean_vision(config) -> bool:
    return (
        abs(float(config.latency_s)) < 1e-12
        and abs(float(config.latency_jitter_s)) < 1e-12
        and abs(float(config.sample_period_s)) < 1e-12
        and abs(float(config.position_noise_std)) < 1e-12
        and abs(float(config.dropout_probability)) < 1e-12
    )


def phase_bc_eligible(phase) -> bool:
    """Teacher BC is restricted to the clean one-note speed curriculum."""

    return (
        int(phase.notes) == 1
        and phase.name in PHASE_BC_NAMES
        and _is_clean_vision(phase.vision)
    )


def _phase_demo_bpm(index: int, phase, rng: random.Random) -> float:
    """Oversample both new range edges while retaining continuous coverage."""

    low = float(phase.bpm_min)
    high = float(phase.bpm_max)
    if abs(high - low) < 1e-12:
        return low
    slot = index % 4
    if slot == 0:
        return low
    if slot == 1:
        return high
    return rng.uniform(low, high)


def _teacher_for(control_dt: float):
    key = round(float(control_dt), 12)
    cached = _TEACHER_CACHE.get(key)
    if cached is not None:
        return cached
    calibration = v039.calibrate_single_press_lead(
        control_dt_s=float(control_dt),
        same_hand=True,
    )
    teacher = v039.PrivilegedLeadTeacher(calibration)
    _TEACHER_CACHE[key] = (teacher, calibration)
    return teacher, calibration


def collect_phase_demonstrations(
    teacher,
    *,
    phase,
    episodes: int,
    control_dt: float,
    device,
    seed: int,
):
    """Generate teacher actions over the current clean phase BPM range."""

    torch = core.torch
    rng = random.Random(seed)
    demonstrations = []
    bpms: list[float] = []

    for episode in range(episodes):
        bpm = _phase_demo_bpm(episode, phase, rng)
        start_s = max(
            0.050,
            core.curriculum_start_s(1)
            + rng.uniform(
                -float(phase.train_phase_jitter_ms),
                float(phase.train_phase_jitter_ms),
            )
            / 1000.0,
        )
        env, target = v039._make_teacher_env(
            bpm=bpm,
            start_s=start_s,
            control_dt=control_dt,
            seed=seed + episode * 1009,
        )
        observation = env.reset()
        xs = []
        ys = []
        while True:
            action = teacher.action(
                now_s=env.motor.diagnostics().time_s,
                target_time_s=target.time_s,
                motor=observation.motor,
            )
            xs.append(v039.observation_tensor(observation, device).detach())
            ys.append(
                torch.tensor(
                    [action.left, action.right],
                    dtype=torch.float32,
                    device=device,
                )
            )
            transition = env.step(action)
            observation = transition.observation
            if transition.done:
                break
        demonstrations.append(v039.Demonstration(torch.stack(xs), torch.stack(ys)))
        bpms.append(bpm)

    return demonstrations, tuple(bpms)


def _phase_seed(phase) -> int:
    return (
        PHASE_BC_SEED * 100000
        + int(round(float(phase.bpm_min) * 10.0)) * 1009
        + int(round(float(phase.bpm_max) * 10.0))
    ) & 0x7FFFFFFF


def _reset_optimizer_after_bc(optimizer, args) -> None:
    """Discard stale Adam moments after supervised actor/GRU weight changes."""

    optimizer.state.clear()
    for group in optimizer.param_groups:
        group["lr"] = float(args.lr)


def _phase_aware_evaluate(
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
    """On screen00, try short phase-specific BC before falling back to PPO."""

    global _BC_OPTIMIZER_RESET_PENDING

    probe, retention = _ORIGINAL_V042_RAW_EVALUATE(
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

    if (
        label != "screen 00"
        or not phase_bc_eligible(phase)
        or base.passes(probe, retention, args, phase.notes)
    ):
        return probe, retention

    before_state = copy.deepcopy(model.state_dict())
    teacher, calibration = _teacher_for(args.control_dt)
    demonstrations, demo_bpms = collect_phase_demonstrations(
        teacher,
        phase=phase,
        episodes=PHASE_BC_EPISODES,
        control_dt=args.control_dt,
        device=device,
        seed=_phase_seed(phase),
    )
    bc_loss = v039.imitation_pretrain(
        model,
        demonstrations,
        epochs=PHASE_BC_EPOCHS,
        lr=PHASE_BC_LR,
        active_weight=PHASE_BC_ACTIVE_WEIGHT,
    )
    bc_probe, bc_retention = _ORIGINAL_V042_RAW_EVALUATE(
        model,
        device,
        phase=phase,
        previous_notes=previous_notes,
        args=args,
        episodes=episodes,
        retention_episodes=retention_episodes,
        seed_base=seed_base,
        label="bc screen",
    )

    accepted = base.rank_key(bc_probe, bc_retention) > base.rank_key(probe, retention)
    _PHASE_BC_HISTORY.append(
        {
            "phase": phase.name,
            "bpm_min": float(phase.bpm_min),
            "bpm_max": float(phase.bpm_max),
            "jitter_ms": float(phase.train_phase_jitter_ms),
            "episodes": PHASE_BC_EPISODES,
            "epochs": PHASE_BC_EPOCHS,
            "lr": PHASE_BC_LR,
            "loss": float(bc_loss),
            "teacher_lead_ms": float(calibration.lead_s) * 1000.0,
            "demo_bpm_min": min(demo_bpms) if demo_bpms else None,
            "demo_bpm_max": max(demo_bpms) if demo_bpms else None,
            "accepted": accepted,
            "quick_pass": bool(base.passes(bc_probe, bc_retention, args, phase.notes)),
            "xacc": base._xacc(bc_probe),
            "perfect_rate": base._pp(bc_probe),
            "min_xacc": base._min_xacc(bc_probe),
            "min_perfect_rate": base._min_pp(bc_probe),
        }
    )

    if not accepted:
        model.load_state_dict(before_state)
        base._write(
            f"  phase BC rejected loss={bc_loss:.5f}; keeping pre-BC policy"
        )
        return probe, retention

    _BC_OPTIMIZER_RESET_PENDING = True
    status = "PASS-candidate" if base.passes(
        bc_probe, bc_retention, args, phase.notes
    ) else "PPO-fallback"
    base._write(
        f"  phase BC accepted {phase.bpm_min:g}..{phase.bpm_max:g} "
        f"{PHASE_BC_EPISODES}ep/{PHASE_BC_EPOCHS}e loss={bc_loss:.5f} "
        f"[{status}]"
    )
    return bc_probe, bc_retention


def _ppo_update(model, optimizer, rollouts, args):
    global _BC_OPTIMIZER_RESET_PENDING

    if _BC_OPTIMIZER_RESET_PENDING:
        _reset_optimizer_after_bc(optimizer, args)
        fresh = copy.deepcopy(optimizer.state_dict())
        v040._GUARD_BEST_OPTIMIZER_STATE = fresh
        v040._LAST_OPTIMIZER_STATE = copy.deepcopy(fresh)
        _BC_OPTIMIZER_RESET_PENDING = False
        base._write(f"  PPO optimizer reset after phase BC; lr={args.lr:.2e}")
    return _ORIGINAL_V040_PPO_UPDATE(model, optimizer, rollouts, args)


def _print_header(args, phases, mode: str) -> None:
    base._write("=== DMDOD / Planet Geometry PPO v0.4.3 ===")
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
    base._write(
        f"input=14D visible motion | bootstrap imitation={int(v039._IMITATION_STATE.get('episodes', 0))}ep "
        f"lead={(float(lead)*1000 if lead is not None else float('nan')):.1f}ms"
    )
    mode_text = (
        f"{v041.ROLLOUT_WORKERS} process(es)"
        if args.device == "cpu"
        else "off (non-CPU device)"
    )
    base._write(
        f"phase BC=P3..P6 clean one-note {PHASE_BC_EPISODES}ep/{PHASE_BC_EPOCHS}epochs "
        f"lr={PHASE_BC_LR:g}; quick-pass -> verify -> skip PPO"
    )
    base._write(
        f"P7+=teacher off | parallel rollouts={mode_text} | "
        f"PPO+next-delta={v038.PREDICTION_COEF:g} | anchor={v040.ANCHOR_COEF_START:g}->0/{v040.ANCHOR_DECAY_UPDATES}upd"
    )
    base._write(
        f"gate guard: rollback only if worst +{v042.GATE_ROLLBACK_WORST_DELTA:.2f} "
        f"AND total +{v042.GATE_ROLLBACK_TOTAL_DELTA:.2f}"
    )
    if v035.VERBOSE_OUTPUT:
        base._write(
            "phase teacher sees target timestamp only while generating BC labels; student input remains motor+visible geometry/delta"
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
    _ORIGINAL_V042_SAVE_CHECKPOINT(
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
    saved["format_version"] = 22
    saved["trainer_ui_version"] = "0.4.3-phase-aware-imitation"
    saved["phase_imitation_v043"] = {
        "eligible_phases": sorted(PHASE_BC_NAMES),
        "episodes": PHASE_BC_EPISODES,
        "epochs": PHASE_BC_EPOCHS,
        "lr": PHASE_BC_LR,
        "edge_oversampling": True,
        "current_range_contains_prior_ranges": True,
        "ppo_only_if_bc_gate_unmet": True,
        "teacher_disabled_from_latency_sampling": True,
        "history": list(_PHASE_BC_HISTORY),
    }
    core.torch.save(saved, path)


def _load_history() -> None:
    checkpoint_arg = v038._arg_value("--checkpoint")
    if not checkpoint_arg:
        return
    path = Path(checkpoint_arg)
    if not path.exists():
        return
    saved = core.torch.load(path, map_location="cpu")
    state = saved.get("phase_imitation_v043")
    if isinstance(state, dict):
        history = state.get("history")
        if isinstance(history, list):
            _PHASE_BC_HISTORY.extend(
                dict(item) for item in history if isinstance(item, dict)
            )


def main() -> None:
    global PHASE_BC_EPISODES, PHASE_BC_EPOCHS, PHASE_BC_LR

    PHASE_BC_EPISODES = v039._pop_positive_int(
        "--phase-bc-episodes", PHASE_BC_EPISODES
    )
    PHASE_BC_EPOCHS = v039._pop_positive_int(
        "--phase-bc-epochs", PHASE_BC_EPOCHS
    )
    PHASE_BC_LR = v039._pop_positive_float("--phase-bc-lr", PHASE_BC_LR)
    _load_history()

    # v0.4.2 keeps the consistent outer rollback state machine and v0.4.1 keeps
    # parallel PPO rollouts. Inject only the screen00 BC stage and the one-time
    # optimizer reset required after supervised weight updates.
    v042._raw_evaluate = _phase_aware_evaluate
    v040.ppo_update = _ppo_update
    v042._print_header = _print_header
    v042.save_checkpoint = save_checkpoint
    v042.main()


if __name__ == "__main__":
    main()
