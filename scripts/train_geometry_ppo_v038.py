from __future__ import annotations

import sys
from pathlib import Path

try:
    from . import train_geometry_ppo_v036 as v036
except ImportError:  # direct script execution
    import train_geometry_ppo_v036 as v036

from dmdod.motion_geometry_env import MotionGeometryEnv
from dmdod.predictive_recurrent_policy import PredictiveRecurrentActorCritic
from dmdod.rhythm_env import RewardConfig, make_regular_targets
from dmdod.toy_policy import MOTION_GEOMETRY_INPUT_DIM


v035 = v036.v035
base = v036.base

# v0.3.8 isolates the P3 problem from chart memory.  The policy gets only a
# two-frame visual delta, while an auxiliary self-supervised head must predict
# the *next* visible delta from the recurrent state.  This makes apparent speed
# useful without exposing BPM, chart time, target time, angle error, or future
# simulator state.
PREDICTION_COEF = 2.0
MOTION_FEATURE_START = 10
MOTION_FEATURE_END = 14
MAX_PREDICTION_TARGET_NORM = 0.50
LAST_PREDICTION_LOSS = 0.0

_ORIGINAL_V036_SAVE_CHECKPOINT = v036.save_checkpoint
_ORIGINAL_WARM_START = base.core.warm_start


def make_env(
    *,
    bpm: float,
    notes: int,
    start_s: float,
    control_dt: float,
    config,
    seed: int,
) -> MotionGeometryEnv:
    return MotionGeometryEnv(
        make_regular_targets(bpm=bpm, count=notes, start_s=start_s, pattern="left"),
        bpm=bpm,
        same_hand=True,
        control_dt_s=control_dt,
        vision_config=config,
        perception_seed=seed,
        reward_config=RewardConfig(too_early_penalty=v035.TOO_EARLY_PENALTY),
    )


def _copy_recurrent_weights(model, source_state: dict[str, object]) -> str:
    """Migrate an older geometry policy without importing chart-memory inputs."""

    target = model.state_dict()
    copied: list[str] = []
    for name, target_value in target.items():
        source_value = source_state.get(name)
        if source_value is None:
            continue
        if source_value.shape == target_value.shape:
            target_value.copy_(source_value)
            copied.append(name)
            continue
        if (
            name == "input_layer.weight"
            and source_value.ndim == 2
            and target_value.ndim == 2
            and source_value.shape[0] == target_value.shape[0]
        ):
            # Columns 0..9 are the old motor+current-geometry observation.
            # New explicit motion columns start with zero influence, preserving
            # the v0.3.6 policy exactly when migrating a 10-D checkpoint.
            target_value.zero_()
            columns = min(10, source_value.shape[1], target_value.shape[1])
            target_value[:, :columns].copy_(source_value[:, :columns])
            copied.append(f"{name}[0:{columns}]")

    model.load_state_dict(target)
    return ", ".join(copied) or "no compatible recurrent weights"


def warm_start(model, checkpoint: Path, device) -> str:
    saved = base.core.torch.load(checkpoint, map_location=device)
    source = saved.get("model")
    if saved.get("experiment") == "planet-geometry-ppo-v0.3" and isinstance(source, dict):
        return _copy_recurrent_weights(model, source)
    return _ORIGINAL_WARM_START(model, checkpoint, device)


def _arg_value(name: str, default: str | None = None) -> str | None:
    prefix = name + "="
    for i, value in enumerate(sys.argv):
        if value.startswith(prefix):
            return value[len(prefix) :]
        if value == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def _pop_arg_value(name: str) -> str | None:
    prefix = name + "="
    for i, value in enumerate(tuple(sys.argv)):
        if value.startswith(prefix):
            sys.argv.remove(value)
            return value[len(prefix) :]
        if value == name:
            if i + 1 >= len(sys.argv):
                raise SystemExit(f"{name} requires a value")
            result = sys.argv[i + 1]
            del sys.argv[i : i + 2]
            return result
    return None


def _optimizer_lr(saved: dict[str, object]) -> float:
    optimizer = saved.get("optimizer")
    if isinstance(optimizer, dict):
        groups = optimizer.get("param_groups")
        if isinstance(groups, list) and groups:
            lr = groups[0].get("lr")
            if lr is not None:
                return float(lr)
    return 1e-4


def _upgrade_resume_if_requested() -> None:
    """Create a 14-D predictive checkpoint while preserving curriculum P*."""

    source_arg = _pop_arg_value("--upgrade-resume")
    if source_arg is None:
        return

    source = Path(source_arg)
    target = Path(
        _arg_value("--checkpoint", "checkpoints/planet_geometry_v10_predictive.pt")
        or "checkpoints/planet_geometry_v10_predictive.pt"
    )
    if source.resolve() == target.resolve():
        raise SystemExit("--upgrade-resume requires a different --checkpoint destination")
    if not source.exists():
        raise SystemExit(f"upgrade source checkpoint not found: {source}")

    saved = base.core.torch.load(source, map_location="cpu")
    if saved.get("experiment") != "planet-geometry-ppo-v0.3":
        raise SystemExit("upgrade source is not geometry PPO v0.3")
    source_state = saved.get("model")
    if not isinstance(source_state, dict):
        raise SystemExit("upgrade source has no model state")

    model = PredictiveRecurrentActorCritic(
        input_dim=MOTION_GEOMETRY_INPUT_DIM,
        hidden_dim=int(saved.get("hidden_dim", 64)),
        initial_log_std=float(saved.get("initial_log_std", -0.70)),
    )
    copied = _copy_recurrent_weights(model, source_state)
    optimizer = base.core.torch.optim.Adam(model.parameters(), lr=_optimizer_lr(saved))

    migrated = dict(saved)
    # v0.3.7's chart memory is deliberately not part of this experiment even if
    # somebody migrates from a v09 checkpoint instead of the recommended v08.
    migrated.pop("pattern_memory_v037", None)
    migrated.pop("pattern_memory_config", None)
    migrated.pop("v037_migration", None)
    migrated["format_version"] = 17
    migrated["model"] = model.state_dict()
    migrated["optimizer"] = optimizer.state_dict()
    migrated["input_dim"] = MOTION_GEOMETRY_INPUT_DIM
    migrated["trainer_ui_version"] = "0.3.8-predictive-motion"
    migrated["predictive_motion_v038"] = {
        "source": str(source),
        "copied": copied,
        "optimizer_reset": True,
        "curriculum_position_preserved": True,
        "explicit_frame_delta": True,
        "next_delta_prediction": True,
        "prediction_coef": PREDICTION_COEF,
        "chart_memory": False,
        "privileged_targets": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    base.core.torch.save(migrated, target)

    if "--resume" not in sys.argv:
        sys.argv.append("--resume")
    base._write(
        f"v0.3.8 checkpoint upgrade: {source.name} -> {target.name}; "
        f"resume P{int(saved.get('curriculum_phase_index', 1))}"
    )


def ppo_update(model, optimizer, rollouts, args):
    """PPO plus self-supervised next-visible-motion prediction."""

    global LAST_PREDICTION_LOSS
    torch = base.core.torch
    nn = base.core.nn

    all_advantages = torch.cat([rollout.advantages for rollout in rollouts])
    adv_mean = all_advantages.mean()
    adv_std = all_advantages.std(unbiased=False).clamp_min(1e-6)
    old_log_probs = torch.cat([rollout.old_log_probs for rollout in rollouts])
    returns = torch.cat([rollout.returns for rollout in rollouts])

    final_policy = final_value = final_entropy = final_kl = 0.0
    final_prediction = 0.0
    epochs_done = 0
    model.train()

    for epoch in range(args.ppo_epochs):
        new_log_probs_list = []
        new_values_list = []
        entropies_list = []
        normalized_advantages = []
        prediction_losses = []

        for rollout in rollouts:
            new_log_probs, new_values, entropies, predictions = (
                model.evaluate_latent_sequence_predictive(
                    rollout.observations,
                    rollout.latents,
                )
            )
            new_log_probs_list.append(new_log_probs)
            new_values_list.append(new_values)
            entropies_list.append(entropies)
            normalized_advantages.append((rollout.advantages - adv_mean) / adv_std)

            if rollout.observations.shape[0] > 1:
                # Observation[t+1, 10:14] is the visible delta from frame t to
                # frame t+1.  It is therefore a self-supervised target available
                # from the same observations the agent receives, not simulator
                # truth.  Large jumps are masked because they are usually visual
                # dropout or target-switch discontinuities rather than motion.
                target = rollout.observations[
                    1:, MOTION_FEATURE_START:MOTION_FEATURE_END
                ].detach()
                prediction = predictions[:-1]
                valid = torch.linalg.vector_norm(target, dim=1) <= MAX_PREDICTION_TARGET_NORM
                if bool(valid.any()):
                    prediction_losses.append(
                        ((prediction[valid] - target[valid]) ** 2).mean()
                    )

        new_log_probs = torch.cat(new_log_probs_list)
        new_values = torch.cat(new_values_list)
        entropies = torch.cat(entropies_list)
        advantages = torch.cat(normalized_advantages)
        ratios = torch.exp(new_log_probs - old_log_probs)
        unclipped = ratios * advantages
        clipped = torch.clamp(
            ratios, 1.0 - args.ppo_clip, 1.0 + args.ppo_clip
        ) * advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        value_loss = ((new_values - returns) ** 2).mean()
        entropy = entropies.mean()
        prediction_loss = (
            torch.stack(prediction_losses).mean()
            if prediction_losses
            else torch.zeros((), dtype=value_loss.dtype, device=value_loss.device)
        )
        loss = (
            policy_loss
            + args.value_coef * value_loss
            - args.entropy_coef * entropy
            + PREDICTION_COEF * prediction_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        with torch.no_grad():
            approx_kl = (old_log_probs - new_log_probs).mean().abs()
        final_policy = float(policy_loss.item())
        final_value = float(value_loss.item())
        final_entropy = float(entropy.item())
        final_prediction = float(prediction_loss.item())
        final_kl = float(approx_kl.item())
        epochs_done = epoch + 1
        if args.target_kl > 0.0 and final_kl > args.target_kl:
            break

    LAST_PREDICTION_LOSS = final_prediction
    return final_policy, final_value, final_entropy, final_kl, epochs_done


def _print_header(args, phases: tuple[base.core.CurriculumPhase, ...], mode: str) -> None:
    base._write("=== DMDOD / Planet Geometry PPO v0.3.8 ===")
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
        f"input={MOTION_GEOMETRY_INPUT_DIM}D motor+geometry+visible-delta | "
        f"self-supervised next-delta coef={PREDICTION_COEF:g} | chart-memory=off"
    )
    base._write(
        f"focus EMA={v036.FOCUS_EMA_ALPHA:.2f} hold={v036.FOCUS_CLEAR_SCREENS} screens"
    )
    if v035.VERBOSE_OUTPUT:
        base._write(
            "prediction target=next observed frame delta only; hidden=BPM/time/target-angle/error/direction"
        )
    else:
        base._write("compact output; use --verbose for per-BPM details")
    base._write()


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    *,
    args,
    phase_index: int,
    phase: base.core.CurriculumPhase,
    global_update: int,
    probe: base.core.Probe,
) -> None:
    _ORIGINAL_V036_SAVE_CHECKPOINT(
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
    saved["format_version"] = 17
    saved["trainer_ui_version"] = "0.3.8-predictive-motion"
    saved["input_dim"] = model.input_dim
    saved["predictive_motion_v038"] = {
        "explicit_frame_delta": True,
        "next_delta_prediction": True,
        "prediction_coef": PREDICTION_COEF,
        "last_prediction_loss": LAST_PREDICTION_LOSS,
        "max_prediction_target_norm": MAX_PREDICTION_TARGET_NORM,
        "chart_memory": False,
        "privileged_targets": False,
    }
    base.core.torch.save(saved, path)


def main() -> None:
    global PREDICTION_COEF

    coef = _pop_arg_value("--prediction-coef")
    if coef is not None:
        PREDICTION_COEF = float(coef)
        if PREDICTION_COEF < 0.0:
            raise SystemExit("--prediction-coef must be non-negative")

    _upgrade_resume_if_requested()

    base.core.GEOMETRY_INPUT_DIM = MOTION_GEOMETRY_INPUT_DIM
    base.core.RecurrentActorCritic = PredictiveRecurrentActorCritic
    base.core.warm_start = warm_start
    base.core.ppo_update = ppo_update
    v035.make_env = make_env
    v036._print_header = _print_header
    v036.save_checkpoint = save_checkpoint
    v036.main()


if __name__ == "__main__":
    main()
