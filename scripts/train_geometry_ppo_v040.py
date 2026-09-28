from __future__ import annotations

import copy
from pathlib import Path

try:
    from . import train_geometry_ppo_v039 as v039
except ImportError:  # direct script execution
    import train_geometry_ppo_v039 as v039


v038 = v039.v038
v036 = v039.v036
v035 = v039.v035
base = v039.base

# v0.4.0 protects a good imitation/bootstrap policy while PPO adapts it to
# wider BPM bands.  v0.3.9 showed that PPO could trade a tiny clear-rate gain
# for a very large XAcc/Perfect regression, then keep the damaged policy as the
# phase best.  This frontend changes only fine-tuning/selection, not game rules.
ANCHOR_COEF_START = 0.50
ANCHOR_DECAY_UPDATES = 16
PRECISION_ROLLBACK_XACC_DROP = 4.0
PRECISION_ROLLBACK_MIN_XACC_DROP = 5.0
PRECISION_ROLLBACK_PP_DROP = 0.10
PRECISION_ROLLBACK_MIN_PP_DROP = 0.15

_ORIGINAL_V036_EVALUATE = v036._evaluate
_ORIGINAL_V039_HEADER = v039._print_header
_ORIGINAL_V039_SAVE_CHECKPOINT = v039.save_checkpoint

_PHASE_KEY: tuple[object, ...] | None = None
_GUARD_BEST_PROBE = None
_GUARD_BEST_MODEL_STATE = None
_GUARD_BEST_OPTIMIZER_STATE = None
_LAST_OPTIMIZER_STATE = None
_ANCHOR_MODEL = None
_ROLLBACK_PENDING = False
_PHASE_PPO_UPDATES = 0
_LAST_ANCHOR_LOSS = 0.0
_LAST_ANCHOR_COEF = 0.0


def _precision_deficits(probe) -> tuple[float, float, float, float]:
    return (
        max(0.0, (base.PRECISION_MIN_BPM_XACC - base._min_xacc(probe)) / 100.0),
        max(0.0, (base.PRECISION_OVERALL_XACC - base._xacc(probe)) / 100.0),
        max(0.0, base.PRECISION_MIN_BPM_PP - base._min_pp(probe)),
        max(0.0, base.PRECISION_OVERALL_PP - base._pp(probe)),
    )


def _completion_deficits(probe, retention) -> tuple[float, ...]:
    return (
        max(0.0, base.COMPLETION_MIN_BPM_HIT - base._min_hit(probe)),
        max(0.0, base.COMPLETION_MIN_BPM_FULL - base._min_full(probe)),
        max(0.0, 1.0 - retention.min_hit_rate),
        max(0.0, 1.0 - retention.min_full_rate),
    )


def rank_key(probe, retention) -> tuple[float, ...]:
    """Rank by distance to *all* gates instead of clear-rate lexicography.

    A one-hit improvement must not outrank a large collapse in Perfect/XAcc.
    The worst normalized gate deficit is minimized first, then the sum of all
    deficits.  Raw precision and completion metrics are only tie breakers.
    """

    deficits = _completion_deficits(probe, retention) + _precision_deficits(probe)
    worst = max(deficits, default=0.0)
    total = sum(deficits)
    mae = probe.mae_ms
    bias = probe.max_abs_bpm_bias_ms
    return (
        1.0 if retention.overloads == 0 and probe.overloads == 0 else 0.0,
        -float(retention.overloads + probe.overloads),
        -worst,
        -total,
        base._min_xacc(probe),
        base._xacc(probe),
        base._min_pp(probe),
        base._pp(probe),
        base._min_hit(probe),
        base._min_full(probe),
        probe.hit_rate,
        probe.full_rate,
        -(bias if bias is not None else float("inf")),
        -(mae if mae is not None else float("inf")),
    )


def precision_regressed(probe, best_probe) -> bool:
    """Return True when PPO damaged timing quality beyond evaluation noise."""

    return (
        base._xacc(probe) < base._xacc(best_probe) - PRECISION_ROLLBACK_XACC_DROP
        or base._min_xacc(probe)
        < base._min_xacc(best_probe) - PRECISION_ROLLBACK_MIN_XACC_DROP
        or base._pp(probe) < base._pp(best_probe) - PRECISION_ROLLBACK_PP_DROP
        or base._min_pp(probe)
        < base._min_pp(best_probe) - PRECISION_ROLLBACK_MIN_PP_DROP
    )


def _set_anchor_from_model(model) -> None:
    global _ANCHOR_MODEL
    _ANCHOR_MODEL = copy.deepcopy(model)
    _ANCHOR_MODEL.eval()
    for parameter in _ANCHOR_MODEL.parameters():
        parameter.requires_grad_(False)


def _reset_phase_guard(model, phase) -> None:
    global _PHASE_KEY, _GUARD_BEST_PROBE, _GUARD_BEST_MODEL_STATE
    global _GUARD_BEST_OPTIMIZER_STATE, _LAST_OPTIMIZER_STATE
    global _ROLLBACK_PENDING, _PHASE_PPO_UPDATES

    _PHASE_KEY = v036._phase_identity(phase)
    _GUARD_BEST_PROBE = None
    _GUARD_BEST_MODEL_STATE = copy.deepcopy(model.state_dict())
    _GUARD_BEST_OPTIMIZER_STATE = None
    _LAST_OPTIMIZER_STATE = None
    _ROLLBACK_PENDING = False
    _PHASE_PPO_UPDATES = 0
    _set_anchor_from_model(model)


def _evaluate(
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
    """Evaluate, update the best-policy anchor, and undo precision collapses."""

    global _GUARD_BEST_PROBE, _GUARD_BEST_MODEL_STATE
    global _GUARD_BEST_OPTIMIZER_STATE, _ROLLBACK_PENDING

    phase_key = v036._phase_identity(phase)
    if phase_key != _PHASE_KEY:
        _reset_phase_guard(model, phase)

    probe, retention = _ORIGINAL_V036_EVALUATE(
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

    # Full verification/final probes are measurements, not training screens.
    is_training_screen = label.startswith("screen ")
    if not is_training_screen:
        return probe, retention

    if _GUARD_BEST_PROBE is None:
        _GUARD_BEST_PROBE = probe
        _GUARD_BEST_MODEL_STATE = copy.deepcopy(model.state_dict())
        if _LAST_OPTIMIZER_STATE is not None:
            _GUARD_BEST_OPTIMIZER_STATE = copy.deepcopy(_LAST_OPTIMIZER_STATE)
        _set_anchor_from_model(model)
        return probe, retention

    if label != "screen 00" and precision_regressed(probe, _GUARD_BEST_PROBE):
        if _GUARD_BEST_MODEL_STATE is not None:
            model.load_state_dict(_GUARD_BEST_MODEL_STATE)
            _set_anchor_from_model(model)
            _ROLLBACK_PENDING = True
            base._write(
                "  precision rollback -> "
                f"X={base._xacc(_GUARD_BEST_PROBE):.2f}% "
                f"PP={base._pp(_GUARD_BEST_PROBE):.1%} "
                f"minX={base._min_xacc(_GUARD_BEST_PROBE):.1f}% "
                f"minPP={base._min_pp(_GUARD_BEST_PROBE):.0%}"
            )
        return probe, retention

    # Keep the behavioral anchor synchronized with the same ranking used by the
    # base trainer's best-model selection.
    if rank_key(probe, retention) > rank_key(_GUARD_BEST_PROBE, retention):
        _GUARD_BEST_PROBE = probe
        _GUARD_BEST_MODEL_STATE = copy.deepcopy(model.state_dict())
        if _LAST_OPTIMIZER_STATE is not None:
            _GUARD_BEST_OPTIMIZER_STATE = copy.deepcopy(_LAST_OPTIMIZER_STATE)
        _set_anchor_from_model(model)

    return probe, retention


def _sequence_actions(model, observations):
    torch = base.core.torch
    state = model.initial_state(observations.device)
    actions = []
    for x in observations:
        mean, _, _, state = model.forward_step(x, state)
        actions.append(torch.tanh(mean))
    return torch.stack(actions)


def _anchor_coef() -> float:
    if ANCHOR_DECAY_UPDATES <= 0:
        return 0.0
    progress = min(1.0, _PHASE_PPO_UPDATES / float(ANCHOR_DECAY_UPDATES))
    return ANCHOR_COEF_START * (1.0 - progress)


def ppo_update(model, optimizer, rollouts, args):
    """v0.3.8 PPO + prediction loss + decaying best-policy behavior anchor."""

    global _LAST_OPTIMIZER_STATE, _ROLLBACK_PENDING, _PHASE_PPO_UPDATES
    global _LAST_ANCHOR_LOSS, _LAST_ANCHOR_COEF

    torch = base.core.torch
    nn = base.core.nn

    if _ROLLBACK_PENDING:
        if _GUARD_BEST_OPTIMIZER_STATE is not None:
            optimizer.load_state_dict(copy.deepcopy(_GUARD_BEST_OPTIMIZER_STATE))
        for group in optimizer.param_groups:
            group["lr"] = max(
                args.min_lr,
                float(group["lr"]) * args.rollback_lr_factor,
            )
        _ROLLBACK_PENDING = False

    _PHASE_PPO_UPDATES += 1
    anchor_coef = _anchor_coef()
    anchor = _ANCHOR_MODEL

    all_advantages = torch.cat([rollout.advantages for rollout in rollouts])
    adv_mean = all_advantages.mean()
    adv_std = all_advantages.std(unbiased=False).clamp_min(1e-6)
    old_log_probs = torch.cat([rollout.old_log_probs for rollout in rollouts])
    returns = torch.cat([rollout.returns for rollout in rollouts])

    anchor_targets = []
    if anchor is not None and anchor_coef > 0.0:
        with torch.no_grad():
            for rollout in rollouts:
                anchor_targets.append(_sequence_actions(anchor, rollout.observations))

    final_policy = final_value = final_entropy = final_kl = 0.0
    final_prediction = final_anchor = 0.0
    epochs_done = 0
    model.train()

    for _ in range(args.ppo_epochs):
        new_log_probs_list = []
        new_values_list = []
        entropies_list = []
        normalized_advantages = []
        prediction_losses = []
        anchor_losses = []

        for rollout_index, rollout in enumerate(rollouts):
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
                target = rollout.observations[
                    1:, v038.MOTION_FEATURE_START : v038.MOTION_FEATURE_END
                ].detach()
                prediction = predictions[:-1]
                valid = (
                    torch.linalg.vector_norm(target, dim=1)
                    <= v038.MAX_PREDICTION_TARGET_NORM
                )
                if bool(valid.any()):
                    prediction_losses.append(
                        ((prediction[valid] - target[valid]) ** 2).mean()
                    )

            if anchor_targets:
                current_actions = _sequence_actions(model, rollout.observations)
                anchor_losses.append(
                    ((current_actions - anchor_targets[rollout_index]) ** 2).mean()
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
        anchor_loss = (
            torch.stack(anchor_losses).mean()
            if anchor_losses
            else torch.zeros((), dtype=value_loss.dtype, device=value_loss.device)
        )
        loss = (
            policy_loss
            + args.value_coef * value_loss
            - args.entropy_coef * entropy
            + v038.PREDICTION_COEF * prediction_loss
            + anchor_coef * anchor_loss
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
        final_anchor = float(anchor_loss.item())
        final_kl = float(approx_kl.item())
        epochs_done += 1
        if args.target_kl > 0.0 and final_kl > args.target_kl:
            break

    v038.LAST_PREDICTION_LOSS = final_prediction
    _LAST_ANCHOR_LOSS = final_anchor
    _LAST_ANCHOR_COEF = anchor_coef
    _LAST_OPTIMIZER_STATE = copy.deepcopy(optimizer.state_dict())
    return final_policy, final_value, final_entropy, final_kl, epochs_done


def _print_header(args, phases, mode: str) -> None:
    _ORIGINAL_V039_HEADER(args, phases, mode)
    base._write(
        f"v0.4.0 precision guard: rank=max/sum gate deficit; "
        f"rollback X-{PRECISION_ROLLBACK_XACC_DROP:g} minX-{PRECISION_ROLLBACK_MIN_XACC_DROP:g} "
        f"PP-{PRECISION_ROLLBACK_PP_DROP:.0%} minPP-{PRECISION_ROLLBACK_MIN_PP_DROP:.0%}; "
        f"anchor={ANCHOR_COEF_START:g}->0/{ANCHOR_DECAY_UPDATES}upd"
    )
    base._write()


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    *,
    args,
    phase_index: int,
    phase,
    global_update: int,
    probe,
) -> None:
    _ORIGINAL_V039_SAVE_CHECKPOINT(
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
    saved["format_version"] = 19
    saved["trainer_ui_version"] = "0.4.0-precision-guard"
    saved["precision_guard_v040"] = {
        "rank": "minimize worst then total normalized completion+precision gate deficit",
        "xacc_drop_points": PRECISION_ROLLBACK_XACC_DROP,
        "min_xacc_drop_points": PRECISION_ROLLBACK_MIN_XACC_DROP,
        "perfect_drop": PRECISION_ROLLBACK_PP_DROP,
        "min_perfect_drop": PRECISION_ROLLBACK_MIN_PP_DROP,
        "behavior_anchor_start": ANCHOR_COEF_START,
        "behavior_anchor_decay_updates": ANCHOR_DECAY_UPDATES,
        "last_behavior_anchor_loss": _LAST_ANCHOR_LOSS,
        "last_behavior_anchor_coef": _LAST_ANCHOR_COEF,
    }
    base.core.torch.save(saved, path)


def main() -> None:
    # Patch module globals that the inherited frontends resolve at runtime.
    base.rank_key = rank_key
    v036._evaluate = _evaluate
    v038.ppo_update = ppo_update
    v039._print_header = _print_header
    v039.save_checkpoint = save_checkpoint
    v039.main()


if __name__ == "__main__":
    main()
