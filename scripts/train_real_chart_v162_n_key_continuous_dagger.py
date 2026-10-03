from __future__ import annotations

"""v1.6.2: guarded continuous-action student-state DAgger for N-key policies.

This round is intended for policies that have already crossed the point where
continuous tanh actions outperform hard thresholding.  The student therefore
both collects DAgger states and is evaluated by the Train guard with its raw
continuous action values.  Validation is evaluated only after Train-only epoch
selection; Final is never touched.
"""

import argparse
from pathlib import Path

import torch

import train_real_chart_v080 as v080
import train_real_chart_v160_n_key_bootstrap as v160
import train_real_chart_v161_n_key_dagger as v161
from dmdod.multichart_dataset import discover_multichart_dataset
from dmdod.n_key_motor import NKeyAction, n_key_names
from dmdod.n_key_policy import NKeyRecurrentActorCritic
from dmdod.n_key_real_chart import (
    DiagnosticHudNKeyRealChartMotorEnv,
    encode_n_key_hud_real_chart_observation,
    n_key_hud_real_chart_input_dim,
)
from dmdod.n_key_training import (
    NKeyBCSequence,
    collect_n_key_dagger_sequence,
    collect_n_key_expert_sequence,
)
from dmdod.real_chart_features import DEFAULT_REAL_CHART_FEATURE_CONFIG


TRAINER_VERSION = "1.6.2-n-key-continuous-dagger"
CHECKPOINT_FORMAT_VERSION = 19


def _evaluate_continuous(
    model: NKeyRecurrentActorCritic,
    named,
    *,
    control_dt_s: float,
    physics_dt_s: float,
    device: torch.device,
):
    env = DiagnosticHudNKeyRealChartMotorEnv(
        named.segment,
        key_count=model.key_count,
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )
    observation = env.reset()
    state = model.initial_state(device)
    max_steps = int((named.segment.duration_s + 2.0) / control_dt_s) + 200

    model.eval()
    with torch.no_grad():
        for _ in range(max_steps):
            x = torch.tensor(
                encode_n_key_hud_real_chart_observation(observation),
                dtype=torch.float32,
                device=device,
            )
            mean, _, _, state = model.forward_step(x, state)
            action = NKeyAction(
                tuple(float(value.item()) for value in torch.tanh(mean))
            )
            step = env.step(action)
            observation = step.observation
            if step.done:
                break
        else:
            raise RuntimeError("N-key continuous DAgger evaluation exceeded step budget")
    return env.stats, int(env.physical_keydowns)


def _evaluate_role_continuous(
    model: NKeyRecurrentActorCritic,
    segments,
    *,
    label: str,
    control_dt_s: float,
    physics_dt_s: float,
    device: torch.device,
) -> list[tuple[object, int]]:
    results: list[tuple[object, int]] = []
    total = len(segments)
    for index, named in enumerate(segments, 1):
        stats, keydowns = _evaluate_continuous(
            model,
            named,
            control_dt_s=control_dt_s,
            physics_dt_s=physics_dt_s,
            device=device,
        )
        results.append((stats, keydowns))
        print(
            f"{label} {index:02d}/{total} {named.chart_name}: "
            f"H={stats.hits}/{stats.targets} X={stats.x_accuracy_percent:.2f}% "
            f"PP={stats.perfect_rate * 100.0:.1f}% "
            f"MAE={stats.mean_abs_error_ms if stats.mean_abs_error_ms is not None else float('nan'):.2f}ms "
            f"early={stats.too_early_presses} over={stats.overloaded} keydowns={keydowns}"
        )
    print(f"{label} aggregate: " + v161._aggregate(results))
    return results


def _checkpoint_payload(
    parent: dict,
    *,
    model_state: dict[str, torch.Tensor],
    source_checkpoint: Path,
    output_checkpoint: Path,
    round_index: int,
    completed_epoch: int,
    requested_epochs: int,
    selected_epoch: int,
    lr: float,
    expert_frames: int,
    dagger_frames: int,
    losses: list[float],
    selection_history: list[dict],
) -> dict:
    payload = dict(parent)
    payload.update(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "trainer_version": TRAINER_VERSION,
            "model_state": model_state,
            "dagger_round": int(round_index),
            "completed_dagger_epoch": int(completed_epoch),
            "requested_dagger_epochs": int(requested_epochs),
            "dagger_selected_epoch": int(selected_epoch),
            "dagger_action_mode": "continuous",
            "dagger_press_threshold": None,
            "dagger_release_threshold": None,
            "dagger_lr": float(lr),
            "dagger_expert_frames": int(expert_frames),
            "dagger_student_state_frames": int(dagger_frames),
            "dagger_loss_history": list(losses),
            "dagger_train_selection_history": list(selection_history),
            "dagger_selection_uses_validation": False,
            "dagger_source_checkpoint": str(source_checkpoint),
            "dagger_output_checkpoint": str(output_checkpoint),
            "final_used_for_selection": False,
            "finalized": False,
        }
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one continuous-action student-state N-key DAgger round, select "
            "the best Train-safe epoch, then evaluate Validation once."
        )
    )
    parser.add_argument("dataset")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dagger-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--chunk-steps", type=int, default=None)
    parser.add_argument("--anchor-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.dagger_epochs <= 0:
        raise SystemExit("--dagger-epochs must be positive")
    if args.lr <= 0.0:
        raise SystemExit("--lr must be positive")
    if args.chunk_steps is not None and args.chunk_steps <= 0:
        raise SystemExit("--chunk-steps must be positive")
    if args.anchor_limit is not None and args.anchor_limit <= 0:
        raise SystemExit("--anchor-limit must be positive")
    if args.validation_limit is not None and args.validation_limit <= 0:
        raise SystemExit("--validation-limit must be positive")

    device = v161._device_from_arg(args.device)
    source_checkpoint = Path(args.checkpoint)
    parent = torch.load(source_checkpoint, map_location=device, weights_only=False)

    key_count = int(parent["key_count"])
    input_dim = int(parent["input_dim"])
    expected_input = n_key_hud_real_chart_input_dim(key_count)
    if input_dim != expected_input:
        raise SystemExit(
            f"checkpoint input_dim={input_dim} does not match {key_count}K expected {expected_input}"
        )
    key_names = n_key_names(key_count)

    model = NKeyRecurrentActorCritic(
        input_dim=input_dim,
        key_count=key_count,
        hidden_dim=int(parent["hidden_dim"]),
    ).to(device)
    model.load_state_dict(parent["model_state"])
    model.gru.flatten_parameters()

    calibration = parent.get("calibration") or {}
    if "lead_s" not in calibration:
        raise SystemExit("checkpoint calibration is missing lead_s")
    lead_s = float(calibration["lead_s"])
    control_dt_s = float(parent["control_dt"])
    physics_dt_s = float(parent.get("physics_dt", 0.001))
    chunk_steps = int(args.chunk_steps or parent.get("chunk_steps", 192))

    dataset = discover_multichart_dataset(args.dataset)
    train_charts = v080._compile_role(dataset.train)
    validation_charts = v080._compile_role(dataset.validation)
    anchors = v080._build_anchor_segments(
        train_charts,
        window_s=float(parent["train_window"]),
        anchors_per_chart=int(parent["anchors_per_chart"]),
    )
    validation = v080._build_validation_segments(
        validation_charts,
        window_s=float(parent["validation_window"]),
    )

    parent_anchor_limit = parent.get("anchor_limit")
    anchor_limit = args.anchor_limit if args.anchor_limit is not None else parent_anchor_limit
    if anchor_limit is not None:
        anchors = anchors[: int(anchor_limit)]
    parent_validation_limit = parent.get("validation_limit")
    validation_limit = (
        args.validation_limit
        if args.validation_limit is not None
        else parent_validation_limit
    )
    if validation_limit is not None:
        validation = validation[: int(validation_limit)]
    if not anchors:
        raise SystemExit("no Train anchors selected")

    round_index = int(parent.get("dagger_round", 0)) + 1
    output_checkpoint = Path(
        args.output or v161._default_output_path(source_checkpoint, round_index)
    )

    print("=== DMDOD v1.6.2 N-Key Continuous DAgger ===")
    print(
        f"source={source_checkpoint} output={output_checkpoint} round={round_index} "
        f"keys={key_count} input={input_dim}D device={device}"
    )
    print("key-order: " + ",".join(key_names))
    print(
        f"anchors={len(anchors)} validation={len(validation)} dagger-epochs={args.dagger_epochs} "
        "action=continuous | FINAL untouched"
    )
    print(
        "Train epoch guard: safe->overload rejected; per-anchor hit floor uses "
        "the mature real-chart tolerance. Validation is not used for selection."
    )

    print("=== pre-DAgger continuous Train / epoch 0 fallback ===")
    pre_results = _evaluate_role_continuous(
        model,
        anchors,
        label="pre-train",
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
        device=device,
    )
    best_state = v161._clone_model_state(model)
    best_results = pre_results
    best_epoch = 0
    selection_history = [
        v161._selection_record(
            epoch=0,
            loss=None,
            accepted=True,
            selected=True,
            reasons=(),
            results=pre_results,
        )
    ]

    expert_sequences: list[NKeyBCSequence] = []
    dagger_sequences: list[NKeyBCSequence] = []
    for index, named in enumerate(anchors, 1):
        expert = collect_n_key_expert_sequence(
            named.segment,
            key_count=key_count,
            lead_s=lead_s,
            control_dt_s=control_dt_s,
            physics_dt_s=physics_dt_s,
            device=device,
            source=f"dagger{round_index}-expert-{index}-{named.chart_name}",
        )
        expert_sequences.append(expert.sequence)

        rollout = collect_n_key_dagger_sequence(
            model,
            named.segment,
            lead_s=lead_s,
            press_threshold=0.25,
            release_threshold=-0.45,
            control_dt_s=control_dt_s,
            physics_dt_s=physics_dt_s,
            device=device,
            source=f"dagger{round_index}-continuous-student-{index}-{named.chart_name}",
            action_mode="continuous",
        )
        dagger_sequences.append(rollout.sequence)
        print(
            f"collect {index:02d}/{len(anchors)} {named.chart_name}: "
            f"frames={rollout.sequence.frames} H={rollout.stats.hits}/{rollout.stats.targets} "
            f"X={rollout.stats.x_accuracy_percent:.2f}% early={rollout.stats.too_early_presses} "
            f"over={rollout.stats.overloaded} keydowns={rollout.physical_keydowns}"
        )

    training_sequences = [*expert_sequences, *dagger_sequences]
    expert_frames = sum(sequence.frames for sequence in expert_sequences)
    dagger_frames = sum(sequence.frames for sequence in dagger_sequences)
    print(
        f"aggregate-data expert={expert_frames} student-state={dagger_frames} "
        f"total={expert_frames + dagger_frames} frames"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    losses: list[float] = []
    for epoch in range(1, args.dagger_epochs + 1):
        loss = v160._train_bc_epoch(
            model,
            training_sequences,
            optimizer=optimizer,
            chunk_steps=chunk_steps,
        )
        losses.append(loss)
        print(f"dagger-train {epoch:03d}/{args.dagger_epochs} loss={loss:.6f}")
        print(f"=== Train guard epoch {epoch}/{args.dagger_epochs} ===")
        candidate_results = _evaluate_role_continuous(
            model,
            anchors,
            label=f"epoch-{epoch:03d}-train",
            control_dt_s=control_dt_s,
            physics_dt_s=physics_dt_s,
            device=device,
        )
        accepted, reasons = v161._train_safety_guard(pre_results, candidate_results)
        selected = accepted and v161._selection_key(candidate_results) > v161._selection_key(best_results)
        if not accepted:
            print("epoch-selection: REJECT " + "; ".join(reasons))
        elif selected:
            best_state = v161._clone_model_state(model)
            best_results = candidate_results
            best_epoch = epoch
            print(
                f"epoch-selection: ACCEPT new-best epoch={epoch} "
                + v161._aggregate(candidate_results)
            )
        else:
            print(
                f"epoch-selection: SAFE not-best epoch={epoch} "
                + v161._aggregate(candidate_results)
            )
        selection_history.append(
            v161._selection_record(
                epoch=epoch,
                loss=loss,
                accepted=accepted,
                selected=selected,
                reasons=reasons,
                results=candidate_results,
            )
        )

    model.load_state_dict(best_state)
    model.gru.flatten_parameters()
    print("=== selected Train-safe continuous DAgger checkpoint ===")
    print(
        f"selected epoch={best_epoch}/{args.dagger_epochs}: "
        + v161._aggregate(best_results)
    )

    v161._save_checkpoint(
        output_checkpoint,
        _checkpoint_payload(
            parent,
            model_state=best_state,
            source_checkpoint=source_checkpoint,
            output_checkpoint=output_checkpoint,
            round_index=round_index,
            completed_epoch=args.dagger_epochs,
            requested_epochs=args.dagger_epochs,
            selected_epoch=best_epoch,
            lr=args.lr,
            expert_frames=expert_frames,
            dagger_frames=dagger_frames,
            losses=losses,
            selection_history=selection_history,
        ),
    )

    print("=== selected checkpoint continuous Validation ===")
    _evaluate_role_continuous(
        model,
        validation,
        label="validation",
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
        device=device,
    )
    print(f"checkpoint final: {output_checkpoint}")


if __name__ == "__main__":
    main()
