from __future__ import annotations

"""v1.9.1: wall-clock-budgeted adaptive actor action-trust DAgger.

This is the unattended runner for v1.9.  A rejected Train guard does not end
the whole job: the candidate is rolled back to the last accepted policy, the
action-space trust radius is reduced, and the same accepted-policy feature
cache is retried.  Safe candidates advance DAgger and refresh student-state
data.  The best Train-safe model is checkpointed whenever it improves.

Selection remains Train-only.  Validation runs once after the wall-clock
training budget (with a configurable reserve) and Final remains untouched.
"""

import argparse
from pathlib import Path
import time

import torch

import train_real_chart_v080 as v080
import train_real_chart_v161_n_key_dagger as v161
import train_real_chart_v162_n_key_continuous_dagger as v162
import train_real_chart_v171_n_key_connectome_failure_continuation_dagger as v171
import train_real_chart_v181_n_key_connectome_sgd_microstep_dagger as v181
import train_real_chart_v190_n_key_connectome_actor_action_trust_dagger as v190
from dmdod.fly_connectome_policy import (
    N_KEY_POLICY_BACKEND_FLY_CONNECTOME,
)
from dmdod.multichart_dataset import discover_multichart_dataset
from dmdod.n_key_motor import n_key_names
from dmdod.n_key_real_chart import n_key_hud_real_chart_input_dim
from dmdod.n_key_training import NKeyBCSequence, collect_n_key_expert_sequence
from dmdod.random_connectome_policy import (
    N_KEY_POLICY_BACKEND_RANDOM_CONNECTOME,
)


TRAINER_VERSION = "1.9.1-n-key-connectome-budgeted-action-trust-dagger"
CHECKPOINT_FORMAT_VERSION = 32
DEFAULT_HOURS = 8.0
DEFAULT_RESERVE_MINUTES = 8.0
DEFAULT_MAX_TRIALS = 10000
DEFAULT_INITIAL_ACTION_RMS = 0.01
DEFAULT_MIN_ACTION_RMS = 0.00001
DEFAULT_REJECT_SHRINK = 0.5
DEFAULT_SAFE_GROW = 1.25


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _budget_payload(
    parent: dict,
    *,
    model,
    model_state: dict[str, torch.Tensor],
    source_checkpoint: Path,
    output_checkpoint: Path,
    round_index: int,
    requested_hours: float,
    elapsed_seconds: float,
    trial_count: int,
    accepted_steps: int,
    selected_step: int,
    stopped_reason: str,
    actor_steps: int,
    lr: float,
    stay_coef: float,
    max_action_rms: float,
    current_action_rms: float,
    expert_frames: int,
    dagger_frames: int,
    student_frame_history: list[int],
    history: list[dict],
) -> dict:
    payload = v190._checkpoint_payload(
        parent,
        model=model,
        model_state=model_state,
        source_checkpoint=source_checkpoint,
        output_checkpoint=output_checkpoint,
        round_index=round_index,
        requested_steps=max(1, trial_count),
        attempted_steps=trial_count,
        accepted_steps=accepted_steps,
        selected_step=selected_step,
        stopped_unsafe=False,
        actor_steps=actor_steps,
        lr=lr,
        stay_coef=stay_coef,
        max_action_rms=max_action_rms,
        expert_frames=expert_frames,
        dagger_frames=dagger_frames,
        student_frame_history=student_frame_history,
        history=history,
    )
    payload.update(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "trainer_version": TRAINER_VERSION,
            "budget_requested_hours": float(requested_hours),
            "budget_elapsed_seconds": float(elapsed_seconds),
            "budget_trial_count": int(trial_count),
            "budget_accepted_steps": int(accepted_steps),
            "budget_selected_step": int(selected_step),
            "budget_stopped_reason": str(stopped_reason),
            "budget_initial_action_rms": float(max_action_rms),
            "budget_current_action_rms": float(current_action_rms),
            "budget_history": list(history),
        }
    )
    return payload


def _save_best(
    output_checkpoint: Path,
    parent: dict,
    *,
    model,
    best_state: dict[str, torch.Tensor],
    source_checkpoint: Path,
    round_index: int,
    requested_hours: float,
    start_time: float,
    trial_count: int,
    accepted_steps: int,
    best_step: int,
    stopped_reason: str,
    actor_steps: int,
    lr: float,
    stay_coef: float,
    initial_action_rms: float,
    current_action_rms: float,
    expert_frames: int,
    dagger_frames: int,
    student_frame_history: list[int],
    history: list[dict],
) -> None:
    v161._save_checkpoint(
        output_checkpoint,
        _budget_payload(
            parent,
            model=model,
            model_state=best_state,
            source_checkpoint=source_checkpoint,
            output_checkpoint=output_checkpoint,
            round_index=round_index,
            requested_hours=requested_hours,
            elapsed_seconds=time.monotonic() - start_time,
            trial_count=trial_count,
            accepted_steps=accepted_steps,
            selected_step=best_step,
            stopped_reason=stopped_reason,
            actor_steps=actor_steps,
            lr=lr,
            stay_coef=stay_coef,
            max_action_rms=initial_action_rms,
            current_action_rms=current_action_rms,
            expert_frames=expert_frames,
            dagger_frames=dagger_frames,
            student_frame_history=student_frame_history,
            history=history,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run unattended actor-only action-trust DAgger for a wall-clock budget. "
            "Rejected candidates roll back and retry with a smaller action trust radius."
        )
    )
    parser.add_argument("dataset")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--hours", type=float, default=DEFAULT_HOURS)
    parser.add_argument("--reserve-minutes", type=float, default=DEFAULT_RESERVE_MINUTES)
    parser.add_argument("--max-trials", type=int, default=DEFAULT_MAX_TRIALS)
    parser.add_argument("--actor-steps", type=int, default=v190.DEFAULT_ACTOR_STEPS)
    parser.add_argument("--lr", type=float, default=v190.DEFAULT_LR)
    parser.add_argument("--stay-coef", type=float, default=v190.DEFAULT_STAY_COEF)
    parser.add_argument(
        "--initial-action-rms",
        type=float,
        default=DEFAULT_INITIAL_ACTION_RMS,
    )
    parser.add_argument(
        "--min-action-rms",
        type=float,
        default=DEFAULT_MIN_ACTION_RMS,
    )
    parser.add_argument("--reject-shrink", type=float, default=DEFAULT_REJECT_SHRINK)
    parser.add_argument("--safe-grow", type=float, default=DEFAULT_SAFE_GROW)
    parser.add_argument("--lr-backoffs", type=int, default=v190.DEFAULT_LR_BACKOFFS)
    parser.add_argument("--min-lr", type=float, default=v190.DEFAULT_MIN_LR)
    parser.add_argument("--chunk-steps", type=int, default=None)
    parser.add_argument("--anchor-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.hours <= 0.0:
        raise SystemExit("--hours must be positive")
    if args.reserve_minutes < 0.0:
        raise SystemExit("--reserve-minutes must be non-negative")
    if args.max_trials <= 0:
        raise SystemExit("--max-trials must be positive")
    if args.actor_steps <= 0:
        raise SystemExit("--actor-steps must be positive")
    if args.lr <= 0.0 or args.min_lr <= 0.0:
        raise SystemExit("--lr/--min-lr must be positive")
    if args.stay_coef < 0.0:
        raise SystemExit("--stay-coef must be non-negative")
    if args.initial_action_rms <= 0.0 or args.min_action_rms <= 0.0:
        raise SystemExit("action RMS bounds must be positive")
    if args.min_action_rms > args.initial_action_rms:
        raise SystemExit("--min-action-rms cannot exceed --initial-action-rms")
    if not (0.0 < args.reject_shrink < 1.0):
        raise SystemExit("--reject-shrink must be in (0, 1)")
    if args.safe_grow < 1.0:
        raise SystemExit("--safe-grow must be >= 1")
    if args.lr_backoffs < 0:
        raise SystemExit("--lr-backoffs must be non-negative")
    if args.chunk_steps is not None and args.chunk_steps <= 0:
        raise SystemExit("--chunk-steps must be positive")

    total_budget_s = float(args.hours) * 3600.0
    reserve_s = float(args.reserve_minutes) * 60.0
    if reserve_s >= total_budget_s:
        raise SystemExit("reserve must be smaller than the total wall-clock budget")
    train_budget_s = total_budget_s - reserve_s
    start_time = time.monotonic()

    device = v161._device_from_arg(args.device)
    source_checkpoint = Path(args.checkpoint)
    output_checkpoint = Path(args.output)
    parent = torch.load(source_checkpoint, map_location=device, weights_only=False)
    backend = str(parent.get("n_key_policy_backend", "gru"))
    if backend not in {
        N_KEY_POLICY_BACKEND_FLY_CONNECTOME,
        N_KEY_POLICY_BACKEND_RANDOM_CONNECTOME,
    }:
        raise SystemExit(
            "v1.9.1 budget trainer requires fly_connectome or random_connectome"
        )

    key_count = int(parent["key_count"])
    input_dim = int(parent["input_dim"])
    expected_input = n_key_hud_real_chart_input_dim(key_count)
    if input_dim != expected_input:
        raise SystemExit(
            f"checkpoint input_dim={input_dim} does not match {key_count}K expected "
            f"{expected_input}"
        )

    model = v171._build_policy_from_checkpoint(parent, device=device)
    actor_parameters = v190._freeze_actor_only(model)
    key_names = n_key_names(key_count)

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

    anchor_limit = (
        args.anchor_limit
        if args.anchor_limit is not None
        else parent.get("anchor_limit")
    )
    if anchor_limit is not None:
        anchors = anchors[: int(anchor_limit)]
    validation_limit = (
        args.validation_limit
        if args.validation_limit is not None
        else parent.get("validation_limit")
    )
    if validation_limit is not None:
        validation = validation[: int(validation_limit)]
    if not anchors:
        raise SystemExit("no Train anchors selected")

    round_index = int(parent.get("dagger_round", 0)) + 1

    print("=== DMDOD v1.9.1 Budgeted Actor Action-Trust DAgger ===")
    print(
        f"source={source_checkpoint} output={output_checkpoint} round={round_index} "
        f"backend={model.backend_name} keys={key_count} input={input_dim}D device={device}"
    )
    print("key-order: " + ",".join(key_names))
    print(
        f"budget={args.hours:g}h reserve={args.reserve_minutes:g}m "
        f"anchors={len(anchors)} validation={len(validation)} actor-steps={args.actor_steps} "
        f"lr={args.lr:g} stay={args.stay_coef:g} "
        f"action-rms={args.initial_action_rms:g}->{args.min_action_rms:g} "
        f"reject-shrink={args.reject_shrink:g} safe-grow={args.safe_grow:g}"
    )
    print(
        "Rejected candidates rollback and retry from the same accepted policy/data with "
        "a smaller action-space trust radius. Best Train-safe checkpoint is saved on improvement."
    )

    print("=== pre-budget continuous Train / fixed safety baseline ===")
    initial_results = v162._evaluate_role_continuous(
        model,
        anchors,
        label="pre-train",
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
        device=device,
    )
    safety_reference_results = initial_results
    continuation_results = initial_results
    continuation_state = v161._clone_model_state(model)
    best_state = continuation_state
    best_results = initial_results
    best_step = 0

    expert_sequences: list[NKeyBCSequence] = []
    for index, named in enumerate(anchors, 1):
        expert = collect_n_key_expert_sequence(
            named.segment,
            key_count=key_count,
            lead_s=lead_s,
            control_dt_s=control_dt_s,
            physics_dt_s=physics_dt_s,
            device=device,
            source=f"dagger{round_index}-budget-expert-{index}-{named.chart_name}",
        )
        expert_sequences.append(expert.sequence)
    expert_frames = sum(sequence.frames for sequence in expert_sequences)

    dagger_sequences, dagger_frames = v181._collect_student_state_sequences(
        model,
        anchors,
        round_index=round_index,
        collection_index=0,
        lead_s=lead_s,
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
        device=device,
    )
    student_frame_history = [dagger_frames]
    training_sequences = [*expert_sequences, *dagger_sequences]
    print(
        f"aggregate-data generation=0 expert={expert_frames} student-state={dagger_frames} "
        f"total={expert_frames + dagger_frames} frames"
    )

    history: list[dict] = []
    trial_count = 0
    accepted_steps = 0
    current_action_rms = float(args.initial_action_rms)
    stopped_reason = "time-budget"

    cached = None
    try:
        while trial_count < args.max_trials:
            elapsed = time.monotonic() - start_time
            remaining_train = train_budget_s - elapsed
            if remaining_train <= 0.0:
                stopped_reason = "time-budget"
                break
            if current_action_rms < args.min_action_rms:
                # Defensive clamp for old/externally resumed state. Normal
                # shrink paths clamp before returning to the top of the loop.
                current_action_rms = float(args.min_action_rms)

            if cached is None:
                model.load_state_dict(continuation_state)
                model.prepare_recurrent_runtime()
                v190._freeze_actor_only(model)
                print(
                    f"feature-cache: build generation={accepted_steps} "
                    f"remaining={_format_duration(remaining_train)}"
                )
                cached = v190._build_action_trust_sequences(model, training_sequences)

            trial_count += 1
            model.load_state_dict(continuation_state)
            model.prepare_recurrent_runtime()
            actor_parameters = v190._freeze_actor_only(model)
            optimizer = torch.optim.SGD(actor_parameters, lr=args.lr)

            print(
                f"=== budget trial {trial_count} accepted={accepted_steps} "
                f"radius={current_action_rms:.6g} "
                f"elapsed={_format_duration(time.monotonic() - start_time)} "
                f"remaining={_format_duration(train_budget_s - (time.monotonic() - start_time))} ==="
            )

            train_info = v190._train_actor_action_trust(
                model,
                cached,
                optimizer=optimizer,
                actor_steps=args.actor_steps,
                chunk_steps=chunk_steps,
                base_lr=args.lr,
                stay_coef=args.stay_coef,
                max_action_rms=current_action_rms,
                lr_backoffs=args.lr_backoffs,
                min_lr=args.min_lr,
            )

            if int(train_info["accepted_inner_steps"]) == 0:
                history.append(
                    {
                        "trial": int(trial_count),
                        "accepted_step": int(accepted_steps),
                        "radius": float(current_action_rms),
                        **train_info,
                        "guard_accepted": False,
                        "guard_reasons": ("action-trust-no-op",),
                    }
                )
                if current_action_rms <= args.min_action_rms:
                    stopped_reason = "action-rms-floor"
                    print(
                        f"budget stop: action-trust no-op at radius floor "
                        f"{args.min_action_rms:.6g}"
                    )
                    break
                current_action_rms = max(
                    float(args.min_action_rms),
                    current_action_rms * float(args.reject_shrink),
                )
                print(
                    f"budget retry: action-trust no-op; radius -> {current_action_rms:.6g}"
                )
                continue

            candidate_state = v161._clone_model_state(model)
            candidate_results = v162._evaluate_role_continuous(
                model,
                anchors,
                label=f"budget-{trial_count:04d}",
                control_dt_s=control_dt_s,
                physics_dt_s=physics_dt_s,
                device=device,
            )
            safe, reasons = v161._train_safety_guard(
                safety_reference_results,
                candidate_results,
            )
            selected_best = safe and (
                v161._selection_key(candidate_results)
                > v161._selection_key(best_results)
            )
            summary = v161._summarize(candidate_results)
            history.append(
                {
                    "trial": int(trial_count),
                    "accepted_step": int(accepted_steps),
                    "radius": float(current_action_rms),
                    **train_info,
                    "guard_accepted": bool(safe),
                    "guard_reasons": tuple(reasons),
                    "selected_best_when_evaluated": bool(selected_best),
                    "hits": int(summary.hits),
                    "targets": int(summary.targets),
                    "x_accuracy_percent": float(summary.x_accuracy_percent),
                    "early": int(summary.early),
                    "overloaded": bool(summary.overloaded),
                    "keydowns": int(summary.keydowns),
                }
            )

            status = "SAFE+BEST" if selected_best else "SAFE" if safe else "REJECT"
            detail = "" if safe else " " + "; ".join(reasons)
            print(
                f"budget guard={status}: {v161._aggregate(candidate_results)}{detail}"
            )

            if not safe:
                model.load_state_dict(continuation_state)
                model.prepare_recurrent_runtime()
                if current_action_rms <= args.min_action_rms:
                    stopped_reason = "action-rms-floor"
                    print(
                        f"budget stop: rejected at radius floor "
                        f"{args.min_action_rms:.6g}"
                    )
                    break
                current_action_rms = max(
                    float(args.min_action_rms),
                    current_action_rms * float(args.reject_shrink),
                )
                print(
                    f"budget retry: rollback accepted={accepted_steps}; "
                    f"radius -> {current_action_rms:.6g}"
                )
                continue

            continuation_state = candidate_state
            continuation_results = candidate_results
            accepted_steps += 1

            if selected_best:
                best_state = candidate_state
                best_results = candidate_results
                best_step = accepted_steps
                _save_best(
                    output_checkpoint,
                    parent,
                    model=model,
                    best_state=best_state,
                    source_checkpoint=source_checkpoint,
                    round_index=round_index,
                    requested_hours=args.hours,
                    start_time=start_time,
                    trial_count=trial_count,
                    accepted_steps=accepted_steps,
                    best_step=best_step,
                    stopped_reason="running",
                    actor_steps=args.actor_steps,
                    lr=args.lr,
                    stay_coef=args.stay_coef,
                    initial_action_rms=args.initial_action_rms,
                    current_action_rms=current_action_rms,
                    expert_frames=expert_frames,
                    dagger_frames=dagger_frames,
                    student_frame_history=student_frame_history,
                    history=history,
                )
                print(
                    f"autosave best: accepted={accepted_steps} trial={trial_count} "
                    f"{output_checkpoint}"
                )

            current_action_rms = min(
                float(args.initial_action_rms),
                current_action_rms * float(args.safe_grow),
            )
            print(
                f"budget continuation: ACCEPT step={accepted_steps} "
                f"{v161._aggregate(continuation_results)} "
                f"next-radius={current_action_rms:.6g}"
            )

            del cached
            cached = None

            if time.monotonic() - start_time >= train_budget_s:
                stopped_reason = "time-budget"
                break

            model.load_state_dict(continuation_state)
            model.prepare_recurrent_runtime()
            dagger_sequences, dagger_frames = v181._collect_student_state_sequences(
                model,
                anchors,
                round_index=round_index,
                collection_index=accepted_steps,
                lead_s=lead_s,
                control_dt_s=control_dt_s,
                physics_dt_s=physics_dt_s,
                device=device,
            )
            student_frame_history.append(dagger_frames)
            training_sequences = [*expert_sequences, *dagger_sequences]
            print(
                f"aggregate-data generation={accepted_steps} expert={expert_frames} "
                f"student-state={dagger_frames} "
                f"total={expert_frames + dagger_frames} frames"
            )
        else:
            stopped_reason = "max-trials"
    finally:
        if cached is not None:
            del cached

    model.load_state_dict(best_state)
    model.prepare_recurrent_runtime()
    v190._freeze_actor_only(model)

    _save_best(
        output_checkpoint,
        parent,
        model=model,
        best_state=best_state,
        source_checkpoint=source_checkpoint,
        round_index=round_index,
        requested_hours=args.hours,
        start_time=start_time,
        trial_count=trial_count,
        accepted_steps=accepted_steps,
        best_step=best_step,
        stopped_reason=stopped_reason,
        actor_steps=args.actor_steps,
        lr=args.lr,
        stay_coef=args.stay_coef,
        initial_action_rms=args.initial_action_rms,
        current_action_rms=current_action_rms,
        expert_frames=expert_frames,
        dagger_frames=dagger_frames,
        student_frame_history=student_frame_history,
        history=history,
    )

    print("=== selected Train-safe budget checkpoint ===")
    print(
        f"stop={stopped_reason} elapsed={_format_duration(time.monotonic() - start_time)} "
        f"trials={trial_count} accepted={accepted_steps} selected={best_step}: "
        + v161._aggregate(best_results)
    )

    print("=== selected checkpoint continuous Validation ===")
    v162._evaluate_role_continuous(
        model,
        validation,
        label="validation",
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
        device=device,
    )
    print(
        f"budget final elapsed={_format_duration(time.monotonic() - start_time)} "
        f"checkpoint={output_checkpoint}"
    )


if __name__ == "__main__":
    main()
