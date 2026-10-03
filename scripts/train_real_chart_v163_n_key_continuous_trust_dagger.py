from __future__ import annotations

"""v1.6.3: trust-region continuous-action DAgger for N-key policies.

A full BC epoch can cross a sharp closed-loop boundary even while the
teacher-forced loss improves.  Each epoch here therefore produces only a
proposal direction.  The proposal is line-searched from the current accepted
Train-safe state.  Only a Train-safe interpolation that improves the Train
selection key is accepted, and the following epoch starts from that accepted
state rather than from a rejected proposal.  Validation is evaluated once after
Train-only selection; Final is never touched.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

import train_real_chart_v080 as v080
import train_real_chart_v160_n_key_bootstrap as v160
import train_real_chart_v161_n_key_dagger as v161
import train_real_chart_v162_n_key_continuous_dagger as v162
from dmdod.multichart_dataset import discover_multichart_dataset
from dmdod.n_key_motor import n_key_names
from dmdod.n_key_policy import NKeyRecurrentActorCritic
from dmdod.n_key_real_chart import n_key_hud_real_chart_input_dim
from dmdod.n_key_training import (
    NKeyBCSequence,
    collect_n_key_dagger_sequence,
    collect_n_key_expert_sequence,
)


TRAINER_VERSION = "1.6.3-n-key-continuous-trust-dagger"
CHECKPOINT_FORMAT_VERSION = 20
DEFAULT_TRUST_ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)


@dataclass(frozen=True, slots=True)
class TrustCandidate:
    alpha: float
    state: dict[str, torch.Tensor]
    results: list[tuple[object, int]]
    guard_accepted: bool
    guard_reasons: tuple[str, ...]


def _parse_alphas(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("trust alpha list must not be empty")
    if any(not (0.0 < value <= 1.0) for value in values):
        raise ValueError("trust alphas must be in (0, 1]")
    if len(set(values)) != len(values):
        raise ValueError("trust alphas must be unique")
    return values


def _interpolate_state(
    base: dict[str, torch.Tensor],
    proposal: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if base.keys() != proposal.keys():
        raise ValueError("state dictionaries must have identical keys")

    blended: dict[str, torch.Tensor] = {}
    for name in base:
        left = base[name]
        right = proposal[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"state tensor mismatch for {name}")
        if torch.is_floating_point(left) or torch.is_complex(left):
            blended[name] = (left + alpha * (right - left)).detach().cpu().clone()
        else:
            if not torch.equal(left, right):
                raise ValueError(f"non-floating state changed for {name}")
            blended[name] = left.detach().cpu().clone()
    return blended


def _select_improving_candidate(
    reference_results: list[tuple[object, int]],
    candidates: list[TrustCandidate],
) -> TrustCandidate | None:
    reference_key = v161._selection_key(reference_results)
    eligible = [
        candidate
        for candidate in candidates
        if candidate.guard_accepted
        and v161._selection_key(candidate.results) > reference_key
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda candidate: v161._selection_key(candidate.results))


def _trust_record(
    *,
    epoch: int,
    alpha: float,
    loss: float,
    candidate: TrustCandidate,
    accepted_for_continuation: bool,
) -> dict:
    summary = v161._summarize(candidate.results)
    return {
        "epoch": int(epoch),
        "alpha": float(alpha),
        "proposal_loss": float(loss),
        "guard_accepted": bool(candidate.guard_accepted),
        "accepted_for_continuation": bool(accepted_for_continuation),
        "guard_reasons": tuple(candidate.guard_reasons),
        "hits": int(summary.hits),
        "targets": int(summary.targets),
        "x_accuracy_percent": float(summary.x_accuracy_percent),
        "early": int(summary.early),
        "overloaded": bool(summary.overloaded),
        "keydowns": int(summary.keydowns),
    }


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
    selected_alpha: float,
    lr: float,
    expert_frames: int,
    dagger_frames: int,
    losses: list[float],
    trust_alphas: tuple[float, ...],
    trust_history: list[dict],
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
            "dagger_selected_alpha": float(selected_alpha),
            "dagger_action_mode": "continuous",
            "dagger_press_threshold": None,
            "dagger_release_threshold": None,
            "dagger_lr": float(lr),
            "dagger_expert_frames": int(expert_frames),
            "dagger_student_state_frames": int(dagger_frames),
            "dagger_loss_history": list(losses),
            "dagger_trust_alphas": tuple(float(alpha) for alpha in trust_alphas),
            "dagger_trust_history": list(trust_history),
            "dagger_trust_continuation": "accepted-state",
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
            "Run continuous-action N-key DAgger with Train-only trust-region "
            "line search and accepted-state continuation."
        )
    )
    parser.add_argument("dataset")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dagger-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--chunk-steps", type=int, default=None)
    parser.add_argument(
        "--trust-alphas",
        default=",".join(str(value) for value in DEFAULT_TRUST_ALPHAS),
    )
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
    try:
        trust_alphas = _parse_alphas(args.trust_alphas)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

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

    print("=== DMDOD v1.6.3 N-Key Continuous Trust DAgger ===")
    print(
        f"source={source_checkpoint} output={output_checkpoint} round={round_index} "
        f"keys={key_count} input={input_dim}D device={device}"
    )
    print("key-order: " + ",".join(key_names))
    print(
        f"anchors={len(anchors)} validation={len(validation)} dagger-epochs={args.dagger_epochs} "
        f"action=continuous lr={args.lr:g} | FINAL untouched"
    )
    print("trust-alphas=" + ",".join(f"{alpha:g}" for alpha in trust_alphas))
    print(
        "Each epoch proposes from the current accepted state; line-search candidates "
        "are guarded against that state. Validation is not used for selection."
    )

    print("=== pre-DAgger continuous Train / accepted state 0 ===")
    accepted_results = v162._evaluate_role_continuous(
        model,
        anchors,
        label="pre-train",
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
        device=device,
    )
    accepted_state = v161._clone_model_state(model)
    accepted_epoch = 0
    accepted_alpha = 0.0

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
            source=f"dagger{round_index}-trust-expert-{index}-{named.chart_name}",
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
            source=f"dagger{round_index}-trust-student-{index}-{named.chart_name}",
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
    trust_history: list[dict] = []

    for epoch in range(1, args.dagger_epochs + 1):
        model.load_state_dict(accepted_state)
        model.gru.flatten_parameters()
        base_state = accepted_state
        base_results = accepted_results

        loss = v160._train_bc_epoch(
            model,
            training_sequences,
            optimizer=optimizer,
            chunk_steps=chunk_steps,
        )
        losses.append(loss)
        proposal_state = v161._clone_model_state(model)
        print(f"dagger-proposal {epoch:03d}/{args.dagger_epochs} loss={loss:.6f}")
        print(f"=== Train trust line search epoch {epoch}/{args.dagger_epochs} ===")

        candidates: list[TrustCandidate] = []
        for alpha in trust_alphas:
            candidate_state = _interpolate_state(base_state, proposal_state, alpha)
            model.load_state_dict(candidate_state)
            model.gru.flatten_parameters()
            candidate_results = v162._evaluate_role_continuous(
                model,
                anchors,
                label=f"epoch-{epoch:03d}-a{alpha:g}",
                control_dt_s=control_dt_s,
                physics_dt_s=physics_dt_s,
                device=device,
            )
            safe, reasons = v161._train_safety_guard(base_results, candidate_results)
            candidate = TrustCandidate(
                alpha=float(alpha),
                state=candidate_state,
                results=candidate_results,
                guard_accepted=safe,
                guard_reasons=reasons,
            )
            candidates.append(candidate)
            improved = safe and (
                v161._selection_key(candidate_results) > v161._selection_key(base_results)
            )
            status = "SAFE+IMPROVE" if improved else "SAFE" if safe else "REJECT"
            detail = "" if safe else " " + "; ".join(reasons)
            print(
                f"trust alpha={alpha:g}: {status} {v161._aggregate(candidate_results)}{detail}"
            )

        chosen = _select_improving_candidate(base_results, candidates)
        for candidate in candidates:
            trust_history.append(
                _trust_record(
                    epoch=epoch,
                    alpha=candidate.alpha,
                    loss=loss,
                    candidate=candidate,
                    accepted_for_continuation=(chosen is candidate),
                )
            )

        if chosen is None:
            accepted_state = base_state
            accepted_results = base_results
            model.load_state_dict(accepted_state)
            model.gru.flatten_parameters()
            print(
                f"epoch-continuation: KEEP previous accepted state epoch={accepted_epoch} "
                f"alpha={accepted_alpha:g} {v161._aggregate(accepted_results)}"
            )
        else:
            accepted_state = chosen.state
            accepted_results = chosen.results
            accepted_epoch = epoch
            accepted_alpha = chosen.alpha
            model.load_state_dict(accepted_state)
            model.gru.flatten_parameters()
            print(
                f"epoch-continuation: ACCEPT epoch={epoch} alpha={chosen.alpha:g} "
                + v161._aggregate(accepted_results)
            )

    model.load_state_dict(accepted_state)
    model.gru.flatten_parameters()
    print("=== selected Train-safe trust-region checkpoint ===")
    print(
        f"selected epoch={accepted_epoch}/{args.dagger_epochs} alpha={accepted_alpha:g}: "
        + v161._aggregate(accepted_results)
    )

    v161._save_checkpoint(
        output_checkpoint,
        _checkpoint_payload(
            parent,
            model_state=accepted_state,
            source_checkpoint=source_checkpoint,
            output_checkpoint=output_checkpoint,
            round_index=round_index,
            completed_epoch=args.dagger_epochs,
            requested_epochs=args.dagger_epochs,
            selected_epoch=accepted_epoch,
            selected_alpha=accepted_alpha,
            lr=args.lr,
            expert_frames=expert_frames,
            dagger_frames=dagger_frames,
            losses=losses,
            trust_alphas=trust_alphas,
            trust_history=trust_history,
        ),
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
    print(f"checkpoint final: {output_checkpoint}")


if __name__ == "__main__":
    main()
