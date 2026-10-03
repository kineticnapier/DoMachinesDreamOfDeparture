from __future__ import annotations

"""v1.6.1: one guarded student-state DAgger round for N-key policies.

The input checkpoint is left untouched.  The student executes hard actions in
closed loop while the privileged teacher labels the visited student states.  The
new student-state sequences are aggregated with fresh expert-anchor sequences
and trained with the set-valued N-key actuation loss.

Every DAgger epoch is evaluated only on Train anchors.  A candidate epoch is
rejected if it turns a previously safe anchor into overload or regresses an
anchor's hits beyond the mature real-chart hit tolerance.  Among safe epochs,
the checkpoint with the best Train aggregate hits is retained, with XAcc and
TooEarly used only as tie-breakers.  The pre-DAgger model is epoch 0 and remains
a fallback, so a DAgger round cannot force a worse checkpoint.  Validation is
evaluated once after selection and Final is never touched.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

import train_real_chart_v080 as v080
import train_real_chart_v160_n_key_bootstrap as v160
from dmdod.multichart_dataset import discover_multichart_dataset
from dmdod.n_key_motor import n_key_names
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
    discretize_n_key_action,
)
from dmdod.real_chart_features import DEFAULT_REAL_CHART_FEATURE_CONFIG


TRAINER_VERSION = "1.6.1-n-key-dagger"
CHECKPOINT_FORMAT_VERSION = 18
DEFAULT_PRESS_THRESHOLD = 0.25
DEFAULT_RELEASE_THRESHOLD = -0.45


@dataclass(frozen=True, slots=True)
class RoleSummary:
    hits: int
    targets: int
    x_accuracy_percent: float
    early: int
    overloaded: bool
    keydowns: int


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    return device


def _default_output_path(source: Path, round_index: int) -> Path:
    suffix = source.suffix or ".pt"
    stem = source.name[: -len(suffix)] if source.name.endswith(suffix) else source.name
    return source.with_name(f"{stem}_dagger{round_index}{suffix}")


def _evaluate_hard(
    model: NKeyRecurrentActorCritic,
    named,
    *,
    control_dt_s: float,
    physics_dt_s: float,
    press_threshold: float,
    release_threshold: float,
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
            values = tuple(float(value.item()) for value in torch.tanh(mean))
            action = discretize_n_key_action(
                values,
                press_threshold=press_threshold,
                release_threshold=release_threshold,
            )
            step = env.step(action)
            observation = step.observation
            if step.done:
                break
        else:
            raise RuntimeError("N-key DAgger evaluation exceeded step budget")
    return env.stats, int(env.physical_keydowns)


def _summarize(results: list[tuple[object, int]]) -> RoleSummary:
    if not results:
        return RoleSummary(0, 0, 0.0, 0, False, 0)
    targets = sum(int(stats.targets) for stats, _ in results)
    hits = sum(int(stats.hits) for stats, _ in results)
    early = sum(int(stats.too_early_presses) for stats, _ in results)
    keydowns = sum(int(keydowns) for _, keydowns in results)
    x_points = sum(float(stats.x_accuracy_points) for stats, _ in results)
    x_den = sum(float(stats.x_accuracy_denominator) for stats, _ in results)
    xacc = 100.0 * x_points / x_den if x_den > 0.0 else 0.0
    overloaded = any(bool(stats.overloaded) for stats, _ in results)
    return RoleSummary(hits, targets, xacc, early, overloaded, keydowns)


def _format_summary(summary: RoleSummary) -> str:
    return (
        f"H={summary.hits}/{summary.targets} X={summary.x_accuracy_percent:.2f}% "
        f"early={summary.early} over={summary.overloaded} keydowns={summary.keydowns}"
    )


def _aggregate(results: list[tuple[object, int]]) -> str:
    if not results:
        return "none"
    return _format_summary(_summarize(results))


def _evaluate_role(
    model: NKeyRecurrentActorCritic,
    segments,
    *,
    label: str,
    control_dt_s: float,
    physics_dt_s: float,
    press_threshold: float,
    release_threshold: float,
    device: torch.device,
) -> list[tuple[object, int]]:
    results: list[tuple[object, int]] = []
    total = len(segments)
    for index, named in enumerate(segments, 1):
        stats, keydowns = _evaluate_hard(
            model,
            named,
            control_dt_s=control_dt_s,
            physics_dt_s=physics_dt_s,
            press_threshold=press_threshold,
            release_threshold=release_threshold,
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
    print(f"{label} aggregate: " + _aggregate(results))
    return results


def _anchor_hit_tolerance(reference_stats) -> int:
    return max(
        int(v080.v062.DEFAULT_VALIDATION_HIT_DROP_ABSOLUTE),
        round(
            int(reference_stats.targets)
            * float(v080.v062.DEFAULT_VALIDATION_HIT_DROP_FRACTION)
        ),
    )


def _train_safety_guard(
    references: list[tuple[object, int]],
    candidates: list[tuple[object, int]],
) -> tuple[bool, tuple[str, ...]]:
    """Protect every Train anchor against unsafe overload and large hit loss."""

    if len(references) != len(candidates):
        raise ValueError("Train guard reference/candidate count mismatch")

    reasons: list[str] = []
    for index, ((reference, _), (candidate, _)) in enumerate(
        zip(references, candidates),
        1,
    ):
        if not bool(reference.overloaded) and bool(candidate.overloaded):
            reasons.append(f"anchor {index} safe->overload")
            continue

        tolerance = _anchor_hit_tolerance(reference)
        floor = int(reference.hits) - tolerance
        if int(candidate.hits) < floor:
            reasons.append(
                f"anchor {index} hits {int(candidate.hits)}<{floor} "
                f"(reference={int(reference.hits)} tolerance={tolerance})"
            )

    return not reasons, tuple(reasons)


def _selection_key(results: list[tuple[object, int]]) -> tuple[float, ...]:
    """Train-only best-checkpoint order: hits, then XAcc, then lower noise."""

    summary = _summarize(results)
    return (
        float(summary.hits),
        float(summary.x_accuracy_percent),
        -float(summary.early),
        -float(summary.keydowns),
    )


def _clone_model_state(model: NKeyRecurrentActorCritic) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _selection_record(
    *,
    epoch: int,
    loss: float | None,
    accepted: bool,
    selected: bool,
    reasons: tuple[str, ...],
    results: list[tuple[object, int]],
) -> dict:
    summary = _summarize(results)
    return {
        "epoch": int(epoch),
        "loss": None if loss is None else float(loss),
        "guard_accepted": bool(accepted),
        "selected_when_evaluated": bool(selected),
        "guard_reasons": tuple(reasons),
        "hits": int(summary.hits),
        "targets": int(summary.targets),
        "x_accuracy_percent": float(summary.x_accuracy_percent),
        "early": int(summary.early),
        "overloaded": bool(summary.overloaded),
        "keydowns": int(summary.keydowns),
    }


def _dagger_checkpoint_payload(
    parent: dict,
    *,
    model: NKeyRecurrentActorCritic,
    source_checkpoint: Path,
    output_checkpoint: Path,
    round_index: int,
    dagger_epoch: int,
    dagger_epochs: int,
    press_threshold: float,
    release_threshold: float,
    lr: float,
    expert_frames: int,
    dagger_frames: int,
    losses: list[float],
    model_state: dict[str, torch.Tensor] | None = None,
    selected_epoch: int | None = None,
    selection_history: list[dict] | None = None,
) -> dict:
    payload = dict(parent)
    payload.update(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "trainer_version": TRAINER_VERSION,
            "model_state": model.state_dict() if model_state is None else model_state,
            "dagger_round": int(round_index),
            "completed_dagger_epoch": int(dagger_epoch),
            "requested_dagger_epochs": int(dagger_epochs),
            "dagger_selected_epoch": int(
                dagger_epoch if selected_epoch is None else selected_epoch
            ),
            "dagger_press_threshold": float(press_threshold),
            "dagger_release_threshold": float(release_threshold),
            "dagger_lr": float(lr),
            "dagger_expert_frames": int(expert_frames),
            "dagger_student_state_frames": int(dagger_frames),
            "dagger_loss_history": list(losses),
            "dagger_train_selection_history": list(selection_history or []),
            "dagger_selection_uses_validation": False,
            "dagger_source_checkpoint": str(source_checkpoint),
            "dagger_output_checkpoint": str(output_checkpoint),
            "final_used_for_selection": False,
            "finalized": False,
        }
    )
    return payload


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one hard-action student-state DAgger round on N-key Train anchors, "
            "select the best Train-safe epoch, then evaluate Validation once."
        )
    )
    parser.add_argument("dataset")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dagger-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--chunk-steps", type=int, default=None)
    parser.add_argument("--press-threshold", type=float, default=DEFAULT_PRESS_THRESHOLD)
    parser.add_argument("--release-threshold", type=float, default=DEFAULT_RELEASE_THRESHOLD)
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
    if not (0.0 < args.press_threshold <= 1.0):
        raise SystemExit("--press-threshold must be in (0, 1]")
    if not (-1.0 <= args.release_threshold < 0.0):
        raise SystemExit("--release-threshold must be in [-1, 0)")
    if args.anchor_limit is not None and args.anchor_limit <= 0:
        raise SystemExit("--anchor-limit must be positive")
    if args.validation_limit is not None and args.validation_limit <= 0:
        raise SystemExit("--validation-limit must be positive")

    device = _device_from_arg(args.device)
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
        args.output or _default_output_path(source_checkpoint, round_index)
    )

    print("=== DMDOD v1.6.1 N-Key DAgger ===")
    print(
        f"source={source_checkpoint} output={output_checkpoint} round={round_index} "
        f"keys={key_count} input={input_dim}D device={device}"
    )
    print("key-order: " + ",".join(key_names))
    print(
        f"anchors={len(anchors)} validation={len(validation)} dagger-epochs={args.dagger_epochs} "
        f"hard=p>={args.press_threshold:.2f},r<={args.release_threshold:.2f} | FINAL untouched"
    )
    print(
        "Train epoch guard: safe->overload rejected; per-anchor hit floor uses "
        "the mature real-chart tolerance. Validation is not used for selection."
    )

    print("=== pre-DAgger hard Train / epoch 0 fallback ===")
    pre_results = _evaluate_role(
        model,
        anchors,
        label="pre-train",
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
        press_threshold=args.press_threshold,
        release_threshold=args.release_threshold,
        device=device,
    )
    best_state = _clone_model_state(model)
    best_results = pre_results
    best_epoch = 0
    selection_history = [
        _selection_record(
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
            press_threshold=args.press_threshold,
            release_threshold=args.release_threshold,
            control_dt_s=control_dt_s,
            physics_dt_s=physics_dt_s,
            device=device,
            source=f"dagger{round_index}-student-{index}-{named.chart_name}",
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
        candidate_results = _evaluate_role(
            model,
            anchors,
            label=f"epoch-{epoch:03d}-train",
            control_dt_s=control_dt_s,
            physics_dt_s=physics_dt_s,
            press_threshold=args.press_threshold,
            release_threshold=args.release_threshold,
            device=device,
        )
        accepted, reasons = _train_safety_guard(pre_results, candidate_results)
        selected = accepted and _selection_key(candidate_results) > _selection_key(best_results)
        if not accepted:
            print("epoch-selection: REJECT " + "; ".join(reasons))
        elif selected:
            best_state = _clone_model_state(model)
            best_results = candidate_results
            best_epoch = epoch
            print(
                f"epoch-selection: ACCEPT new-best epoch={epoch} "
                + _aggregate(candidate_results)
            )
        else:
            print(
                f"epoch-selection: SAFE not-best epoch={epoch} "
                + _aggregate(candidate_results)
            )
        selection_history.append(
            _selection_record(
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
    print("=== selected Train-safe DAgger checkpoint ===")
    print(f"selected epoch={best_epoch}/{args.dagger_epochs}: " + _aggregate(best_results))

    _save_checkpoint(
        output_checkpoint,
        _dagger_checkpoint_payload(
            parent,
            model=model,
            model_state=best_state,
            source_checkpoint=source_checkpoint,
            output_checkpoint=output_checkpoint,
            round_index=round_index,
            dagger_epoch=args.dagger_epochs,
            dagger_epochs=args.dagger_epochs,
            selected_epoch=best_epoch,
            press_threshold=args.press_threshold,
            release_threshold=args.release_threshold,
            lr=args.lr,
            expert_frames=expert_frames,
            dagger_frames=dagger_frames,
            losses=losses,
            selection_history=selection_history,
        ),
    )

    print("=== selected checkpoint hard Validation ===")
    _evaluate_role(
        model,
        validation,
        label="validation",
        control_dt_s=control_dt_s,
        physics_dt_s=physics_dt_s,
        press_threshold=args.press_threshold,
        release_threshold=args.release_threshold,
        device=device,
    )
    print(f"checkpoint final: {output_checkpoint}")


if __name__ == "__main__":
    main()
