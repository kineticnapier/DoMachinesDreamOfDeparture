from __future__ import annotations

"""Audit set-valued N-key BC semantics on teacher-forced Train anchors.

Unlike ``audit_n_key_bc_actions.py``, this does not require the student to copy
which *free* finger the privileged center-first teacher happened to choose.
It mirrors the permutation-invariant training objective:

* release identity remains fixed for physically held teacher keys;
* only the number of requested free-key presses matters;
* remaining free keys are audited for extra positive pushes.

This is teacher-forced only.  It answers whether the checkpoint understands the
set-valued action semantics before closed-loop distribution shift is involved.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

import train_real_chart_v080 as v080
from dmdod.multichart_dataset import discover_multichart_dataset
from dmdod.n_key_motor import n_key_names
from dmdod.n_key_policy import NKeyRecurrentActorCritic
from dmdod.n_key_real_chart import n_key_hud_real_chart_input_dim
from dmdod.n_key_training import collect_n_key_expert_sequence


TEACHER_ACTIVE_THRESHOLD = 0.25
PRESS_THRESHOLDS = (0.25, 0.50, 0.70)
RELEASE_THRESHOLD = -0.30
EXTRA_PUSH_THRESHOLDS = (0.05, 0.25)


@dataclass(frozen=True, slots=True)
class SetActionAuditCounts:
    frames: int = 0
    press_slots: int = 0
    press_ge_025: int = 0
    press_ge_050: int = 0
    press_ge_070: int = 0
    press_frames: int = 0
    count_exact_070_frames: int = 0
    count_under_070_frames: int = 0
    count_over_070_frames: int = 0
    release_slots: int = 0
    release_le_m030: int = 0
    extra_slots: int = 0
    extra_gt_005: int = 0
    extra_gt_025: int = 0
    extra_gt_005_frames: int = 0
    extra_gt_025_frames: int = 0
    strict_frames: int = 0

    def __add__(self, other: "SetActionAuditCounts") -> "SetActionAuditCounts":
        return SetActionAuditCounts(
            frames=self.frames + other.frames,
            press_slots=self.press_slots + other.press_slots,
            press_ge_025=self.press_ge_025 + other.press_ge_025,
            press_ge_050=self.press_ge_050 + other.press_ge_050,
            press_ge_070=self.press_ge_070 + other.press_ge_070,
            press_frames=self.press_frames + other.press_frames,
            count_exact_070_frames=self.count_exact_070_frames + other.count_exact_070_frames,
            count_under_070_frames=self.count_under_070_frames + other.count_under_070_frames,
            count_over_070_frames=self.count_over_070_frames + other.count_over_070_frames,
            release_slots=self.release_slots + other.release_slots,
            release_le_m030=self.release_le_m030 + other.release_le_m030,
            extra_slots=self.extra_slots + other.extra_slots,
            extra_gt_005=self.extra_gt_005 + other.extra_gt_005,
            extra_gt_025=self.extra_gt_025 + other.extra_gt_025,
            extra_gt_005_frames=self.extra_gt_005_frames + other.extra_gt_005_frames,
            extra_gt_025_frames=self.extra_gt_025_frames + other.extra_gt_025_frames,
            strict_frames=self.strict_frames + other.strict_frames,
        )


def _count(mask: torch.Tensor) -> int:
    return int(mask.sum().item())


def audit_set_actions(predicted: torch.Tensor, target: torch.Tensor) -> SetActionAuditCounts:
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("predicted and target must have matching shape [T, K]")
    key_count = int(predicted.shape[1])
    if key_count < 2 or key_count % 2 != 0:
        raise ValueError("action width must be an even integer >= 2")

    press = target > TEACHER_ACTIVE_THRESHOLD
    release = target < -TEACHER_ACTIVE_THRESHOLD
    available = ~release
    press_count = press.sum(dim=1)
    available_count = available.sum(dim=1)
    if bool((press_count > available_count).any()):
        raise ValueError("teacher requests more presses than non-held keys")

    ranked, _ = torch.sort(
        predicted.masked_fill(release, -2.0),
        dim=1,
        descending=True,
    )
    ranks = torch.arange(key_count, device=predicted.device).reshape(1, key_count)
    ranked_press = ranks < press_count.reshape(-1, 1)
    ranked_available = ranks < available_count.reshape(-1, 1)
    ranked_extra = ranked_available & ~ranked_press

    press_025 = ranked_press & (ranked >= PRESS_THRESHOLDS[0])
    press_050 = ranked_press & (ranked >= PRESS_THRESHOLDS[1])
    press_070 = ranked_press & (ranked >= PRESS_THRESHOLDS[2])

    available_press_count_070 = (
        ((ranked >= PRESS_THRESHOLDS[2]) & ranked_available).sum(dim=1)
    )
    exact_070 = available_press_count_070 == press_count
    under_070 = available_press_count_070 < press_count
    over_070 = available_press_count_070 > press_count

    release_good = release & (predicted <= RELEASE_THRESHOLD)
    extra_005 = ranked_extra & (ranked > EXTRA_PUSH_THRESHOLDS[0])
    extra_025 = ranked_extra & (ranked > EXTRA_PUSH_THRESHOLDS[1])

    press_ok = ((~ranked_press) | (ranked >= PRESS_THRESHOLDS[2])).all(dim=1)
    release_ok = ((~release) | (predicted <= RELEASE_THRESHOLD)).all(dim=1)
    extra_ok = ((~ranked_extra) | (ranked <= EXTRA_PUSH_THRESHOLDS[0])).all(dim=1)
    strict = press_ok & release_ok & extra_ok

    return SetActionAuditCounts(
        frames=int(predicted.shape[0]),
        press_slots=_count(ranked_press),
        press_ge_025=_count(press_025),
        press_ge_050=_count(press_050),
        press_ge_070=_count(press_070),
        press_frames=_count(press_count > 0),
        count_exact_070_frames=_count(exact_070),
        count_under_070_frames=_count(under_070),
        count_over_070_frames=_count(over_070),
        release_slots=_count(release),
        release_le_m030=_count(release_good),
        extra_slots=_count(ranked_extra),
        extra_gt_005=_count(extra_005),
        extra_gt_025=_count(extra_025),
        extra_gt_005_frames=_count(extra_005.any(dim=1)),
        extra_gt_025_frames=_count(extra_025.any(dim=1)),
        strict_frames=_count(strict),
    )


def _pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator > 0 else 0.0


def _format_counts(prefix: str, counts: SetActionAuditCounts) -> str:
    return (
        f"{prefix} frames={counts.frames} "
        f"press-slots={counts.press_slots} "
        f"recall@.25={_pct(counts.press_ge_025, counts.press_slots):.2f}% "
        f"@.50={_pct(counts.press_ge_050, counts.press_slots):.2f}% "
        f"@.70={_pct(counts.press_ge_070, counts.press_slots):.2f}% | "
        f"count@.70 exact={_pct(counts.count_exact_070_frames, counts.frames):.2f}% "
        f"under={_pct(counts.count_under_070_frames, counts.frames):.2f}% "
        f"over={_pct(counts.count_over_070_frames, counts.frames):.2f}% | "
        f"release={counts.release_slots} "
        f"recall@-.30={_pct(counts.release_le_m030, counts.release_slots):.2f}% | "
        f"extra-free={counts.extra_slots} "
        f">.05={_pct(counts.extra_gt_005, counts.extra_slots):.2f}% "
        f">.25={_pct(counts.extra_gt_025, counts.extra_slots):.2f}% "
        f"frames>.05={_pct(counts.extra_gt_005_frames, counts.frames):.2f}% | "
        f"strict-frame={_pct(counts.strict_frames, counts.frames):.2f}%"
    )


def _teacher_forced_predictions(
    model: NKeyRecurrentActorCritic,
    observations: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        state = model.initial_state(observations.device)
        means, _, _ = model.forward_sequence(observations, state)
        return torch.tanh(means)


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit set-valued N-key actions on teacher-forced Train anchors."
    )
    parser.add_argument("dataset")
    parser.add_argument("checkpoint")
    parser.add_argument("--anchor-limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.anchor_limit is not None and args.anchor_limit <= 0:
        raise SystemExit("--anchor-limit must be positive")

    device = _device_from_arg(args.device)
    checkpoint_path = Path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)

    key_count = int(payload["key_count"])
    key_names = n_key_names(key_count)
    input_dim = int(payload["input_dim"])
    expected_input = n_key_hud_real_chart_input_dim(key_count)
    if input_dim != expected_input:
        raise SystemExit(
            f"checkpoint input_dim={input_dim} does not match {key_count}K expected {expected_input}"
        )

    model = NKeyRecurrentActorCritic(
        input_dim=input_dim,
        key_count=key_count,
        hidden_dim=int(payload["hidden_dim"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.gru.flatten_parameters()

    dataset = discover_multichart_dataset(args.dataset)
    train_charts = v080._compile_role(dataset.train)
    anchors = v080._build_anchor_segments(
        train_charts,
        window_s=float(payload["train_window"]),
        anchors_per_chart=int(payload["anchors_per_chart"]),
    )
    checkpoint_limit = payload.get("anchor_limit")
    limit = args.anchor_limit if args.anchor_limit is not None else checkpoint_limit
    if limit is not None:
        anchors = anchors[: int(limit)]

    calibration = payload.get("calibration") or {}
    if "lead_s" not in calibration:
        raise SystemExit("checkpoint calibration is missing lead_s")
    lead_s = float(calibration["lead_s"])
    control_dt_s = float(payload["control_dt"])
    physics_dt_s = float(payload.get("physics_dt", 0.001))

    print("=== DMDOD N-Key Set-Valued Teacher-Forced Audit ===")
    print(
        f"checkpoint={checkpoint_path} keys={key_count} input={input_dim}D "
        f"anchors={len(anchors)} device={device}"
    )
    print("key-order: " + ",".join(key_names))

    aggregate = SetActionAuditCounts()
    for index, named in enumerate(anchors, 1):
        rollout = collect_n_key_expert_sequence(
            named.segment,
            key_count=key_count,
            lead_s=lead_s,
            control_dt_s=control_dt_s,
            physics_dt_s=physics_dt_s,
            device=device,
            source=f"set-audit-{index}-{named.chart_name}",
        )
        predicted = _teacher_forced_predictions(model, rollout.sequence.observations)
        counts = audit_set_actions(predicted, rollout.sequence.teacher_actions)
        aggregate = aggregate + counts
        print(_format_counts(f"{index:03d}/{len(anchors)} {named.chart_name}:", counts))

    print("=== aggregate ===")
    print(_format_counts("all:", aggregate))


if __name__ == "__main__":
    main()
