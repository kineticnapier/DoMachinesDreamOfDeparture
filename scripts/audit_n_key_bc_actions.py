from __future__ import annotations

"""Audit N-key BC action recall on the exact teacher-forced training trajectories.

This deliberately does not run the student closed-loop.  It feeds the saved
teacher observation sequence through the checkpointed recurrent policy and
measures whether the policy has actually learned the sparse press/release
commands that matter physically.
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
NEUTRAL_PUSH_THRESHOLDS = (0.05, 0.25)


@dataclass(frozen=True, slots=True)
class ActionAuditCounts:
    press_total: int = 0
    press_ge_025: int = 0
    press_ge_050: int = 0
    press_ge_070: int = 0
    release_total: int = 0
    release_le_m030: int = 0
    neutral_total: int = 0
    neutral_gt_005: int = 0
    neutral_gt_025: int = 0

    def __add__(self, other: "ActionAuditCounts") -> "ActionAuditCounts":
        return ActionAuditCounts(
            press_total=self.press_total + other.press_total,
            press_ge_025=self.press_ge_025 + other.press_ge_025,
            press_ge_050=self.press_ge_050 + other.press_ge_050,
            press_ge_070=self.press_ge_070 + other.press_ge_070,
            release_total=self.release_total + other.release_total,
            release_le_m030=self.release_le_m030 + other.release_le_m030,
            neutral_total=self.neutral_total + other.neutral_total,
            neutral_gt_005=self.neutral_gt_005 + other.neutral_gt_005,
            neutral_gt_025=self.neutral_gt_025 + other.neutral_gt_025,
        )


@dataclass(frozen=True, slots=True)
class ActionAuditReport:
    overall: ActionAuditCounts
    per_key: tuple[ActionAuditCounts, ...]


def _count_mask(mask: torch.Tensor) -> int:
    return int(mask.sum().item())


def _audit_column(predicted: torch.Tensor, target: torch.Tensor) -> ActionAuditCounts:
    press = target > TEACHER_ACTIVE_THRESHOLD
    release = target < -TEACHER_ACTIVE_THRESHOLD
    neutral = ~(press | release)
    return ActionAuditCounts(
        press_total=_count_mask(press),
        press_ge_025=_count_mask(press & (predicted >= PRESS_THRESHOLDS[0])),
        press_ge_050=_count_mask(press & (predicted >= PRESS_THRESHOLDS[1])),
        press_ge_070=_count_mask(press & (predicted >= PRESS_THRESHOLDS[2])),
        release_total=_count_mask(release),
        release_le_m030=_count_mask(release & (predicted <= RELEASE_THRESHOLD)),
        neutral_total=_count_mask(neutral),
        neutral_gt_005=_count_mask(neutral & (predicted > NEUTRAL_PUSH_THRESHOLDS[0])),
        neutral_gt_025=_count_mask(neutral & (predicted > NEUTRAL_PUSH_THRESHOLDS[1])),
    )


def audit_actions(predicted: torch.Tensor, target: torch.Tensor) -> ActionAuditReport:
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("predicted and target must have matching shape [T, K]")
    if predicted.shape[1] < 2 or predicted.shape[1] % 2 != 0:
        raise ValueError("action width must be an even integer >= 2")

    per_key = tuple(
        _audit_column(predicted[:, index], target[:, index])
        for index in range(predicted.shape[1])
    )
    overall = ActionAuditCounts()
    for counts in per_key:
        overall = overall + counts
    return ActionAuditReport(overall=overall, per_key=per_key)


def merge_reports(reports: list[ActionAuditReport]) -> ActionAuditReport:
    if not reports:
        raise ValueError("at least one audit report is required")
    width = len(reports[0].per_key)
    if any(len(report.per_key) != width for report in reports):
        raise ValueError("all audit reports must have the same key width")

    overall = ActionAuditCounts()
    per_key = [ActionAuditCounts() for _ in range(width)]
    for report in reports:
        overall = overall + report.overall
        for index, counts in enumerate(report.per_key):
            per_key[index] = per_key[index] + counts
    return ActionAuditReport(overall=overall, per_key=tuple(per_key))


def _pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator > 0 else 0.0


def _format_counts(prefix: str, counts: ActionAuditCounts) -> str:
    return (
        f"{prefix} press={counts.press_total} "
        f"recall@.25={_pct(counts.press_ge_025, counts.press_total):.2f}% "
        f"@.50={_pct(counts.press_ge_050, counts.press_total):.2f}% "
        f"@.70={_pct(counts.press_ge_070, counts.press_total):.2f}% | "
        f"release={counts.release_total} "
        f"recall@-.30={_pct(counts.release_le_m030, counts.release_total):.2f}% | "
        f"neutral={counts.neutral_total} "
        f"false-push>.05={_pct(counts.neutral_gt_005, counts.neutral_total):.3f}% "
        f">.25={_pct(counts.neutral_gt_025, counts.neutral_total):.3f}%"
    )


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    return device


def _teacher_forced_predictions(
    model: NKeyRecurrentActorCritic,
    observations: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        state = model.initial_state(observations.device)
        means, _, _ = model.forward_sequence(observations, state)
        return torch.tanh(means)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit N-key checkpoint action recall on teacher-forced Train anchors."
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

    print("=== DMDOD N-Key Teacher-Forced Action Audit ===")
    print(
        f"checkpoint={checkpoint_path} keys={key_count} input={input_dim}D "
        f"anchors={len(anchors)} device={device}"
    )
    print("key-order: " + ",".join(key_names))

    reports: list[ActionAuditReport] = []
    for index, named in enumerate(anchors, 1):
        rollout = collect_n_key_expert_sequence(
            named.segment,
            key_count=key_count,
            lead_s=lead_s,
            control_dt_s=control_dt_s,
            physics_dt_s=physics_dt_s,
            device=device,
            source=f"audit-{index}-{named.chart_name}",
        )
        predicted = _teacher_forced_predictions(model, rollout.sequence.observations)
        report = audit_actions(predicted, rollout.sequence.teacher_actions)
        reports.append(report)
        print(_format_counts(f"{index:03d}/{len(anchors)} {named.chart_name}:", report.overall))

    aggregate = merge_reports(reports)
    print("=== aggregate ===")
    print(_format_counts("all:", aggregate.overall))
    print("=== per-key ===")
    for key, counts in zip(key_names, aggregate.per_key):
        print(_format_counts(f"{key}:", counts))


if __name__ == "__main__":
    main()
