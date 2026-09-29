from __future__ import annotations

import argparse
from pathlib import Path

import torch

import train_real_chart_v054 as v054
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import load_compiled_adofai
from dmdod.real_chart_features import REAL_CHART_INPUT_DIM
from dmdod.recurrent_policy import RecurrentActorCritic


DEFAULT_CONTROL_DT_S = 0.010


def _checkpoint_metadata(payload: dict) -> tuple[int, int]:
    input_dim = int(payload.get("input_dim", -1))
    hidden_dim = int(payload.get("hidden_dim", -1))
    model_state = payload.get("model_state")

    if input_dim != REAL_CHART_INPUT_DIM:
        raise SystemExit(
            f"checkpoint input dimension {input_dim} does not match current encoder "
            f"({REAL_CHART_INPUT_DIM})"
        )
    if hidden_dim <= 0:
        raise SystemExit("checkpoint has no valid hidden_dim")
    if not isinstance(model_state, dict):
        raise SystemExit("checkpoint has no model_state")
    return input_dim, hidden_dim


def _resolve_eval_config(
    payload: dict,
    *,
    same_hand_override: bool | None,
    control_dt_override: float | None,
) -> tuple[bool, float, str, str]:
    signature = payload.get("signature")
    if not isinstance(signature, dict):
        signature = {}

    if same_hand_override is None:
        same_hand = bool(signature.get("same_hand", True))
        hand_source = "checkpoint" if "same_hand" in signature else "default"
    else:
        same_hand = bool(same_hand_override)
        hand_source = "override"

    if control_dt_override is None:
        control_dt_s = float(signature.get("control_dt", DEFAULT_CONTROL_DT_S))
        control_source = "checkpoint" if "control_dt" in signature else "default"
    else:
        control_dt_s = float(control_dt_override)
        control_source = "override"

    if control_dt_s <= 0.0:
        raise SystemExit("control dt must be positive")
    return same_hand, control_dt_s, hand_source, control_source


def _resolve_range(duration_s: float, start_s: float, end_s: float | None) -> tuple[float, float]:
    if duration_s <= 0.0:
        raise SystemExit("chart has no positive duration")
    if start_s < 0.0:
        raise SystemExit("--start must be non-negative")
    if start_s >= duration_s:
        raise SystemExit(
            f"--start={start_s:g}s is outside chart duration {duration_s:g}s"
        )

    resolved_end = duration_s if end_s is None else min(float(end_s), duration_s)
    if resolved_end <= start_s:
        raise SystemExit("--end must be greater than --start")
    return float(start_s), float(resolved_end)


def _is_different_chart(payload: dict, target_chart: str) -> bool | None:
    source = payload.get("chart")
    if not isinstance(source, str) or not source:
        return None
    try:
        return Path(source).resolve() != Path(target_chart).resolve()
    except OSError:
        return source != target_chart


def _load_model(payload: dict, *, device: torch.device) -> RecurrentActorCritic:
    _, hidden_dim = _checkpoint_metadata(payload)
    model = RecurrentActorCritic(
        input_dim=REAL_CHART_INPUT_DIM,
        hidden_dim=hidden_dim,
        initial_log_std=-1.20,
    ).to(device)
    try:
        model.load_state_dict(payload["model_state"])
    except RuntimeError as exc:
        raise SystemExit(f"checkpoint model state is incompatible: {exc}") from exc
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained DMDOD real-chart checkpoint on an arbitrary .adofai "
            "without performing any training or checkpoint writes."
        )
    )
    parser.add_argument("checkpoint", help="trained real-chart .pt checkpoint")
    parser.add_argument("chart", help="target .adofai chart")
    parser.add_argument("--start", type=float, default=0.0, help="segment start in chart seconds")
    parser.add_argument("--end", type=float, default=None, help="segment end in chart seconds; default=chart end")
    parser.add_argument(
        "--control-dt",
        type=float,
        default=None,
        help="override checkpoint control step in seconds",
    )
    hand_group = parser.add_mutually_exclusive_group()
    hand_group.add_argument(
        "--same-hand",
        dest="same_hand_override",
        action="store_const",
        const=True,
        default=None,
        help="override checkpoint and use same-hand fingers",
    )
    hand_group.add_argument(
        "--cross-hand",
        dest="same_hand_override",
        action="store_const",
        const=False,
        help="override checkpoint and use bilateral fingers",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")

    device = torch.device("cpu")
    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise SystemExit("checkpoint payload is not a dictionary")

    model = _load_model(payload, device=device)
    same_hand, control_dt_s, hand_source, control_source = _resolve_eval_config(
        payload,
        same_hand_override=args.same_hand_override,
        control_dt_override=args.control_dt,
    )

    compiled = load_compiled_adofai(args.chart)
    start_s, end_s = _resolve_range(compiled.duration_s, args.start, args.end)
    segment = build_playable_segment(compiled, start_s=start_s, end_s=end_s)
    if not segment.targets:
        raise SystemExit("evaluation segment contains no playable targets")

    different_chart = _is_different_chart(payload, args.chart)
    zero_shot_text = (
        "yes" if different_chart is True else "no" if different_chart is False else "unknown"
    )
    source_chart = payload.get("chart", "<unknown>")
    trainer_version = payload.get("trainer_version", "<unknown>")
    format_version = payload.get("format_version", "<unknown>")

    print("=== DMDOD / Real Chart Checkpoint Evaluator ===")
    print(
        f"checkpoint={checkpoint_path} trainer={trainer_version} format={format_version} "
        f"finalized={payload.get('finalized', '<unknown>')}"
    )
    print(f"model-source-chart={source_chart}")
    print(f"target-chart={args.chart} different-chart={zero_shot_text}")
    print(
        f"segment={start_s:g}..{end_s:g}s targets={len(segment.targets)} "
        f"hidden={model.hidden_dim}"
    )
    print(
        f"body={'same-hand' if same_hand else 'cross-hand'}({hand_source}) "
        f"control={control_dt_s * 1000.0:.1f}ms({control_source}) training=DISABLED"
    )

    result = v054._evaluate_student(
        model,
        segment,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        device=device,
    )
    print(v054._format_eval("student eval", result))


if __name__ == "__main__":
    main()
