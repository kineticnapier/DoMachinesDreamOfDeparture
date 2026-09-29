from __future__ import annotations

import argparse
from pathlib import Path

import torch

import eval_real_chart as base
import train_real_chart_v054 as v054
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import load_compiled_adofai
from dmdod.real_chart_features import DEFAULT_REAL_CHART_FEATURE_CONFIG
from dmdod.real_chart_hud import DiagnosticHudRealChartMotorEnv
from dmdod.real_chart_hud_features import (
    HUD_FEATURE_VERSION,
    HUD_REAL_CHART_INPUT_DIM,
    encode_hud_real_chart_observation,
)
from dmdod.recurrent_policy import RecurrentActorCritic


EXPECTED_FORMAT_VERSION = 14


def _load_hud_model(payload: dict, *, device: torch.device) -> RecurrentActorCritic:
    input_dim = int(payload.get("input_dim", -1))
    hidden_dim = int(payload.get("hidden_dim", -1))
    if input_dim != HUD_REAL_CHART_INPUT_DIM:
        raise SystemExit(
            f"checkpoint input dimension {input_dim} is not HUD encoder dimension "
            f"{HUD_REAL_CHART_INPUT_DIM}"
        )
    if hidden_dim <= 0 or not isinstance(payload.get("model_state"), dict):
        raise SystemExit("checkpoint has no valid hidden_dim/model_state")
    if payload.get("hud_observation") != HUD_FEATURE_VERSION:
        raise SystemExit("checkpoint does not use the current human-visible HUD observation")

    model = RecurrentActorCritic(
        input_dim=HUD_REAL_CHART_INPUT_DIM,
        hidden_dim=hidden_dim,
        initial_log_std=-1.20,
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def _evaluate_hud(model, segment, *, same_hand: bool, control_dt_s: float, device: torch.device):
    env = DiagnosticHudRealChartMotorEnv(
        segment,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )
    observation = env.reset()
    state = model.initial_state(device)
    max_steps = int((segment.duration_s + 2.0) / control_dt_s) + 200
    with torch.no_grad():
        for _ in range(max_steps):
            x = torch.tensor(
                encode_hud_real_chart_observation(observation),
                dtype=torch.float32,
                device=device,
            )
            action, state = model.deterministic_action(x, state)
            step = env.step(action)
            observation = step.observation
            if step.done:
                break
        else:
            raise RuntimeError("HUD student real-chart episode exceeded step budget")
    return v054.StudentEvalResult(env.stats, env.physical_keydowns)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a v0.7.0 HUD-aware checkpoint without training or checkpoint writes."
    )
    parser.add_argument("checkpoint")
    parser.add_argument("chart")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--control-dt", type=float, default=None)
    hand_group = parser.add_mutually_exclusive_group()
    hand_group.add_argument("--same-hand", dest="same_hand_override", action="store_const", const=True, default=None)
    hand_group.add_argument("--cross-hand", dest="same_hand_override", action="store_const", const=False)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")

    device = torch.device("cpu")
    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise SystemExit("checkpoint payload is not a dictionary")
    if int(payload.get("format_version", -1)) != EXPECTED_FORMAT_VERSION:
        raise SystemExit(
            f"expected v0.7.0 format {EXPECTED_FORMAT_VERSION}, got {payload.get('format_version')}"
        )

    model = _load_hud_model(payload, device=device)
    same_hand, control_dt_s, hand_source, control_source = base._resolve_eval_config(
        payload,
        same_hand_override=args.same_hand_override,
        control_dt_override=args.control_dt,
    )
    compiled = load_compiled_adofai(args.chart)
    start_s, end_s = base._resolve_range(compiled.duration_s, args.start, args.end)
    segment = build_playable_segment(compiled, start_s=start_s, end_s=end_s)
    if not segment.targets:
        raise SystemExit("evaluation segment contains no playable targets")

    different_chart = base._is_different_chart(payload, args.chart)
    zero_shot_text = "yes" if different_chart is True else "no" if different_chart is False else "unknown"
    print("=== DMDOD / v0.7.0 HUD Checkpoint Evaluator ===")
    print(
        f"checkpoint={checkpoint_path} trainer={payload.get('trainer_version', '<unknown>')} "
        f"format={payload.get('format_version')} finalized={payload.get('finalized', '<unknown>')}"
    )
    print(f"model-source-chart={payload.get('chart', '<unknown>')}")
    print(f"target-chart={args.chart} different-chart={zero_shot_text}")
    print(
        f"segment={start_s:g}..{end_s:g}s targets={len(segment.targets)} "
        f"hidden={model.hidden_dim} input={HUD_REAL_CHART_INPUT_DIM}D"
    )
    print(
        f"HUD=TBPM+RBPM always, judgement/error transient | "
        f"body={'same-hand' if same_hand else 'cross-hand'}({hand_source}) "
        f"control={control_dt_s * 1000.0:.1f}ms({control_source}) training=DISABLED"
    )
    result = _evaluate_hud(
        model,
        segment,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        device=device,
    )
    print(v054._format_eval("student eval", result))


if __name__ == "__main__":
    main()
