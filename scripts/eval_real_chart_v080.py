from __future__ import annotations

import argparse
from pathlib import Path

import torch

import eval_real_chart as base
import eval_real_chart_v070 as v070_eval
import train_real_chart_v054 as v054
from fatal_diagnostics import FatalTrackingMixin, format_fatal_summary
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import load_compiled_adofai
from dmdod.real_chart_features import DEFAULT_REAL_CHART_FEATURE_CONFIG
from dmdod.real_chart_hud import DiagnosticHudRealChartMotorEnv
from dmdod.real_chart_hud_features import (
    HUD_REAL_CHART_INPUT_DIM,
    encode_hud_real_chart_observation,
)


EXPECTED_FORMAT_VERSION = 15


class FatalHudRealChartEnv(FatalTrackingMixin, DiagnosticHudRealChartMotorEnv):
    """v0.8 evaluation environment with non-invasive fatal diagnostics."""


def _evaluate_hud_with_fatal(
    model,
    segment,
    *,
    same_hand: bool,
    control_dt_s: float,
    device: torch.device,
):
    env = FatalHudRealChartEnv(
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
    return v054.StudentEvalResult(env.stats, env.physical_keydowns), env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a v0.8.0 multi-chart HUD checkpoint without training or writes."
    )
    parser.add_argument("checkpoint")
    parser.add_argument("chart")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--control-dt", type=float, default=None)
    hand_group = parser.add_mutually_exclusive_group()
    hand_group.add_argument(
        "--same-hand", dest="same_hand_override", action="store_const", const=True, default=None
    )
    hand_group.add_argument(
        "--cross-hand", dest="same_hand_override", action="store_const", const=False
    )
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
            f"expected v0.8.0 format {EXPECTED_FORMAT_VERSION}, got {payload.get('format_version')}"
        )

    model = v070_eval._load_hud_model(payload, device=device)
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

    dataset_signature = payload.get("dataset_signature")
    known_hashes: set[str] = set()
    if isinstance(dataset_signature, dict):
        for role in ("train", "validation", "final"):
            items = dataset_signature.get(role)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and isinstance(item.get("sha256"), str):
                        known_hashes.add(item["sha256"])

    print("=== DMDOD / v0.8.0 Multi-Chart HUD Evaluator ===")
    print(
        f"checkpoint={checkpoint_path} trainer={payload.get('trainer_version', '<unknown>')} "
        f"format={payload.get('format_version')} finalized={payload.get('finalized', '<unknown>')}"
    )
    print(
        f"source-dataset={payload.get('dataset_root', '<unknown>')} "
        f"known-chart-count={len(known_hashes)}"
    )
    print(f"target-chart={args.chart}")
    print(
        f"segment={start_s:g}..{end_s:g}s targets={len(segment.targets)} "
        f"hidden={model.hidden_dim} input={HUD_REAL_CHART_INPUT_DIM}D"
    )
    print(
        f"HUD=TBPM+RBPM always, judgement/error transient | "
        f"body={'same-hand' if same_hand else 'cross-hand'}({hand_source}) "
        f"control={control_dt_s * 1000.0:.1f}ms({control_source}) training=DISABLED"
    )
    result, env = _evaluate_hud_with_fatal(
        model,
        segment,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        device=device,
    )
    print(v054._format_eval("student eval", result))
    print(format_fatal_summary(env))


if __name__ == "__main__":
    main()
