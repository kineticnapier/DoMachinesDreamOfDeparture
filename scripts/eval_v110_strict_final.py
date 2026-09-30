from __future__ import annotations

"""Evaluate a finalized v1.1-style multi-chart checkpoint on Final charts using Strict timing.

This is evaluation-only. It does not alter the checkpoint, training selection, or
Normal-difficulty metrics saved by the trainer.
"""

import argparse
from pathlib import Path

import torch

import train_real_chart_v054 as v054
import train_real_chart_v070 as v070
import train_real_chart_v080 as v080
from dmdod.adofai_rules import TimingDifficulty
from dmdod.multichart_dataset import discover_multichart_dataset
from dmdod.real_chart_features import DEFAULT_REAL_CHART_FEATURE_CONFIG
from dmdod.real_chart_hud import DiagnosticHudRealChartMotorEnv
from dmdod.real_chart_hud_features import HUD_REAL_CHART_INPUT_DIM, encode_hud_real_chart_observation
from dmdod.recurrent_policy import RecurrentActorCritic


def _evaluate_strict(
    model: RecurrentActorCritic,
    segment,
    *,
    same_hand: bool,
    control_dt_s: float,
    device: torch.device,
) -> v054.StudentEvalResult:
    env = DiagnosticHudRealChartMotorEnv(
        segment,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        difficulty=TimingDifficulty.STRICT,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )
    observation = env.reset()
    state = model.initial_state(device)
    max_steps = int((segment.duration_s + 2.0) / control_dt_s) + 200

    model.eval()
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
            raise RuntimeError("strict FINAL evaluation exceeded step budget")

    return v054.StudentEvalResult(env.stats, int(env.physical_keydowns))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Final holdout charts with ADOFAI Strict timing windows."
    )
    parser.add_argument("dataset")
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    device = torch.device("cpu")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_dim = int(payload.get("hidden_dim", -1))
    input_dim = int(payload.get("input_dim", -1))
    if hidden_dim <= 0:
        raise SystemExit("checkpoint is missing a valid hidden_dim")
    if input_dim != HUD_REAL_CHART_INPUT_DIM:
        raise SystemExit(
            f"checkpoint input dimension {input_dim} != HUD input {HUD_REAL_CHART_INPUT_DIM}"
        )
    if "model_state" not in payload:
        raise SystemExit("checkpoint is missing model_state")

    signature = payload.get("signature") or {}
    same_hand = bool(signature.get("same_hand", True))
    control_dt_s = float(signature.get("control_dt", 0.010))

    # Keep the same HUD installation used by the multi-chart trainer.
    v070._install_v070()

    try:
        dataset = discover_multichart_dataset(args.dataset)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    final_charts = v080._compile_role(dataset.final)
    final_segments = v080._full_segments(final_charts, "strict-final-holdout")

    model = RecurrentActorCritic(
        input_dim=HUD_REAL_CHART_INPUT_DIM,
        hidden_dim=hidden_dim,
        initial_log_std=-1.20,
    ).to(device)
    model.load_state_dict(payload["model_state"])

    print("=== DMDOD Strict FINAL holdout ===")
    print(
        f"checkpoint={checkpoint_path} difficulty=STRICT hidden={hidden_dim} "
        f"control={control_dt_s * 1000.0:.1f}ms same-hand={same_hand}"
    )

    results: list[v054.StudentEvalResult] = []
    for named in final_segments:
        result = _evaluate_strict(
            model,
            named.segment,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
            device=device,
        )
        results.append(result)
        print(v054._format_eval(f"STRICT FINAL holdout [{named.chart_name}]", result))

    hits, targets, mean_xacc, overloaded = v080.v063._summary_metrics(results)
    print(
        f"STRICT FINAL holdout aggregate: H{hits}/{targets} "
        f"meanX{mean_xacc:.1f}% over={overloaded}"
    )


if __name__ == "__main__":
    main()
