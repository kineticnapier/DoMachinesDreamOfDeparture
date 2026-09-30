from __future__ import annotations

"""Evaluate a v1.1-style checkpoint on any dataset role using Strict timing.

This is evaluation-only. It loads difficulty labels from the balanced TUF
manifest so P7/P8/P9/P10 can be reported separately without changing or
retraining the checkpoint.
"""

import argparse
import json
import re
from dataclasses import dataclass
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


_P_LABEL_RE = re.compile(r"^P([1-9][0-9]*)$", re.IGNORECASE)
_ROLE_CHOICES = ("Train", "Validation", "Final")


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    charts: int
    hits: int
    targets: int
    misses: int
    x_accuracy_percent: float
    perfect_rate: float
    mean_abs_error_ms: float | None
    too_early_presses: int
    overloaded: bool
    physical_keydowns: int


def _difficulty_sort_key(label: str) -> tuple[int, int | str]:
    match = _P_LABEL_RE.fullmatch(label.strip())
    if match is not None:
        return 0, int(match.group(1))
    return 1, label.casefold()


def _load_difficulty_by_chart_name(dataset_root: str | Path, role: str) -> dict[str, str]:
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("difficulty aggregation requires a directory dataset with manifest.json")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"dataset manifest not found: {manifest_path}")

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read dataset manifest: {manifest_path}: {exc}") from exc

    entries = payload.get(role)
    if not isinstance(entries, list):
        raise ValueError(f"manifest has no list for role {role}")

    result: dict[str, str] = {}
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError(f"manifest {role} entry is not an object")
        raw_path = item.get("path")
        raw_difficulty = item.get("difficulty_name")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"manifest {role} entry is missing path")
        if not isinstance(raw_difficulty, str) or not raw_difficulty.strip():
            raise ValueError(f"manifest {role} entry {raw_path!r} is missing difficulty_name")

        # Manifests can be generated on Windows and later inspected elsewhere.
        chart_name = Path(raw_path.replace("\\", "/")).stem
        if chart_name in result:
            raise ValueError(f"duplicate chart name in manifest {role}: {chart_name}")
        result[chart_name] = raw_difficulty.strip()
    return result


def _aggregate(results: list[v054.StudentEvalResult]) -> AggregateMetrics:
    if not results:
        raise ValueError("cannot aggregate an empty result set")

    hits = sum(int(result.stats.hits) for result in results)
    targets = sum(int(result.stats.targets) for result in results)
    misses = sum(int(result.stats.misses) for result in results)
    early = sum(int(result.stats.too_early_presses) for result in results)
    keydowns = sum(int(result.physical_keydowns) for result in results)
    overloaded = any(bool(result.stats.overloaded) for result in results)

    x_points = sum(float(result.stats.x_accuracy_points) for result in results)
    x_denominator = sum(int(result.stats.x_accuracy_denominator) for result in results)
    x_accuracy = 0.0 if x_denominator <= 0 else 100.0 * x_points / x_denominator

    perfects = sum(int(result.stats.perfects) for result in results)
    margin_count = sum(int(result.stats.hit_margin_count) for result in results)
    perfect_rate = 0.0 if margin_count <= 0 else perfects / margin_count

    mae_numerator = 0.0
    mae_denominator = 0
    for result in results:
        mae = result.stats.mean_abs_error_ms
        result_hits = int(result.stats.hits)
        if mae is not None and result_hits > 0:
            mae_numerator += float(mae) * result_hits
            mae_denominator += result_hits
    mean_abs_error_ms = None if mae_denominator == 0 else mae_numerator / mae_denominator

    return AggregateMetrics(
        charts=len(results),
        hits=hits,
        targets=targets,
        misses=misses,
        x_accuracy_percent=x_accuracy,
        perfect_rate=perfect_rate,
        mean_abs_error_ms=mean_abs_error_ms,
        too_early_presses=early,
        overloaded=overloaded,
        physical_keydowns=keydowns,
    )


def _format_aggregate(label: str, metrics: AggregateMetrics) -> str:
    mae_text = "nan" if metrics.mean_abs_error_ms is None else f"{metrics.mean_abs_error_ms:.2f}"
    return (
        f"{label}: charts={metrics.charts} H={metrics.hits}/{metrics.targets} "
        f"miss={metrics.misses} X={metrics.x_accuracy_percent:.2f}% "
        f"PP={metrics.perfect_rate * 100.0:.1f}% MAE={mae_text}ms "
        f"early={metrics.too_early_presses} overload={metrics.overloaded} "
        f"keydowns={metrics.physical_keydowns}"
    )


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
            raise RuntimeError("strict dataset evaluation exceeded step budget")

    return v054.StudentEvalResult(env.stats, int(env.physical_keydowns))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Train, Validation, or Final charts with ADOFAI Strict timing "
            "and print per-difficulty aggregates from the dataset manifest."
        )
    )
    parser.add_argument("dataset")
    parser.add_argument("checkpoint")
    parser.add_argument("--role", choices=_ROLE_CHOICES, default="Validation")
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

    v070._install_v070()

    try:
        dataset = discover_multichart_dataset(args.dataset)
        difficulty_by_name = _load_difficulty_by_chart_name(args.dataset, args.role)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    role_items = getattr(dataset, args.role.casefold())
    compiled = v080._compile_role(role_items)
    segments = v080._full_segments(compiled, f"strict-{args.role.casefold()}")

    missing = [named.chart_name for named in segments if named.chart_name not in difficulty_by_name]
    if missing:
        raise SystemExit(
            f"manifest {args.role} difficulty missing for {len(missing)} chart(s): "
            + ", ".join(missing[:5])
        )

    model = RecurrentActorCritic(
        input_dim=HUD_REAL_CHART_INPUT_DIM,
        hidden_dim=hidden_dim,
        initial_log_std=-1.20,
    ).to(device)
    model.load_state_dict(payload["model_state"])

    print(f"=== DMDOD Strict {args.role} dataset eval ===")
    print(
        f"dataset={Path(args.dataset)} checkpoint={checkpoint_path} difficulty=STRICT "
        f"charts={len(segments)} hidden={hidden_dim} "
        f"control={control_dt_s * 1000.0:.1f}ms same-hand={same_hand}"
    )

    by_difficulty: dict[str, list[v054.StudentEvalResult]] = {}
    all_results: list[v054.StudentEvalResult] = []

    for index, named in enumerate(segments, 1):
        difficulty = difficulty_by_name[named.chart_name]
        result = _evaluate_strict(
            model,
            named.segment,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
            device=device,
        )
        all_results.append(result)
        by_difficulty.setdefault(difficulty, []).append(result)
        print(
            v054._format_eval(
                f"STRICT {args.role} {index:02d}/{len(segments):02d} "
                f"[{difficulty} | {named.chart_name}]",
                result,
            )
        )

    print("\n=== Difficulty aggregates ===")
    for difficulty in sorted(by_difficulty, key=_difficulty_sort_key):
        print(
            _format_aggregate(
                f"STRICT {args.role} aggregate [{difficulty}]",
                _aggregate(by_difficulty[difficulty]),
            )
        )

    print(_format_aggregate(f"STRICT {args.role} aggregate [ALL]", _aggregate(all_results)))


if __name__ == "__main__":
    main()
