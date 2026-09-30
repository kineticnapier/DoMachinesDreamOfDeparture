from __future__ import annotations

"""Benchmark only the v1.0+ start-micro gameplay evaluation path.

This intentionally excludes BC and long anchor/validation evaluation.  Each
repeat clears the exact evaluation cache so every start-micro segment is really
simulated, while the process pool is kept alive across repeats.  That makes run
1 a cold-process measurement and later runs representative of steady-state
bootstrap evaluation.
"""

import argparse
import statistics
import time
from pathlib import Path

import torch

import train_real_chart_v080 as v080
import train_real_chart_v100_start_micro as v100
from dmdod.multichart_dataset import discover_multichart_dataset


def _checkpoint_eval_config(payload: dict) -> tuple[bool, float, int]:
    signature = payload.get("signature")
    if not isinstance(signature, dict):
        signature = {}
    same_hand = bool(signature.get("same_hand", True))
    control_dt_s = float(signature.get("control_dt", 0.010))
    hidden_dim = int(payload.get("hidden_dim", 128))
    if control_dt_s <= 0.0:
        raise SystemExit("checkpoint control_dt must be positive")
    if hidden_dim <= 0:
        raise SystemExit("checkpoint hidden_dim must be positive")
    return same_hand, control_dt_s, hidden_dim


def _build_start_micro_segments(train_charts, *, target_count: int):
    segments = []
    for chart in train_charts:
        end_s = v100._start_micro_end_s(chart, target_count)
        segments.append(
            v080._named_segment(
                chart,
                f"start-micro-{target_count}",
                0.0,
                end_s,
            )
        )
    return segments


def _force_cache_miss() -> None:
    v080._EVAL_CACHE.clear()
    v080._EVAL_CACHE_HITS = 0
    v080._EVAL_CACHE_MISSES = 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the 48-ish start-micro bootstrap guard segments only."
    )
    parser.add_argument("dataset")
    parser.add_argument("checkpoint")
    parser.add_argument("--start-micro-targets", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    if args.start_micro_targets <= 0:
        raise SystemExit("--start-micro-targets must be positive")
    if args.repeat <= 0:
        raise SystemExit("--repeat must be positive")

    checkpoint_path = Path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise SystemExit("checkpoint payload is not a dictionary")
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise SystemExit("checkpoint has no model_state")

    same_hand, control_dt_s, hidden_dim = _checkpoint_eval_config(payload)
    input_dim = int(payload.get("input_dim", v080.v070.HUD_REAL_CHART_INPUT_DIM))
    if input_dim != v080.v070.HUD_REAL_CHART_INPUT_DIM:
        raise SystemExit(
            f"checkpoint input_dim={input_dim} is not the 245D HUD expected by start-micro"
        )

    dataset = discover_multichart_dataset(args.dataset)
    train_charts = v080._compile_role(dataset.train)
    start_segments = _build_start_micro_segments(
        train_charts,
        target_count=args.start_micro_targets,
    )
    if not start_segments:
        raise SystemExit("dataset contains no Train charts")

    model = v080.RecurrentActorCritic(
        input_dim=v080.v070.HUD_REAL_CHART_INPUT_DIM,
        hidden_dim=hidden_dim,
        initial_log_std=-1.20,
    ).cpu()
    model.load_state_dict(state)
    model.eval()

    workers = v080._configured_workers()
    target_total = sum(len(named.segment.targets) for named in start_segments)
    duration_total = sum(float(named.segment.duration_s) for named in start_segments)
    print("=== DMDOD start-micro eval benchmark ===")
    print(
        f"charts={len(start_segments)} targets={target_total} "
        f"simulated-duration={duration_total:.3f}s workers={workers} "
        f"control={control_dt_s * 1000.0:.1f}ms same-hand={same_hand}"
    )
    print("BC=excluded long-anchors=excluded validation=excluded cache=forced-miss")

    elapsed_runs: list[float] = []
    try:
        for run in range(1, args.repeat + 1):
            _force_cache_miss()
            started = time.perf_counter()
            raw = v080._evaluate_states_on_segments(
                model,
                {0.0: state},
                start_segments,
                same_hand=same_hand,
                control_dt_s=control_dt_s,
            )
            elapsed = time.perf_counter() - started
            elapsed_runs.append(elapsed)

            evaluations = [raw[(0.0, named.key)] for named in start_segments]
            early = sum(int(item.stats.too_early_presses) for item in evaluations)
            hits = sum(int(item.stats.hits) for item in evaluations)
            targets = sum(int(item.stats.targets) for item in evaluations)
            label = "cold" if run == 1 else "steady"
            print(
                f"run {run}/{args.repeat} [{label}]: {elapsed:.3f}s "
                f"H={hits}/{targets} start-early={early} "
                f"cache-miss={v080._EVAL_CACHE_MISSES}"
            )
    finally:
        v080._shutdown_pool()

    if len(elapsed_runs) > 1:
        steady = elapsed_runs[1:]
        print(
            f"steady median={statistics.median(steady):.3f}s "
            f"min={min(steady):.3f}s max={max(steady):.3f}s"
        )
    else:
        print(f"single-run={elapsed_runs[0]:.3f}s")


if __name__ == "__main__":
    main()
