from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover - user-facing dependency message
    raise SystemExit(
        'PyTorch is required. Run: uv sync --extra dev --extra rl --inexact'
    ) from exc

from dmdod.rhythm_env import EpisodeStats, RhythmMotorEnv, make_regular_targets
from dmdod.toy_policy import ActorCritic, observation_tensor


@dataclass(frozen=True)
class EvalRun:
    stats: EpisodeStats
    timing_errors_ms: tuple[float, ...]


def p95_abs(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(abs(value) for value in values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def run_episode(
    model: ActorCritic,
    device: torch.device,
    *,
    bpm: float,
    notes: int,
    pattern: str,
    same_hand: bool,
    control_dt: float,
    start_s: float,
) -> EvalRun:
    targets = make_regular_targets(
        bpm=bpm,
        count=notes,
        start_s=start_s,
        pattern=pattern,
    )
    env = RhythmMotorEnv(targets, same_hand=same_hand, control_dt_s=control_dt)
    observation = env.reset()

    while True:
        x = observation_tensor(observation, device)
        action = model.deterministic_action(x)
        transition = env.step(action)
        observation = transition.observation
        if transition.done:
            break

    return EvalRun(env.stats, env.timing_errors_ms)


def print_single(label: str, run: EvalRun) -> None:
    stats = run.stats
    mae = "--" if stats.mean_abs_error_ms is None else f"{stats.mean_abs_error_ms:.2f} ms"
    p95 = p95_abs(list(run.timing_errors_ms))
    p95_text = "--" if p95 is None else f"{p95:.2f} ms"
    print(label)
    print(f"  hits:   {stats.hits}/{stats.targets}")
    print(f"  misses: {stats.misses}")
    print(f"  stray:  {stats.stray_presses}")
    print(f"  MAE:    {mae}")
    print(f"  P95:    {p95_text}")
    print(f"  reward: {stats.total_reward:.3f}")


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model"]
    hidden_dim = int(checkpoint.get("hidden_dim", state["backbone.0.weight"].shape[0]))
    model = ActorCritic(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(state)
    model.eval()

    bpm = float(checkpoint.get("bpm", 180.0))
    notes = int(checkpoint.get("notes", 16))
    pattern = str(checkpoint.get("pattern", "left"))
    control_dt = float(checkpoint.get("control_dt", 0.010))
    same_hand = pattern != "alternate" or bool(checkpoint.get("same_hand", False))

    print("=== Deterministic Toy Policy Evaluation ===")
    print(f"checkpoint: {checkpoint_path}")
    print(f"task: {bpm:g} BPM, {notes} notes, pattern={pattern}, control_dt={control_dt*1000:.1f} ms")
    print("action: tanh(actor mean), no sampling / exploration noise")
    if "format_version" not in checkpoint:
        print("note: legacy checkpoint; it predates exact pre-update best-policy saving")
    print()

    exact = run_episode(
        model,
        device,
        bpm=bpm,
        notes=notes,
        pattern=pattern,
        same_hand=same_hand,
        control_dt=control_dt,
        start_s=0.750,
    )
    print_single("Exact training schedule:", exact)

    rng = random.Random(args.seed)
    runs: list[EvalRun] = []
    jitter_s = args.start_jitter_ms / 1000.0
    for _ in range(args.episodes):
        offset = rng.uniform(-jitter_s, jitter_s) if jitter_s > 0.0 else 0.0
        start_s = max(0.050, 0.750 + offset)
        runs.append(
            run_episode(
                model,
                device,
                bpm=bpm,
                notes=notes,
                pattern=pattern,
                same_hand=same_hand,
                control_dt=control_dt,
                start_s=start_s,
            )
        )

    total_hits = sum(run.stats.hits for run in runs)
    total_targets = sum(run.stats.targets for run in runs)
    total_misses = sum(run.stats.misses for run in runs)
    total_strays = sum(run.stats.stray_presses for run in runs)
    all_errors = [error for run in runs for error in run.timing_errors_ms]
    full_hits = sum(run.stats.hits == run.stats.targets for run in runs)
    zero_stray = sum(run.stats.stray_presses == 0 for run in runs)
    clean_clears = sum(
        run.stats.hits == run.stats.targets and run.stats.stray_presses == 0
        for run in runs
    )
    mean_abs_error = (
        sum(abs(error) for error in all_errors) / len(all_errors)
        if all_errors
        else None
    )
    p95 = p95_abs(all_errors)

    print()
    print(
        f"Phase-jitter robustness: {args.episodes} deterministic episodes, "
        f"start offset ±{args.start_jitter_ms:g} ms"
    )
    if args.start_jitter_ms == 0.0 and args.episodes > 1:
        print("  note: zero jitter repeats the same deterministic trajectory")
    print(f"  hit rate:       {total_hits}/{total_targets} = {total_hits/max(total_targets, 1):.4f}")
    print(f"  mean hits:      {total_hits/args.episodes:.3f}/{notes}")
    print(f"  mean misses:    {total_misses/args.episodes:.3f}")
    print(f"  mean stray:     {total_strays/args.episodes:.3f}")
    print(f"  full-hit runs:  {full_hits}/{args.episodes}")
    print(f"  zero-stray:     {zero_stray}/{args.episodes}")
    print(f"  clean clears:   {clean_clears}/{args.episodes}")
    print(f"  MAE all hits:   {'--' if mean_abs_error is None else f'{mean_abs_error:.2f} ms'}")
    print(f"  P95 abs error:  {'--' if p95 is None else f'{p95:.2f} ms'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a toy motor policy without exploration noise.")
    parser.add_argument("--checkpoint", default="checkpoints/toy_policy.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--start-jitter-ms",
        type=float,
        default=100.0,
        help="uniformly vary chart start time to test cue dependence/generalization",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("episodes must be positive")
    if args.start_jitter_ms < 0.0:
        parser.error("start-jitter-ms must be non-negative")
    evaluate(args)


if __name__ == "__main__":
    main()
