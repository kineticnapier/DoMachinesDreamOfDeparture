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
    env = RhythmMotorEnv(
        targets,
        bpm=bpm,
        same_hand=same_hand,
        control_dt_s=control_dt,
    )
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
    print(f"  hits:       {stats.hits}/{stats.targets}")
    print(f"  misses:     {stats.misses}")
    print(f"  Perfect:    {stats.perfects}")
    print(f"  E/LPerfect: {stats.early_late_perfects}")
    print(f"  Early/Late: {stats.early_late_hits}")
    print(f"  Too Early:  {stats.too_early_presses}")
    print(f"  overload:   {stats.overload_counter}/6{'  OVERLOAD!' if stats.overloaded else ''}")
    print(f"  MAE:        {mae}")
    print(f"  P95:        {p95_text}")
    print(f"  reward:     {stats.total_reward:.3f}")


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
    target_notes = int(checkpoint.get("target_notes", notes))
    pattern = str(checkpoint.get("pattern", "left"))
    control_dt = float(checkpoint.get("control_dt", 0.010))
    base_start_s = float(checkpoint.get("start_s", 0.750))
    same_hand = pattern != "alternate" or bool(checkpoint.get("same_hand", False))

    probe_targets = make_regular_targets(
        bpm=bpm,
        count=max(1, notes),
        start_s=base_start_s,
        pattern=pattern,
    )
    probe_env = RhythmMotorEnv(probe_targets, bpm=bpm, same_hand=same_hand, control_dt_s=control_dt)
    windows = probe_env.timing_windows

    print("=== Deterministic Toy Policy Evaluation ===")
    print(f"checkpoint: {checkpoint_path}")
    print(
        f"task: {bpm:g} BPM, {notes} notes, pattern={pattern}, "
        f"control_dt={control_dt*1000:.1f} ms, start={base_start_s*1000:.0f} ms"
    )
    if bool(checkpoint.get("curriculum", False)):
        stage_index = checkpoint.get("curriculum_stage_index", "?")
        print(f"curriculum checkpoint: stage {stage_index}, {notes}/{target_notes} notes")
    print("action: tanh(actor mean), no sampling / exploration noise")
    print(
        f"Normal timing: Perfect ±{windows.perfect_s*1000:.2f} ms, "
        f"E/L Perfect ±{windows.early_late_perfect_s*1000:.2f} ms, "
        f"Pass ±{windows.pass_s*1000:.2f} ms"
    )
    print("OVERLOAD: Too Early +2, valid hit -1, fail at 6")
    if int(checkpoint.get("format_version", 0)) < 2:
        print("note: checkpoint predates ADOFAI timing/OVERLOAD training; this is an out-of-distribution rules test")
    if int(checkpoint.get("format_version", 0)) == 2:
        print("note: checkpoint predates deterministic-eval curriculum selection")
    print()

    exact = run_episode(
        model,
        device,
        bpm=bpm,
        notes=notes,
        pattern=pattern,
        same_hand=same_hand,
        control_dt=control_dt,
        start_s=base_start_s,
    )
    print_single("Exact checkpoint schedule:", exact)

    rng = random.Random(args.seed)
    runs: list[EvalRun] = []
    jitter_s = args.start_jitter_ms / 1000.0
    for _ in range(args.episodes):
        offset = rng.uniform(-jitter_s, jitter_s) if jitter_s > 0.0 else 0.0
        start_s = max(0.050, base_start_s + offset)
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
    total_too_early = sum(run.stats.too_early_presses for run in runs)
    overloads = sum(run.stats.overloaded for run in runs)
    all_errors = [error for run in runs for error in run.timing_errors_ms]
    full_hits = sum(
        run.stats.hits == run.stats.targets and not run.stats.overloaded
        for run in runs
    )
    zero_too_early = sum(run.stats.too_early_presses == 0 for run in runs)
    clean_clears = sum(
        run.stats.hits == run.stats.targets
        and run.stats.too_early_presses == 0
        and not run.stats.overloaded
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
        f"start offset ±{args.start_jitter_ms:g} ms around {base_start_s*1000:.0f} ms"
    )
    if args.start_jitter_ms == 0.0 and args.episodes > 1:
        print("  note: zero jitter repeats the same deterministic trajectory")
    print(f"  hit rate:       {total_hits}/{total_targets} = {total_hits/max(total_targets, 1):.4f}")
    print(f"  mean hits:      {total_hits/args.episodes:.3f}/{notes}")
    print(f"  mean misses:    {total_misses/args.episodes:.3f}")
    print(f"  mean Too Early: {total_too_early/args.episodes:.3f}")
    print(f"  OVERLOAD runs:  {overloads}/{args.episodes}")
    print(f"  full-hit runs:  {full_hits}/{args.episodes}")
    print(f"  zero Too Early: {zero_too_early}/{args.episodes}")
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
        help="uniformly vary chart start time around the checkpoint's training start",
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
