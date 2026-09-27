from __future__ import annotations

import argparse
import random
from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required. Run: uv sync --extra dev --extra rl --inexact"
    ) from exc

from dmdod.curriculum import curriculum_start_s
from dmdod.recurrent_policy import RecurrentActorCritic
from dmdod.toy_policy import GEOMETRY_INPUT_DIM
from eval_geometry_rl import config_from_dict, print_row, run_episode, summarize


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    path = Path(args.checkpoint)
    if not path.exists():
        raise SystemExit(f"checkpoint not found: {path}")
    saved = torch.load(path, map_location=device)
    if saved.get("experiment") != "planet-geometry-ppo-v0.3":
        raise SystemExit("checkpoint is not planet-geometry-ppo-v0.3")

    model = RecurrentActorCritic(
        input_dim=int(saved.get("input_dim", GEOMETRY_INPUT_DIM)),
        hidden_dim=int(saved.get("hidden_dim", 64)),
        initial_log_std=float(saved.get("initial_log_std", -0.70)),
    ).to(device)
    model.load_state_dict(saved["model"])
    model.eval()

    raw_phase = saved.get("curriculum_phase", {})
    if not isinstance(raw_phase, dict):
        raise SystemExit("checkpoint has no curriculum phase metadata")
    notes = int(raw_phase.get("notes", 1))
    low = float(raw_phase.get("bpm_min", saved.get("final_bpm_min", 120.0)))
    high = float(raw_phase.get("bpm_max", saved.get("final_bpm_max", 300.0)))
    phase_jitter_ms = float(raw_phase.get("eval_phase_jitter_ms", 100.0))
    config = config_from_dict(raw_phase.get("vision_config", {}))
    control_dt = float(saved.get("control_dt", 0.010))
    base_start = curriculum_start_s(notes)
    phase_index = int(saved.get("curriculum_phase_index", 1))
    phase_name = str(saved.get("curriculum_phase_name", "unknown"))
    jitter_ms = phase_jitter_ms if args.phase_jitter_ms is None else args.phase_jitter_ms

    print("=== Planet Geometry PPO Policy Evaluation ===")
    print(f"checkpoint: {path}")
    print(f"phase: {phase_index} ({phase_name})")
    print(f"task: {notes} notes, BPM domain={low:g}..{high:g}, control_dt={control_dt*1000:.1f} ms")
    print("policy: recurrent GRU trained with clipped PPO")
    print("agent-visible: motor + orbit(x,y) + next-tile vector(x,y)")
    print("no exact time/BPM/target-angle/angle-error/direction flag")

    rng = random.Random(args.seed)
    starts = [
        max(0.050, base_start + rng.uniform(-jitter_ms, jitter_ms) / 1000.0)
        for _ in range(args.episodes)
    ]
    bpms = [rng.uniform(low, high) if high > low else low for _ in range(args.episodes)]

    def batch(mode: str):
        return summarize(
            [
                run_episode(
                    model,
                    device,
                    bpm=bpm,
                    notes=notes,
                    start_s=start,
                    control_dt=control_dt,
                    config=config,
                    seed=args.seed + i * 1009,
                    mode=mode,
                )
                for i, (bpm, start) in enumerate(zip(bpms, starts))
            ]
        )

    print()
    print(f"Domain robustness: {args.episodes} episodes with random BPM/phase/sensor seeds")
    normal = batch("normal")
    print_row("normal", normal)

    if args.ablation:
        print()
        print("Geometry ablation on identical BPM/phase/sensor schedule")
        print_row("normal", normal)
        print_row("zero", batch("zero"))
        print_row("random", batch("random"))
        print_row("freeze", batch("freeze"))

    if args.bpm_sweep:
        print()
        print(f"BPM sweep: {args.sweep_episodes} episodes/point")
        points = [low] if abs(high - low) < 1e-12 else [low + (high - low) * i / 4.0 for i in range(5)]
        for point_index, bpm in enumerate(points):
            runs = []
            for i in range(args.sweep_episodes):
                local_rng = random.Random(args.seed + point_index * 100000 + i)
                start = max(
                    0.050,
                    base_start + local_rng.uniform(-jitter_ms, jitter_ms) / 1000.0,
                )
                runs.append(
                    run_episode(
                        model,
                        device,
                        bpm=bpm,
                        notes=notes,
                        start_s=start,
                        control_dt=control_dt,
                        config=config,
                        seed=args.seed + point_index * 10000 + i * 1009,
                        mode="normal",
                    )
                )
            print_row(f"{bpm:g} BPM", summarize(runs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate geometry PPO v0.3.")
    parser.add_argument("--checkpoint", default="checkpoints/planet_geometry_v03_ppo.pt")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--phase-jitter-ms", type=float, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--bpm-sweep", action="store_true")
    parser.add_argument("--sweep-episodes", type=int, default=20)
    args = parser.parse_args()
    if args.episodes <= 0 or args.sweep_episodes <= 0:
        parser.error("episode counts must be positive")
    if args.phase_jitter_ms is not None and args.phase_jitter_ms < 0.0:
        parser.error("phase-jitter-ms must be non-negative")
    evaluate(args)


if __name__ == "__main__":
    main()
