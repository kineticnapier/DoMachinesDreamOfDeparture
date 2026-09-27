from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required. Run: uv sync --extra dev --extra rl --inexact"
    ) from exc

from dmdod.curriculum import curriculum_start_s
from dmdod.geometry_rhythm_env import GeometryRhythmEnv, GeometryRhythmObservation
from dmdod.planet_perception import PlanetGeometryObservation, PlanetVisionConfig
from dmdod.recurrent_policy import RecurrentActorCritic
from dmdod.rhythm_env import make_regular_targets
from dmdod.toy_policy import GEOMETRY_INPUT_DIM, observation_tensor


@dataclass(frozen=True)
class Run:
    hits: int
    targets: int
    misses: int
    early: int
    overloaded: bool
    errors_ms: tuple[float, ...]


@dataclass(frozen=True)
class Summary:
    episodes: int
    hits: int
    targets: int
    full: int
    clean: int
    overloads: int
    early: int
    errors_ms: tuple[float, ...]

    @property
    def hit_rate(self) -> float:
        return self.hits / max(1, self.targets)

    @property
    def mae(self) -> float | None:
        if not self.errors_ms:
            return None
        return sum(abs(x) for x in self.errors_ms) / len(self.errors_ms)

    @property
    def mean_error(self) -> float | None:
        if not self.errors_ms:
            return None
        return sum(self.errors_ms) / len(self.errors_ms)

    @property
    def p95(self) -> float | None:
        if not self.errors_ms:
            return None
        ordered = sorted(abs(x) for x in self.errors_ms)
        return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def config_from_dict(raw: object) -> PlanetVisionConfig:
    if not isinstance(raw, dict):
        raw = {}
    return PlanetVisionConfig(
        latency_s=float(raw.get("latency_s", 0.0)),
        latency_jitter_s=float(raw.get("latency_jitter_s", 0.0)),
        sample_period_s=float(raw.get("sample_period_s", 0.0)),
        position_noise_std=float(raw.get("position_noise_std", 0.0)),
        dropout_probability=float(raw.get("dropout_probability", 0.0)),
    )


def transform_geometry(
    observation: GeometryRhythmObservation,
    *,
    mode: str,
    rng: random.Random,
    frozen: PlanetGeometryObservation | None,
) -> tuple[GeometryRhythmObservation, PlanetGeometryObservation | None]:
    geometry = observation.geometry
    if mode == "normal":
        return observation, frozen
    if mode == "zero":
        geometry = PlanetGeometryObservation(0.0, 0.0, 0.0, 0.0)
    elif mode == "random":
        geometry = PlanetGeometryObservation(
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
        )
    elif mode == "freeze":
        if frozen is None:
            frozen = geometry
        geometry = frozen
    else:
        raise ValueError(f"unknown ablation mode: {mode}")
    return GeometryRhythmObservation(observation.motor, geometry), frozen


def run_episode(
    model: RecurrentActorCritic,
    device: torch.device,
    *,
    bpm: float,
    notes: int,
    start_s: float,
    control_dt: float,
    config: PlanetVisionConfig,
    seed: int,
    mode: str,
) -> Run:
    targets = make_regular_targets(bpm=bpm, count=notes, start_s=start_s, pattern="left")
    env = GeometryRhythmEnv(
        targets,
        bpm=bpm,
        same_hand=True,
        control_dt_s=control_dt,
        vision_config=config,
        perception_seed=seed,
    )
    rng = random.Random(seed ^ 0x5EED5EED)
    frozen = None
    observation, frozen = transform_geometry(env.reset(), mode=mode, rng=rng, frozen=frozen)
    state = model.initial_state(device)

    while True:
        action, state = model.deterministic_action(
            observation_tensor(observation, device), state
        )
        transition = env.step(action)
        observation, frozen = transform_geometry(
            transition.observation,
            mode=mode,
            rng=rng,
            frozen=frozen,
        )
        if transition.done:
            break

    stats = env.stats
    return Run(
        stats.hits,
        stats.targets,
        stats.misses,
        stats.too_early_presses,
        stats.overloaded,
        env.timing_errors_ms,
    )


def summarize(runs: list[Run]) -> Summary:
    return Summary(
        episodes=len(runs),
        hits=sum(x.hits for x in runs),
        targets=sum(x.targets for x in runs),
        full=sum(x.hits == x.targets and not x.overloaded for x in runs),
        clean=sum(x.hits == x.targets and x.early == 0 and not x.overloaded for x in runs),
        overloads=sum(x.overloaded for x in runs),
        early=sum(x.early for x in runs),
        errors_ms=tuple(error for x in runs for error in x.errors_ms),
    )


def fmt(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "--"
    return f"{value:+.1f}" if signed else f"{value:.1f}"


def print_row(label: str, summary: Summary) -> None:
    print(
        f"  {label:<14} hit={summary.hit_rate:6.3f} "
        f"full={summary.full:3d}/{summary.episodes:<3d} "
        f"clean={summary.clean:3d}/{summary.episodes:<3d} "
        f"ovl={summary.overloads:3d}/{summary.episodes:<3d} "
        f"early={summary.early/max(1, summary.episodes):5.2f} "
        f"meanErr={fmt(summary.mean_error, signed=True):>6}ms "
        f"MAE={fmt(summary.mae):>5}ms P95={fmt(summary.p95):>5}ms"
    )


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    path = Path(args.checkpoint)
    if not path.exists():
        raise SystemExit(f"checkpoint not found: {path}")
    saved = torch.load(path, map_location=device)
    if saved.get("experiment") != "planet-geometry-sequence-v0.2":
        raise SystemExit("checkpoint is not planet-geometry-sequence-v0.2")

    input_dim = int(saved.get("input_dim", GEOMETRY_INPUT_DIM))
    hidden_dim = int(saved.get("hidden_dim", 64))
    model = RecurrentActorCritic(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
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
    phase_jitter_ms = float(raw_phase.get("eval_phase_jitter_ms", args.phase_jitter_ms))
    config = config_from_dict(raw_phase.get("vision_config", {}))
    control_dt = float(saved.get("control_dt", 0.010))
    base_start = curriculum_start_s(notes)
    phase_index = int(saved.get("curriculum_phase_index", 1))
    phase_name = str(saved.get("curriculum_phase_name", "unknown"))

    print("=== Planet Geometry Sequence Policy Evaluation ===")
    print(f"checkpoint: {path}")
    print(f"phase: {phase_index} ({phase_name})")
    print(f"task: {notes} notes, BPM domain={low:g}..{high:g}, control_dt={control_dt*1000:.1f} ms")
    print("policy: GRU memory reset at each episode")
    print("agent-visible: motor + orbit(x,y) + next-tile vector(x,y)")
    print("no exact time/BPM/target-angle/angle-error/direction flag")

    rng = random.Random(args.seed)
    jitter_ms = phase_jitter_ms if args.phase_jitter_ms is None else args.phase_jitter_ms
    starts = [
        max(0.050, base_start + rng.uniform(-jitter_ms, jitter_ms) / 1000.0)
        for _ in range(args.episodes)
    ]
    bpms = [rng.uniform(low, high) if high > low else low for _ in range(args.episodes)]

    def batch(mode: str) -> Summary:
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
        points = (
            [low]
            if abs(high - low) < 1e-12
            else [low + (high - low) * i / 4.0 for i in range(5)]
        )
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
    parser = argparse.ArgumentParser(description="Evaluate the recurrent planet-geometry policy.")
    parser.add_argument("--checkpoint", default="checkpoints/planet_geometry_v02.pt")
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
