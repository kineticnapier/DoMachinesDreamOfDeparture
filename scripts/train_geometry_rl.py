from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required. Run: uv sync --extra dev --extra rl --inexact"
    ) from exc

from dmdod.curriculum import curriculum_start_s, note_curriculum
from dmdod.geometry_rhythm_env import GeometryRhythmEnv
from dmdod.planet_perception import PlanetVisionConfig
from dmdod.rhythm_env import make_regular_targets
from dmdod.toy_policy import (
    ActorCritic,
    GEOMETRY_INPUT_DIM,
    discounted_returns,
    observation_tensor,
)


@dataclass(frozen=True)
class Probe:
    episodes: int
    hits: int
    targets: int
    full: int
    clean: int
    overloads: int
    too_early: int
    errors_ms: tuple[float, ...]

    @property
    def hit_rate(self) -> float:
        return self.hits / max(1, self.targets)

    @property
    def full_rate(self) -> float:
        return self.full / max(1, self.episodes)

    @property
    def clean_rate(self) -> float:
        return self.clean / max(1, self.episodes)

    @property
    def mae_ms(self) -> float | None:
        if not self.errors_ms:
            return None
        return sum(abs(x) for x in self.errors_ms) / len(self.errors_ms)


@dataclass(frozen=True)
class Retention:
    probes: tuple[tuple[int, Probe], ...]

    @property
    def overloads(self) -> int:
        return sum(probe.overloads for _, probe in self.probes)

    @property
    def min_hit_rate(self) -> float:
        return min((probe.hit_rate for _, probe in self.probes), default=1.0)

    @property
    def min_full_rate(self) -> float:
        return min((probe.full_rate for _, probe in self.probes), default=1.0)


def vision_config(args: argparse.Namespace) -> PlanetVisionConfig:
    return PlanetVisionConfig(
        latency_s=args.vision_latency_ms / 1000.0,
        latency_jitter_s=args.vision_latency_jitter_ms / 1000.0,
        sample_period_s=(1.0 / args.vision_hz if args.vision_hz > 0.0 else 0.0),
        position_noise_std=args.vision_noise_std,
        dropout_probability=args.vision_dropout,
    )


def vision_dict(config: PlanetVisionConfig) -> dict[str, float]:
    return {
        "latency_s": config.latency_s,
        "latency_jitter_s": config.latency_jitter_s,
        "sample_period_s": config.sample_period_s,
        "position_noise_std": config.position_noise_std,
        "dropout_probability": config.dropout_probability,
    }


def bpm_points(low: float, high: float, count: int) -> tuple[float, ...]:
    if abs(high - low) < 1e-12:
        return (low,)
    count = max(2, count)
    return tuple(low + (high - low) * i / (count - 1) for i in range(count))


def make_env(
    *,
    bpm: float,
    notes: int,
    start_s: float,
    control_dt: float,
    config: PlanetVisionConfig,
    seed: int,
) -> GeometryRhythmEnv:
    targets = make_regular_targets(
        bpm=bpm,
        count=notes,
        start_s=start_s,
        pattern="left",
    )
    return GeometryRhythmEnv(
        targets,
        bpm=bpm,
        same_hand=True,
        control_dt_s=control_dt,
        vision_config=config,
        perception_seed=seed,
    )


def deterministic_probe(
    model: ActorCritic,
    device: torch.device,
    *,
    bpms: tuple[float, ...],
    notes: int,
    base_start_s: float,
    control_dt: float,
    config: PlanetVisionConfig,
    episodes: int,
    phase_jitter_ms: float,
    seed_base: int,
) -> Probe:
    jitter_s = phase_jitter_ms / 1000.0
    offsets = (
        (0.0,)
        if episodes == 1
        else tuple(-jitter_s + 2.0 * jitter_s * i / (episodes - 1) for i in range(episodes))
    )

    hits = targets = full = clean = overloads = too_early = 0
    errors: list[float] = []
    was_training = model.training
    model.eval()

    for i, offset in enumerate(offsets):
        bpm = bpms[i % len(bpms)]
        env = make_env(
            bpm=bpm,
            notes=notes,
            start_s=max(0.050, base_start_s + offset),
            control_dt=control_dt,
            config=config,
            seed=seed_base + i * 1009,
        )
        observation = env.reset()
        while True:
            action = model.deterministic_action(observation_tensor(observation, device))
            transition = env.step(action)
            observation = transition.observation
            if transition.done:
                break

        stats = env.stats
        is_full = stats.hits == stats.targets and not stats.overloaded
        is_clean = is_full and stats.too_early_presses == 0
        hits += stats.hits
        targets += stats.targets
        full += int(is_full)
        clean += int(is_clean)
        overloads += int(stats.overloaded)
        too_early += stats.too_early_presses
        errors.extend(env.timing_errors_ms)

    if was_training:
        model.train()

    return Probe(
        episodes,
        hits,
        targets,
        full,
        clean,
        overloads,
        too_early,
        tuple(errors),
    )


def previous_stage_probes(
    model: ActorCritic,
    device: torch.device,
    *,
    previous_stages: tuple[int, ...],
    args: argparse.Namespace,
    eval_bpms: tuple[float, ...],
    config: PlanetVisionConfig,
) -> Retention:
    result: list[tuple[int, Probe]] = []
    for index, notes in enumerate(previous_stages):
        result.append(
            (
                notes,
                deterministic_probe(
                    model,
                    device,
                    bpms=eval_bpms,
                    notes=notes,
                    base_start_s=curriculum_start_s(notes),
                    control_dt=args.control_dt,
                    config=config,
                    episodes=args.retention_episodes,
                    phase_jitter_ms=args.eval_phase_jitter_ms,
                    seed_base=args.seed * 100000 + index * 10000 + notes,
                ),
            )
        )
    return Retention(tuple(result))


def passes(probe: Probe, retention: Retention, args: argparse.Namespace, notes: int) -> bool:
    if probe.overloads or retention.overloads:
        return False
    if retention.min_hit_rate < args.retention_hit_rate:
        return False
    if retention.min_full_rate < args.retention_full_rate:
        return False
    if notes == 1:
        return probe.clean_rate >= args.stage1_clean_rate
    return probe.hit_rate >= args.advance_hit_rate and probe.full_rate >= args.advance_full_rate


def rank_key(probe: Probe, retention: Retention) -> tuple[float, ...]:
    mae = probe.mae_ms
    return (
        1.0 if retention.overloads == 0 else 0.0,
        -float(retention.overloads),
        retention.min_hit_rate,
        retention.min_full_rate,
        1.0 if probe.overloads == 0 else 0.0,
        -float(probe.overloads),
        probe.hit_rate,
        probe.clean_rate,
        probe.full_rate,
        -(mae if mae is not None else float("inf")),
    )


def print_probe(label: str, probe: Probe) -> None:
    print(
        f"{label}hit={probe.hit_rate:.3f}  full={probe.full}/{probe.episodes}  "
        f"clean={probe.clean}/{probe.episodes}  ovl={probe.overloads}/{probe.episodes}  "
        f"early={probe.too_early/max(1, probe.episodes):.2f}/run  "
        f"MAE={'--' if probe.mae_ms is None else f'{probe.mae_ms:.1f}ms'}"
    )


def print_retention(retention: Retention) -> None:
    if retention.probes:
        print(
            "  retention: "
            + " | ".join(
                f"{notes}n hit={probe.hit_rate:.2f} full={probe.full_rate:.2f} ovl={probe.overloads}/{probe.episodes}"
                for notes, probe in retention.probes
            )
        )


def transplant_warm_start(model: ActorCritic, checkpoint: Path, device: torch.device) -> str:
    saved = torch.load(checkpoint, map_location=device)
    source = saved["model"]
    target = model.state_dict()
    copied: list[str] = []

    # Copy all shape-compatible learned body/action/value layers.
    for name, value in source.items():
        if name == "backbone.0.weight":
            continue
        if name in target and target[name].shape == value.shape:
            target[name] = value.clone()
            copied.append(name)

    # The old Gaussian model has 8 inputs (6 motor + 2 cue); geometry has
    # 10 inputs (6 motor + 4 positions). Preserve only the six motor columns.
    source_first = source.get("backbone.0.weight")
    target_first = target["backbone.0.weight"]
    if source_first is not None and source_first.shape[0] == target_first.shape[0]:
        if source_first.shape[1] == target_first.shape[1]:
            target_first.copy_(source_first)
            copied.append("backbone.0.weight(all)")
        elif source_first.shape[1] >= 6 and target_first.shape[1] >= 6:
            target_first[:, :6].copy_(source_first[:, :6])
            copied.append("backbone.0.weight(motor-columns)")

    model.load_state_dict(target)
    return ", ".join(copied)


def save_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
    config: PlanetVisionConfig,
    stage_index: int,
    stage_notes: int,
    global_episode: int,
    stage_episode: int,
    probe: Probe,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 7,
            "experiment": "planet-geometry-straight-v0.1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "input_dim": GEOMETRY_INPUT_DIM,
            "hidden_dim": 64,
            "bpm_min": args.bpm_min,
            "bpm_max": args.bpm_max,
            "notes": stage_notes,
            "target_notes": args.notes,
            "start_s": curriculum_start_s(stage_notes),
            "control_dt": args.control_dt,
            "vision_config": vision_dict(config),
            "curriculum_stage_index": stage_index,
            "curriculum_stage_notes": stage_notes,
            "global_episode": global_episode,
            "stage_episode": stage_episode,
            "probe": {
                "episodes": probe.episodes,
                "hit_rate": probe.hit_rate,
                "full_rate": probe.full_rate,
                "clean_rate": probe.clean_rate,
                "overloads": probe.overloads,
                "mae_ms": probe.mae_ms,
            },
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    config = vision_config(args)
    eval_bpms = bpm_points(args.bpm_min, args.bpm_max, args.eval_bpm_points)
    stages = note_curriculum(args.notes)

    model = ActorCritic(
        input_dim=GEOMETRY_INPUT_DIM,
        hidden_dim=64,
        initial_log_std=args.initial_log_std,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint = Path(args.checkpoint)

    global_episode = 0
    resume_stage_pos = 0
    if args.resume and args.warm_start:
        raise SystemExit("--resume and --warm-start are mutually exclusive")

    if args.warm_start:
        warm_path = Path(args.warm_start)
        if not warm_path.exists():
            raise SystemExit(f"warm-start checkpoint not found: {warm_path}")
        copied = transplant_warm_start(model, warm_path, device)
        print(f"warm-start: {warm_path}")
        print(f"warm-start copied: {copied}")
        print("warm-start note: Gaussian cue columns are NOT copied into geometry inputs")

    if args.resume:
        if not checkpoint.exists():
            raise SystemExit(f"resume checkpoint not found: {checkpoint}")
        saved = torch.load(checkpoint, map_location=device)
        if saved.get("experiment") != "planet-geometry-straight-v0.1":
            raise SystemExit("checkpoint is not a geometry-v0.1 experiment; use --warm-start")
        if int(saved.get("input_dim", -1)) != GEOMETRY_INPUT_DIM:
            raise SystemExit("geometry checkpoint input dimension mismatch")
        if abs(float(saved.get("bpm_min")) - args.bpm_min) > 1e-9 or abs(float(saved.get("bpm_max")) - args.bpm_max) > 1e-9:
            raise SystemExit("resume BPM domain mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        global_episode = int(saved.get("global_episode", 0))
        saved_notes = int(saved.get("curriculum_stage_notes", 1))
        if saved_notes not in stages:
            raise SystemExit("resume stage is not in current curriculum")
        resume_stage_pos = stages.index(saved_notes)
        print(f"resume: {checkpoint} at {saved_notes} note(s), global episode {global_episode}")

    print("=== Planet Geometry RL v0.1 ===")
    print("agent-visible: motor state + orbit(x,y) + next-tile vector(x,y)")
    print("hidden: timestamp, BPM, target angle, angle error, rotation-direction flag")
    print(f"BPM domain: {args.bpm_min:g}..{args.bpm_max:g}; probes: " + ", ".join(f"{x:g}" for x in eval_bpms))
    sample_text = "continuous" if config.sample_period_s <= 0 else f"{1.0/config.sample_period_s:.0f} Hz"
    print(
        f"vision: latency={config.latency_s*1000:.1f}±{config.latency_jitter_s*1000:.1f} ms, "
        f"sample={sample_text}, pos-noise={config.position_noise_std:.3f}, "
        f"dropout={config.dropout_probability*100:.1f}%"
    )
    print("curriculum: " + " -> ".join(str(x) for x in stages) + " notes")
    print()

    saved_probe: Probe | None = None
    saved_stage = stages[resume_stage_pos]
    reached_stage = saved_stage

    for stage_pos in range(resume_stage_pos, len(stages)):
        stage_notes = stages[stage_pos]
        stage_index = stage_pos + 1
        reached_stage = stage_notes
        previous = tuple(stages[:stage_pos])
        base_start = curriculum_start_s(stage_notes)
        best_key: tuple[float, ...] | None = None
        stage_passed = False

        print(f"--- stage {stage_index}/{len(stages)}: {stage_notes} note(s), start={base_start*1000:.0f} ms ---")
        probe = deterministic_probe(
            model,
            device,
            bpms=eval_bpms,
            notes=stage_notes,
            base_start_s=base_start,
            control_dt=args.control_dt,
            config=config,
            episodes=args.eval_episodes,
            phase_jitter_ms=args.eval_phase_jitter_ms,
            seed_base=args.seed * 1000000 + stage_index * 10000,
        )
        retention = previous_stage_probes(
            model,
            device,
            previous_stages=previous,
            args=args,
            eval_bpms=eval_bpms,
            config=config,
        )
        print_probe("  baseline: ", probe)
        print_retention(retention)
        best_key = rank_key(probe, retention)
        saved_probe = probe
        saved_stage = stage_notes
        save_checkpoint(
            checkpoint,
            model,
            optimizer,
            args=args,
            config=config,
            stage_index=stage_index,
            stage_notes=stage_notes,
            global_episode=global_episode,
            stage_episode=0,
            probe=probe,
        )
        if passes(probe, retention, args, stage_notes):
            stage_passed = True
            print("  stage already passed")
            print()
            if stage_pos == len(stages) - 1:
                break
            continue

        for stage_episode in range(1, args.episodes + 1):
            global_episode += 1
            rollout_notes = stage_notes
            replayed = False
            if previous and rng.random() < args.previous_stage_replay:
                rollout_notes = rng.choice(previous)
                replayed = True

            bpm = rng.uniform(args.bpm_min, args.bpm_max)
            base = curriculum_start_s(rollout_notes)
            start_s = max(
                0.050,
                base + rng.uniform(-args.train_phase_jitter_ms, args.train_phase_jitter_ms) / 1000.0,
            )
            env = make_env(
                bpm=bpm,
                notes=rollout_notes,
                start_s=start_s,
                control_dt=args.control_dt,
                config=config,
                seed=rng.randrange(0, 2**31),
            )
            observation = env.reset()
            rewards: list[float] = []
            log_probs: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []

            while True:
                x = observation_tensor(observation, device)
                action, log_prob, value, entropy = model.sample_action(x)
                transition = env.step(action)
                rewards.append(transition.reward)
                log_probs.append(log_prob)
                values.append(value)
                entropies.append(entropy)
                observation = transition.observation
                if transition.done:
                    break

            returns = discounted_returns(rewards, args.gamma, device)
            values_t = torch.stack(values)
            log_probs_t = torch.stack(log_probs)
            entropies_t = torch.stack(entropies)
            advantages = returns - values_t.detach()
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)

            policy_loss = -(log_probs_t * advantages).mean()
            value_loss = ((values_t - returns) ** 2).mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropies_t.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if stage_episode == 1 or stage_episode % args.log_every == 0 or stage_episode == args.episodes:
                stats = env.stats
                replay = f" replay={rollout_notes}n" if replayed else ""
                print(
                    f"ep {global_episode:4d} stage={stage_episode:3d}/{args.episodes}{replay} "
                    f"bpm={bpm:6.1f} reward={stats.total_reward:8.3f} "
                    f"hits={stats.hits:2d}/{stats.targets:2d} miss={stats.misses:2d} "
                    f"early={stats.too_early_presses:2d} ovl={stats.overload_counter} "
                    f"MAE={'--' if stats.mean_abs_error_ms is None else f'{stats.mean_abs_error_ms:5.1f}ms'} "
                    f"sigma={model.log_std.detach().exp().mean().item():.3f} loss={loss.item():8.4f}"
                )

            if stage_episode != 1 and stage_episode % args.eval_every != 0 and stage_episode != args.episodes:
                continue

            probe = deterministic_probe(
                model,
                device,
                bpms=eval_bpms,
                notes=stage_notes,
                base_start_s=base_start,
                control_dt=args.control_dt,
                config=config,
                episodes=args.eval_episodes,
                phase_jitter_ms=args.eval_phase_jitter_ms,
                seed_base=args.seed * 1000000 + stage_index * 10000,
            )
            retention = previous_stage_probes(
                model,
                device,
                previous_stages=previous,
                args=args,
                eval_bpms=eval_bpms,
                config=config,
            )
            print_probe("  eval: ", probe)
            print_retention(retention)

            key = rank_key(probe, retention)
            if key > best_key:
                best_key = key
                saved_probe = probe
                saved_stage = stage_notes
                save_checkpoint(
                    checkpoint,
                    model,
                    optimizer,
                    args=args,
                    config=config,
                    stage_index=stage_index,
                    stage_notes=stage_notes,
                    global_episode=global_episode,
                    stage_episode=stage_episode,
                    probe=probe,
                )
                print(f"  saved best -> {checkpoint}")

            if passes(probe, retention, args, stage_notes):
                stage_passed = True
                print("  stage passed")
                print()
                break

        if not stage_passed:
            print(f"curriculum stopped at {stage_notes} notes after {args.episodes} episodes")
            break
        if stage_pos == len(stages) - 1:
            break

    print()
    print(f"best checkpoint: {checkpoint}")
    print(f"furthest stage reached: {reached_stage}/{args.notes} notes")
    print(f"checkpoint stage: {saved_stage} notes")
    if saved_probe is not None:
        print_probe("deterministic best: ", saved_probe)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train from visible planet/tile geometry instead of a timing cue.")
    parser.add_argument("--episodes", type=int, default=500, help="maximum episodes per curriculum stage")
    parser.add_argument("--notes", type=int, default=16)
    parser.add_argument("--bpm-min", type=float, default=120.0)
    parser.add_argument("--bpm-max", type=float, default=300.0)
    parser.add_argument("--eval-bpm-points", type=int, default=5)
    parser.add_argument("--control-dt", type=float, default=0.010)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.002)
    parser.add_argument("--initial-log-std", type=float, default=-1.20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--retention-episodes", type=int, default=10)
    parser.add_argument("--train-phase-jitter-ms", type=float, default=100.0)
    parser.add_argument("--eval-phase-jitter-ms", type=float, default=100.0)
    parser.add_argument("--previous-stage-replay", type=float, default=0.20)
    parser.add_argument("--stage1-clean-rate", type=float, default=0.80)
    parser.add_argument("--advance-hit-rate", type=float, default=0.90)
    parser.add_argument("--advance-full-rate", type=float, default=0.80)
    parser.add_argument("--retention-hit-rate", type=float, default=0.70)
    parser.add_argument("--retention-full-rate", type=float, default=0.60)
    parser.add_argument("--vision-latency-ms", type=float, default=50.0)
    parser.add_argument("--vision-latency-jitter-ms", type=float, default=15.0)
    parser.add_argument("--vision-hz", type=float, default=60.0)
    parser.add_argument("--vision-noise-std", type=float, default=0.015)
    parser.add_argument("--vision-dropout", type=float, default=0.01)
    parser.add_argument("--warm-start", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", default="checkpoints/planet_geometry_v01.pt")
    args = parser.parse_args()

    if args.episodes <= 0 or args.notes <= 0:
        parser.error("episodes and notes must be positive")
    if args.bpm_min <= 0.0 or args.bpm_max < args.bpm_min:
        parser.error("BPM domain must satisfy 0 < bpm-min <= bpm-max")
    if args.eval_bpm_points <= 0 or args.eval_every <= 0 or args.eval_episodes <= 0:
        parser.error("evaluation settings must be positive")
    if args.retention_episodes <= 0 or args.log_every <= 0:
        parser.error("retention/log settings must be positive")
    if args.train_phase_jitter_ms < 0.0 or args.eval_phase_jitter_ms < 0.0:
        parser.error("phase jitter must be non-negative")
    if args.vision_latency_ms < 0.0 or args.vision_latency_jitter_ms < 0.0:
        parser.error("vision latency values must be non-negative")
    if args.vision_hz < 0.0 or args.vision_noise_std < 0.0:
        parser.error("vision frequency/noise must be non-negative")
    if not 0.0 <= args.vision_dropout <= 1.0:
        parser.error("vision-dropout must be between 0 and 1")
    for name in (
        "previous_stage_replay",
        "stage1_clean_rate",
        "advance_hit_rate",
        "advance_full_rate",
        "retention_hit_rate",
        "retention_full_rate",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"{name.replace('_', '-')} must be between 0 and 1")

    train(args)


if __name__ == "__main__":
    main()
