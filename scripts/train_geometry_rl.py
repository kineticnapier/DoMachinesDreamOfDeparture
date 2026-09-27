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
from dmdod.recurrent_policy import RecurrentActorCritic
from dmdod.rhythm_env import make_regular_targets
from dmdod.toy_policy import GEOMETRY_INPUT_DIM, discounted_returns, observation_tensor


@dataclass(frozen=True)
class CurriculumPhase:
    name: str
    notes: int
    bpm_min: float
    bpm_max: float
    train_phase_jitter_ms: float
    eval_phase_jitter_ms: float
    vision: PlanetVisionConfig


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


def final_vision_config(args: argparse.Namespace) -> PlanetVisionConfig:
    return PlanetVisionConfig(
        latency_s=args.vision_latency_ms / 1000.0,
        latency_jitter_s=args.vision_latency_jitter_ms / 1000.0,
        sample_period_s=(1.0 / args.vision_hz if args.vision_hz > 0.0 else 0.0),
        position_noise_std=args.vision_noise_std,
        dropout_probability=args.vision_dropout,
    )


def clean_vision_config() -> PlanetVisionConfig:
    return PlanetVisionConfig(
        latency_s=0.0,
        latency_jitter_s=0.0,
        sample_period_s=0.0,
        position_noise_std=0.0,
        dropout_probability=0.0,
    )


def sampled_vision_config(args: argparse.Namespace) -> PlanetVisionConfig:
    final = final_vision_config(args)
    return PlanetVisionConfig(
        latency_s=final.latency_s,
        latency_jitter_s=final.latency_jitter_s,
        sample_period_s=final.sample_period_s,
        position_noise_std=0.0,
        dropout_probability=0.0,
    )


def vision_dict(config: PlanetVisionConfig) -> dict[str, float]:
    return {
        "latency_s": config.latency_s,
        "latency_jitter_s": config.latency_jitter_s,
        "sample_period_s": config.sample_period_s,
        "position_noise_std": config.position_noise_std,
        "dropout_probability": config.dropout_probability,
    }


def build_curriculum(args: argparse.Namespace) -> tuple[CurriculumPhase, ...]:
    final = final_vision_config(args)
    clean = clean_vision_config()
    sampled = sampled_vision_config(args)
    anchor = min(max(180.0, args.bpm_min), args.bpm_max)
    expansion = 0.35
    narrow_low = anchor + (args.bpm_min - anchor) * expansion
    narrow_high = anchor + (args.bpm_max - anchor) * expansion
    narrow_jitter = min(50.0, args.train_phase_jitter_ms)
    narrow_eval_jitter = min(50.0, args.eval_phase_jitter_ms)

    phases: list[CurriculumPhase] = [
        CurriculumPhase("motion-180-clean", 1, anchor, anchor, 0.0, 0.0, clean),
        CurriculumPhase(
            "expand-bpm-clean",
            1,
            narrow_low,
            narrow_high,
            narrow_jitter,
            narrow_eval_jitter,
            clean,
        ),
        CurriculumPhase(
            "full-bpm-clean",
            1,
            args.bpm_min,
            args.bpm_max,
            args.train_phase_jitter_ms,
            args.eval_phase_jitter_ms,
            clean,
        ),
        CurriculumPhase(
            "latency-sampling",
            1,
            args.bpm_min,
            args.bpm_max,
            args.train_phase_jitter_ms,
            args.eval_phase_jitter_ms,
            sampled,
        ),
        CurriculumPhase(
            "full-vision",
            1,
            args.bpm_min,
            args.bpm_max,
            args.train_phase_jitter_ms,
            args.eval_phase_jitter_ms,
            final,
        ),
    ]
    for notes in note_curriculum(args.notes):
        if notes <= 1:
            continue
        phases.append(
            CurriculumPhase(
                f"{notes}-notes-full-vision",
                notes,
                args.bpm_min,
                args.bpm_max,
                args.train_phase_jitter_ms,
                args.eval_phase_jitter_ms,
                final,
            )
        )
    return tuple(phases)


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
    model: RecurrentActorCritic,
    device: torch.device,
    *,
    phase: CurriculumPhase,
    control_dt: float,
    episodes: int,
    eval_bpm_points: int,
    seed_base: int,
) -> Probe:
    jitter_s = phase.eval_phase_jitter_ms / 1000.0
    offsets = (
        (0.0,)
        if episodes == 1
        else tuple(-jitter_s + 2.0 * jitter_s * i / (episodes - 1) for i in range(episodes))
    )
    bpms = bpm_points(phase.bpm_min, phase.bpm_max, eval_bpm_points)

    hits = targets = full = clean = overloads = too_early = 0
    errors: list[float] = []
    was_training = model.training
    model.eval()

    for i, offset in enumerate(offsets):
        bpm = bpms[i % len(bpms)]
        env = make_env(
            bpm=bpm,
            notes=phase.notes,
            start_s=max(0.050, curriculum_start_s(phase.notes) + offset),
            control_dt=control_dt,
            config=phase.vision,
            seed=seed_base + i * 1009,
        )
        observation = env.reset()
        state = model.initial_state(device)
        while True:
            action, state = model.deterministic_action(
                observation_tensor(observation, device), state
            )
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

    return Probe(episodes, hits, targets, full, clean, overloads, too_early, tuple(errors))


def full_vision_phase(args: argparse.Namespace, notes: int) -> CurriculumPhase:
    return CurriculumPhase(
        f"{notes}-notes-retention",
        notes,
        args.bpm_min,
        args.bpm_max,
        args.train_phase_jitter_ms,
        args.eval_phase_jitter_ms,
        final_vision_config(args),
    )


def previous_note_probes(
    model: RecurrentActorCritic,
    device: torch.device,
    *,
    previous_notes: tuple[int, ...],
    args: argparse.Namespace,
) -> Retention:
    result: list[tuple[int, Probe]] = []
    for index, notes in enumerate(previous_notes):
        result.append(
            (
                notes,
                deterministic_probe(
                    model,
                    device,
                    phase=full_vision_phase(args, notes),
                    control_dt=args.control_dt,
                    episodes=args.retention_episodes,
                    eval_bpm_points=args.eval_bpm_points,
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
                f"{notes}n hit={probe.hit_rate:.2f} full={probe.full_rate:.2f} "
                f"ovl={probe.overloads}/{probe.episodes}"
                for notes, probe in retention.probes
            )
        )


def transplant_warm_start(
    model: RecurrentActorCritic,
    checkpoint: Path,
    device: torch.device,
) -> str:
    saved = torch.load(checkpoint, map_location=device)
    source = saved["model"]
    target = model.state_dict()
    copied: list[str] = []

    # Old feed-forward Gaussian/geometry policies used backbone.0 -> backbone.2.
    # Preserve only motor-related input weights; visual cue columns are never
    # transplanted into geometry observations.
    source_first = source.get("backbone.0.weight")
    if source_first is not None and source_first.shape[0] == target["input_layer.weight"].shape[0]:
        columns = min(6, source_first.shape[1], target["input_layer.weight"].shape[1])
        target["input_layer.weight"][:, :columns].copy_(source_first[:, :columns])
        copied.append("input_layer.weight(motor-columns)")
    source_first_bias = source.get("backbone.0.bias")
    if source_first_bias is not None and source_first_bias.shape == target["input_layer.bias"].shape:
        target["input_layer.bias"].copy_(source_first_bias)
        copied.append("input_layer.bias")
    source_post = source.get("backbone.2.weight")
    source_post_bias = source.get("backbone.2.bias")
    if source_post is not None and source_post.shape == target["post.weight"].shape:
        target["post.weight"].copy_(source_post)
        copied.append("post.weight")
    if source_post_bias is not None and source_post_bias.shape == target["post.bias"].shape:
        target["post.bias"].copy_(source_post_bias)
        copied.append("post.bias")

    for name in ("actor_mean.weight", "actor_mean.bias", "critic.weight", "critic.bias"):
        value = source.get(name)
        if value is not None and value.shape == target[name].shape:
            target[name].copy_(value)
            copied.append(name)

    # Intentionally do not copy log_std: recurrent geometry starts with wider
    # exploration so the clean one-note phase can discover a valid key press.
    model.load_state_dict(target)
    return ", ".join(copied)


def save_checkpoint(
    path: Path,
    model: RecurrentActorCritic,
    optimizer: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
    phase_index: int,
    phase: CurriculumPhase,
    global_episode: int,
    phase_episode: int,
    probe: Probe,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 8,
            "experiment": "planet-geometry-sequence-v0.2",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "input_dim": GEOMETRY_INPUT_DIM,
            "hidden_dim": model.hidden_dim,
            "target_notes": args.notes,
            "control_dt": args.control_dt,
            "final_bpm_min": args.bpm_min,
            "final_bpm_max": args.bpm_max,
            "final_vision_config": vision_dict(final_vision_config(args)),
            "curriculum_phase_index": phase_index,
            "curriculum_phase_name": phase.name,
            "curriculum_phase": {
                "notes": phase.notes,
                "bpm_min": phase.bpm_min,
                "bpm_max": phase.bpm_max,
                "train_phase_jitter_ms": phase.train_phase_jitter_ms,
                "eval_phase_jitter_ms": phase.eval_phase_jitter_ms,
                "vision_config": vision_dict(phase.vision),
            },
            "global_episode": global_episode,
            "phase_episode": phase_episode,
            "initial_log_std": args.initial_log_std,
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
    phases = build_curriculum(args)

    model = RecurrentActorCritic(
        input_dim=GEOMETRY_INPUT_DIM,
        hidden_dim=64,
        initial_log_std=args.initial_log_std,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint = Path(args.checkpoint)

    global_episode = 0
    resume_phase_index = 0
    if args.resume and args.warm_start:
        raise SystemExit("--resume and --warm-start are mutually exclusive")

    if args.warm_start:
        warm_path = Path(args.warm_start)
        if not warm_path.exists():
            raise SystemExit(f"warm-start checkpoint not found: {warm_path}")
        copied = transplant_warm_start(model, warm_path, device)
        print(f"warm-start: {warm_path}")
        print(f"warm-start copied: {copied}")
        print("warm-start note: no Gaussian/geometry timing columns or old log_std copied")

    if args.resume:
        if not checkpoint.exists():
            raise SystemExit(f"resume checkpoint not found: {checkpoint}")
        saved = torch.load(checkpoint, map_location=device)
        if saved.get("experiment") != "planet-geometry-sequence-v0.2":
            raise SystemExit("checkpoint is not geometry sequence v0.2; use --warm-start")
        if int(saved.get("input_dim", -1)) != GEOMETRY_INPUT_DIM:
            raise SystemExit("geometry checkpoint input dimension mismatch")
        if int(saved.get("target_notes", args.notes)) != args.notes:
            raise SystemExit("resume target note count mismatch")
        if abs(float(saved.get("final_bpm_min")) - args.bpm_min) > 1e-9 or abs(float(saved.get("final_bpm_max")) - args.bpm_max) > 1e-9:
            raise SystemExit("resume BPM domain mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        global_episode = int(saved.get("global_episode", 0))
        resume_phase_index = int(saved.get("curriculum_phase_index", 1)) - 1
        if not 0 <= resume_phase_index < len(phases):
            raise SystemExit("resume curriculum phase is invalid")
        print(
            f"resume: {checkpoint} at phase {resume_phase_index + 1}/{len(phases)}, "
            f"global episode {global_episode}"
        )

    print("=== Planet Geometry Sequence RL v0.2 ===")
    print("policy: GRU memory; recurrent state resets only at episode boundaries")
    print("agent-visible: motor state + orbit(x,y) + next-tile vector(x,y)")
    print("hidden: timestamp, BPM, target angle, angle error, rotation-direction flag")
    print(f"final BPM domain: {args.bpm_min:g}..{args.bpm_max:g}")
    print(
        "curriculum: fixed clean motion -> BPM expansion -> latency/sampling -> "
        "noise/dropout -> note-count expansion"
    )
    print(f"exploration: initial sigma={model.log_std.detach().exp().mean().item():.3f}")
    print()

    saved_probe: Probe | None = None
    saved_phase = resume_phase_index + 1
    reached_phase = saved_phase

    for phase_pos in range(resume_phase_index, len(phases)):
        phase = phases[phase_pos]
        phase_index = phase_pos + 1
        reached_phase = phase_index
        previous_notes = tuple(
            notes
            for notes in note_curriculum(phase.notes)
            if notes < phase.notes
        ) if phase.notes > 1 else ()
        best_key: tuple[float, ...] | None = None
        phase_passed = False
        bpms = bpm_points(phase.bpm_min, phase.bpm_max, args.eval_bpm_points)
        sample_text = (
            "continuous"
            if phase.vision.sample_period_s <= 0.0
            else f"{1.0 / phase.vision.sample_period_s:.0f}Hz"
        )

        print(
            f"--- phase {phase_index}/{len(phases)} {phase.name}: {phase.notes} note(s), "
            f"BPM={phase.bpm_min:g}..{phase.bpm_max:g}, "
            f"phase-jitter=±{phase.train_phase_jitter_ms:g}ms ---"
        )
        print(
            f"  vision latency={phase.vision.latency_s*1000:.1f}±"
            f"{phase.vision.latency_jitter_s*1000:.1f}ms sample={sample_text} "
            f"noise={phase.vision.position_noise_std:.3f} "
            f"dropout={phase.vision.dropout_probability*100:.1f}%"
        )
        print("  eval BPM probes: " + ", ".join(f"{x:g}" for x in bpms))

        probe = deterministic_probe(
            model,
            device,
            phase=phase,
            control_dt=args.control_dt,
            episodes=args.eval_episodes,
            eval_bpm_points=args.eval_bpm_points,
            seed_base=args.seed * 1000000 + phase_index * 10000,
        )
        retention = previous_note_probes(
            model,
            device,
            previous_notes=previous_notes,
            args=args,
        )
        print_probe("  baseline: ", probe)
        print_retention(retention)
        best_key = rank_key(probe, retention)
        saved_probe = probe
        saved_phase = phase_index
        save_checkpoint(
            checkpoint,
            model,
            optimizer,
            args=args,
            phase_index=phase_index,
            phase=phase,
            global_episode=global_episode,
            phase_episode=0,
            probe=probe,
        )

        if passes(probe, retention, args, phase.notes):
            phase_passed = True
            print("  phase already passed")
            print()
            continue

        for phase_episode in range(1, args.episodes + 1):
            global_episode += 1
            rollout_notes = phase.notes
            replayed = False
            rollout_phase = phase
            if previous_notes and rng.random() < args.previous_stage_replay:
                rollout_notes = rng.choice(previous_notes)
                rollout_phase = full_vision_phase(args, rollout_notes)
                replayed = True

            bpm = (
                rollout_phase.bpm_min
                if abs(rollout_phase.bpm_max - rollout_phase.bpm_min) < 1e-12
                else rng.uniform(rollout_phase.bpm_min, rollout_phase.bpm_max)
            )
            start_s = max(
                0.050,
                curriculum_start_s(rollout_notes)
                + rng.uniform(
                    -rollout_phase.train_phase_jitter_ms,
                    rollout_phase.train_phase_jitter_ms,
                )
                / 1000.0,
            )
            env = make_env(
                bpm=bpm,
                notes=rollout_notes,
                start_s=start_s,
                control_dt=args.control_dt,
                config=rollout_phase.vision,
                seed=rng.randrange(0, 2**31),
            )
            observation = env.reset()
            state = model.initial_state(device)
            rewards: list[float] = []
            log_probs: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []

            while True:
                x = observation_tensor(observation, device)
                action, log_prob, value, entropy, state = model.sample_action(x, state)
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
                advantages = (advantages - advantages.mean()) / (
                    advantages.std(unbiased=False) + 1e-6
                )

            policy_loss = -(log_probs_t * advantages).mean()
            value_loss = ((values_t - returns) ** 2).mean()
            loss = (
                policy_loss
                + args.value_coef * value_loss
                - args.entropy_coef * entropies_t.mean()
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if phase_episode == 1 or phase_episode % args.log_every == 0 or phase_episode == args.episodes:
                stats = env.stats
                replay = f" replay={rollout_notes}n" if replayed else ""
                print(
                    f"ep {global_episode:4d} phase={phase_episode:3d}/{args.episodes}{replay} "
                    f"bpm={bpm:6.1f} reward={stats.total_reward:8.3f} "
                    f"hits={stats.hits:2d}/{stats.targets:2d} miss={stats.misses:2d} "
                    f"early={stats.too_early_presses:2d} ovl={stats.overload_counter} "
                    f"MAE={'--' if stats.mean_abs_error_ms is None else f'{stats.mean_abs_error_ms:5.1f}ms'} "
                    f"sigma={model.log_std.detach().exp().mean().item():.3f} "
                    f"loss={loss.item():8.4f}"
                )

            if phase_episode != 1 and phase_episode % args.eval_every != 0 and phase_episode != args.episodes:
                continue

            probe = deterministic_probe(
                model,
                device,
                phase=phase,
                control_dt=args.control_dt,
                episodes=args.eval_episodes,
                eval_bpm_points=args.eval_bpm_points,
                seed_base=args.seed * 1000000 + phase_index * 10000,
            )
            retention = previous_note_probes(
                model,
                device,
                previous_notes=previous_notes,
                args=args,
            )
            print_probe("  eval: ", probe)
            print_retention(retention)

            key = rank_key(probe, retention)
            if key > best_key:
                best_key = key
                saved_probe = probe
                saved_phase = phase_index
                save_checkpoint(
                    checkpoint,
                    model,
                    optimizer,
                    args=args,
                    phase_index=phase_index,
                    phase=phase,
                    global_episode=global_episode,
                    phase_episode=phase_episode,
                    probe=probe,
                )
                print(f"  saved best -> {checkpoint}")

            if passes(probe, retention, args, phase.notes):
                phase_passed = True
                print("  phase passed")
                print()
                break

        if not phase_passed:
            print(
                f"curriculum stopped at phase {phase_index}/{len(phases)} "
                f"({phase.name}) after {args.episodes} episodes"
            )
            break

    print()
    print(f"best checkpoint: {checkpoint}")
    print(f"furthest curriculum phase reached: {reached_phase}/{len(phases)}")
    print(f"checkpoint phase: {saved_phase}/{len(phases)}")
    if saved_probe is not None:
        print_probe("deterministic best: ", saved_probe)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a recurrent policy from visible planet/tile geometry."
    )
    parser.add_argument("--episodes", type=int, default=500, help="maximum episodes per curriculum phase")
    parser.add_argument("--notes", type=int, default=16)
    parser.add_argument("--bpm-min", type=float, default=120.0)
    parser.add_argument("--bpm-max", type=float, default=300.0)
    parser.add_argument("--eval-bpm-points", type=int, default=5)
    parser.add_argument("--control-dt", type=float, default=0.010)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.002)
    parser.add_argument("--initial-log-std", type=float, default=-0.70)
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
    parser.add_argument("--checkpoint", default="checkpoints/planet_geometry_v02.pt")
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
