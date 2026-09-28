from __future__ import annotations

import argparse
import copy
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


@dataclass
class Rollout:
    observations: torch.Tensor
    latents: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    hits: int
    targets: int
    too_early: int
    overloaded: bool
    reward: float


@dataclass(frozen=True)
class BpmSlice:
    bpm: float
    episodes: int
    hits: int
    targets: int
    full: int
    clean: int
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
    def mean_error_ms(self) -> float | None:
        if not self.errors_ms:
            return None
        return sum(self.errors_ms) / len(self.errors_ms)

    @property
    def mae_ms(self) -> float | None:
        if not self.errors_ms:
            return None
        return sum(abs(x) for x in self.errors_ms) / len(self.errors_ms)


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
    bpm_slices: tuple[BpmSlice, ...] = ()

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

    @property
    def mean_error_ms(self) -> float | None:
        if not self.errors_ms:
            return None
        return sum(self.errors_ms) / len(self.errors_ms)

    @property
    def max_abs_bpm_bias_ms(self) -> float | None:
        values = [
            abs(value)
            for item in self.bpm_slices
            if (value := item.mean_error_ms) is not None
        ]
        return max(values) if values else None


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


def _bounded_range(
    args: argparse.Namespace,
    proposed_low: float,
    proposed_high: float,
) -> tuple[float, float]:
    low = max(args.bpm_min, proposed_low)
    high = min(args.bpm_max, proposed_high)
    if low <= high:
        return low, high
    anchor = min(max(180.0, args.bpm_min), args.bpm_max)
    return anchor, anchor


def build_curriculum(args: argparse.Namespace) -> tuple[CurriculumPhase, ...]:
    clean = clean_vision_config()
    sampled = sampled_vision_config(args)
    final = final_vision_config(args)
    anchor = min(max(180.0, args.bpm_min), args.bpm_max)
    bands = (
        (anchor, anchor, 0.0, "motion-180-clean"),
        (*_bounded_range(args, 159.0, 222.0), 50.0, "bpm-159-222-clean"),
        (*_bounded_range(args, 145.0, 240.0), 65.0, "bpm-145-240-clean"),
        (*_bounded_range(args, 135.0, 260.0), 80.0, "bpm-135-260-clean"),
        (*_bounded_range(args, 125.0, 280.0), 90.0, "bpm-125-280-clean"),
        (args.bpm_min, args.bpm_max, args.train_phase_jitter_ms, "full-bpm-clean"),
    )

    phases: list[CurriculumPhase] = []
    for low, high, jitter, name in bands:
        phases.append(
            CurriculumPhase(
                name,
                1,
                float(low),
                float(high),
                min(float(jitter), args.train_phase_jitter_ms),
                min(float(jitter), args.eval_phase_jitter_ms),
                clean,
            )
        )

    phases.append(
        CurriculumPhase(
            "latency-sampling",
            1,
            args.bpm_min,
            args.bpm_max,
            args.train_phase_jitter_ms,
            args.eval_phase_jitter_ms,
            sampled,
        )
    )
    phases.append(
        CurriculumPhase(
            "full-vision",
            1,
            args.bpm_min,
            args.bpm_max,
            args.train_phase_jitter_ms,
            args.eval_phase_jitter_ms,
            final,
        )
    )
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


def training_bpm_schedule(
    phase: CurriculumPhase,
    *,
    episodes: int,
    points: int,
    rng: random.Random,
) -> list[float]:
    """Stratify rollouts and deliberately revisit both BPM edges."""

    anchors = list(bpm_points(phase.bpm_min, phase.bpm_max, points))
    if len(anchors) > 1:
        anchors = [anchors[0], *anchors, anchors[-1]]
    schedule = [anchors[i % len(anchors)] for i in range(episodes)]
    rng.shuffle(schedule)
    return schedule


def make_env(
    *,
    bpm: float,
    notes: int,
    start_s: float,
    control_dt: float,
    config: PlanetVisionConfig,
    seed: int,
) -> GeometryRhythmEnv:
    return GeometryRhythmEnv(
        make_regular_targets(bpm=bpm, count=notes, start_s=start_s, pattern="left"),
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
    buckets: dict[float, dict[str, object]] = {
        bpm: {"episodes": 0, "hits": 0, "targets": 0, "full": 0, "clean": 0, "errors": []}
        for bpm in bpms
    }

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
        episode_errors = env.timing_errors_ms

        hits += stats.hits
        targets += stats.targets
        full += int(is_full)
        clean += int(is_clean)
        overloads += int(stats.overloaded)
        too_early += stats.too_early_presses
        errors.extend(episode_errors)

        bucket = buckets[bpm]
        bucket["episodes"] = int(bucket["episodes"]) + 1
        bucket["hits"] = int(bucket["hits"]) + stats.hits
        bucket["targets"] = int(bucket["targets"]) + stats.targets
        bucket["full"] = int(bucket["full"]) + int(is_full)
        bucket["clean"] = int(bucket["clean"]) + int(is_clean)
        bucket_errors = bucket["errors"]
        assert isinstance(bucket_errors, list)
        bucket_errors.extend(episode_errors)

    if was_training:
        model.train()

    slices = tuple(
        BpmSlice(
            bpm=bpm,
            episodes=int(bucket["episodes"]),
            hits=int(bucket["hits"]),
            targets=int(bucket["targets"]),
            full=int(bucket["full"]),
            clean=int(bucket["clean"]),
            errors_ms=tuple(float(x) for x in bucket["errors"]),
        )
        for bpm, bucket in buckets.items()
    )
    return Probe(
        episodes,
        hits,
        targets,
        full,
        clean,
        overloads,
        too_early,
        tuple(errors),
        slices,
    )


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


def completion_passes(
    probe: Probe,
    retention: Retention,
    args: argparse.Namespace,
    notes: int,
) -> bool:
    if probe.overloads or retention.overloads:
        return False
    if retention.min_hit_rate < args.retention_hit_rate:
        return False
    if retention.min_full_rate < args.retention_full_rate:
        return False
    if notes == 1:
        return probe.hit_rate >= args.stage1_hit_rate and probe.full_rate >= args.stage1_full_rate
    return probe.hit_rate >= args.advance_hit_rate and probe.full_rate >= args.advance_full_rate


def precision_passes(probe: Probe, args: argparse.Namespace, notes: int) -> bool:
    if notes != 1:
        return True
    if probe.clean_rate < args.stage1_clean_rate:
        return False
    bias = probe.max_abs_bpm_bias_ms
    if bias is not None and bias > args.precision_max_bpm_bias_ms:
        return False
    return True


def passes(probe: Probe, retention: Retention, args: argparse.Namespace, notes: int) -> bool:
    return completion_passes(probe, retention, args, notes) and precision_passes(probe, args, notes)


def rank_key(probe: Probe, retention: Retention) -> tuple[float, ...]:
    mae = probe.mae_ms
    bias = probe.max_abs_bpm_bias_ms
    return (
        1.0 if retention.overloads == 0 else 0.0,
        -float(retention.overloads),
        retention.min_hit_rate,
        retention.min_full_rate,
        1.0 if probe.overloads == 0 else 0.0,
        -float(probe.overloads),
        probe.hit_rate,
        probe.full_rate,
        probe.clean_rate,
        -(bias if bias is not None else float("inf")),
        -(mae if mae is not None else float("inf")),
    )


def _fmt_ms(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "--"
    return f"{value:+.1f}" if signed else f"{value:.1f}"


def print_probe(label: str, probe: Probe) -> None:
    print(
        f"{label}hit={probe.hit_rate:.3f} full={probe.full}/{probe.episodes} "
        f"clean={probe.clean}/{probe.episodes} ovl={probe.overloads}/{probe.episodes} "
        f"early={probe.too_early/max(1, probe.episodes):.2f}/run "
        f"meanErr={_fmt_ms(probe.mean_error_ms, signed=True)}ms "
        f"MAE={_fmt_ms(probe.mae_ms)}ms"
    )
    if probe.bpm_slices:
        print(
            "      BPM: "
            + " | ".join(
                f"{item.bpm:g} hit={item.hit_rate:.2f} full={item.full_rate:.2f} "
                f"err={_fmt_ms(item.mean_error_ms, signed=True)}ms"
                for item in probe.bpm_slices
            )
        )


def print_gate_status(
    probe: Probe,
    retention: Retention,
    args: argparse.Namespace,
    notes: int,
) -> None:
    completion = completion_passes(probe, retention, args, notes)
    precision = precision_passes(probe, args, notes)
    bias = probe.max_abs_bpm_bias_ms
    if notes == 1:
        print(
            f"      gates: completion={'PASS' if completion else 'pending'} "
            f"precision={'PASS' if precision else 'pending'} "
            f"clean={probe.clean_rate:.3f}/{args.stage1_clean_rate:.3f} "
            f"maxBpmBias={_fmt_ms(bias)}ms/{args.precision_max_bpm_bias_ms:.1f}ms"
        )
    else:
        print(f"      gate: completion={'PASS' if completion else 'pending'}")


def collect_rollout(
    model: RecurrentActorCritic,
    device: torch.device,
    *,
    phase: CurriculumPhase,
    control_dt: float,
    gamma: float,
    rng: random.Random,
    bpm_override: float | None = None,
) -> Rollout:
    bpm = (
        bpm_override
        if bpm_override is not None
        else (
            phase.bpm_min
            if abs(phase.bpm_max - phase.bpm_min) < 1e-12
            else rng.uniform(phase.bpm_min, phase.bpm_max)
        )
    )
    start_s = max(
        0.050,
        curriculum_start_s(phase.notes)
        + rng.uniform(-phase.train_phase_jitter_ms, phase.train_phase_jitter_ms) / 1000.0,
    )
    env = make_env(
        bpm=bpm,
        notes=phase.notes,
        start_s=start_s,
        control_dt=control_dt,
        config=phase.vision,
        seed=rng.randrange(0, 2**31),
    )
    observation = env.reset()
    state = model.initial_state(device)
    observations: list[torch.Tensor] = []
    latents: list[torch.Tensor] = []
    old_log_probs: list[torch.Tensor] = []
    old_values: list[torch.Tensor] = []
    rewards: list[float] = []

    model.eval()
    with torch.no_grad():
        while True:
            x = observation_tensor(observation, device)
            action, latent, log_prob, value, _, state = model.sample_action_latent(x, state)
            transition = env.step(action)
            observations.append(x.detach())
            latents.append(latent.detach())
            old_log_probs.append(log_prob.detach())
            old_values.append(value.detach())
            rewards.append(transition.reward)
            observation = transition.observation
            if transition.done:
                break

    returns = discounted_returns(rewards, gamma, device)
    old_values_t = torch.stack(old_values)
    advantages = returns - old_values_t
    stats = env.stats
    return Rollout(
        observations=torch.stack(observations),
        latents=torch.stack(latents),
        old_log_probs=torch.stack(old_log_probs),
        old_values=old_values_t,
        returns=returns,
        advantages=advantages,
        hits=stats.hits,
        targets=stats.targets,
        too_early=stats.too_early_presses,
        overloaded=stats.overloaded,
        reward=stats.total_reward,
    )


def ppo_update(
    model: RecurrentActorCritic,
    optimizer: torch.optim.Optimizer,
    rollouts: list[Rollout],
    args: argparse.Namespace,
) -> tuple[float, float, float, float, int]:
    all_advantages = torch.cat([rollout.advantages for rollout in rollouts])
    adv_mean = all_advantages.mean()
    adv_std = all_advantages.std(unbiased=False).clamp_min(1e-6)
    old_log_probs = torch.cat([rollout.old_log_probs for rollout in rollouts])
    returns = torch.cat([rollout.returns for rollout in rollouts])

    final_policy = final_value = final_entropy = final_kl = 0.0
    epochs_done = 0
    model.train()
    for epoch in range(args.ppo_epochs):
        new_log_probs_list: list[torch.Tensor] = []
        new_values_list: list[torch.Tensor] = []
        entropies_list: list[torch.Tensor] = []
        normalized_advantages: list[torch.Tensor] = []

        for rollout in rollouts:
            new_log_probs, new_values, entropies = model.evaluate_latent_sequence(
                rollout.observations,
                rollout.latents,
            )
            new_log_probs_list.append(new_log_probs)
            new_values_list.append(new_values)
            entropies_list.append(entropies)
            normalized_advantages.append((rollout.advantages - adv_mean) / adv_std)

        new_log_probs = torch.cat(new_log_probs_list)
        new_values = torch.cat(new_values_list)
        entropies = torch.cat(entropies_list)
        advantages = torch.cat(normalized_advantages)
        ratios = torch.exp(new_log_probs - old_log_probs)
        unclipped = ratios * advantages
        clipped = torch.clamp(ratios, 1.0 - args.ppo_clip, 1.0 + args.ppo_clip) * advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        value_loss = ((new_values - returns) ** 2).mean()
        entropy = entropies.mean()
        loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        with torch.no_grad():
            approx_kl = (old_log_probs - new_log_probs).mean().abs()
        final_policy = float(policy_loss.item())
        final_value = float(value_loss.item())
        final_entropy = float(entropy.item())
        final_kl = float(approx_kl.item())
        epochs_done = epoch + 1
        if args.target_kl > 0.0 and final_kl > args.target_kl:
            break

    return final_policy, final_value, final_entropy, final_kl, epochs_done


def warm_start(
    model: RecurrentActorCritic,
    checkpoint: Path,
    device: torch.device,
) -> str:
    saved = torch.load(checkpoint, map_location=device)
    source = saved["model"]
    target = model.state_dict()

    if saved.get("experiment") in {"planet-geometry-sequence-v0.2", "planet-geometry-ppo-v0.3"}:
        compatible = all(name in source and source[name].shape == value.shape for name, value in target.items())
        if compatible:
            model.load_state_dict(source)
            return "all recurrent weights"

    copied: list[str] = []
    first = source.get("backbone.0.weight")
    if first is not None and first.shape[0] == target["input_layer.weight"].shape[0]:
        columns = min(6, first.shape[1], target["input_layer.weight"].shape[1])
        target["input_layer.weight"][:, :columns].copy_(first[:, :columns])
        copied.append("input_layer.weight(motor-columns)")
    mapping = {
        "backbone.0.bias": "input_layer.bias",
        "backbone.2.weight": "post.weight",
        "backbone.2.bias": "post.bias",
        "actor_mean.weight": "actor_mean.weight",
        "actor_mean.bias": "actor_mean.bias",
        "critic.weight": "critic.weight",
        "critic.bias": "critic.bias",
    }
    for source_name, target_name in mapping.items():
        value = source.get(source_name)
        if value is not None and value.shape == target[target_name].shape:
            target[target_name].copy_(value)
            copied.append(target_name)
    model.load_state_dict(target)
    return ", ".join(copied) or "no compatible weights"


def save_checkpoint(
    path: Path,
    model: RecurrentActorCritic,
    optimizer: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
    phase_index: int,
    phase: CurriculumPhase,
    global_update: int,
    probe: Probe,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 10,
            "experiment": "planet-geometry-ppo-v0.3",
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
            "global_update": global_update,
            "initial_log_std": args.initial_log_std,
            "ppo": {
                "rollout_episodes": args.rollout_episodes,
                "epochs": args.ppo_epochs,
                "clip": args.ppo_clip,
                "target_kl": args.target_kl,
                "rollback_drop": args.rollback_drop,
                "train_bpm_points": args.train_bpm_points,
            },
            "probe": {
                "episodes": probe.episodes,
                "hit_rate": probe.hit_rate,
                "full_rate": probe.full_rate,
                "clean_rate": probe.clean_rate,
                "overloads": probe.overloads,
                "mean_error_ms": probe.mean_error_ms,
                "mae_ms": probe.mae_ms,
                "max_abs_bpm_bias_ms": probe.max_abs_bpm_bias_ms,
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
    resume_phase = 0
    global_update = 0

    if args.resume and args.warm_start:
        raise SystemExit("--resume and --warm-start are mutually exclusive")
    if args.warm_start:
        source = Path(args.warm_start)
        if not source.exists():
            raise SystemExit(f"warm-start checkpoint not found: {source}")
        copied = warm_start(model, source, device)
        print(f"warm-start: {source}")
        print(f"warm-start copied: {copied}")
    if args.resume:
        if not checkpoint.exists():
            raise SystemExit(f"resume checkpoint not found: {checkpoint}")
        saved = torch.load(checkpoint, map_location=device)
        if saved.get("experiment") != "planet-geometry-ppo-v0.3":
            raise SystemExit("checkpoint is not geometry PPO v0.3")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        resume_phase = int(saved.get("curriculum_phase_index", 1)) - 1
        global_update = int(saved.get("global_update", 0))

    print("=== Planet Geometry Recurrent PPO v0.3.1 ===")
    print("policy: GRU; PPO uses complete episode sequences")
    print("agent-visible: motor + orbit(x,y) + next-tile vector(x,y)")
    print("hidden: timestamp, BPM, target angle, angle error, rotation-direction flag")
    print(
        f"PPO: {args.rollout_episodes} episodes/update, {args.ppo_epochs} epochs, "
        f"clip={args.ppo_clip:.2f}, target-KL={args.target_kl:.3f}, lr={args.lr:g}"
    )
    print(
        f"probe: {args.eval_episodes} episodes, {args.eval_bpm_points} BPM points; "
        f"training: {args.train_bpm_points} stratified BPM points with edge oversampling"
    )
    print(
        f"one-note gates: hit>={args.stage1_hit_rate:.2f}, full>={args.stage1_full_rate:.2f}; "
        f"precision clean>={args.stage1_clean_rate:.2f}, "
        f"max BPM bias<={args.precision_max_bpm_bias_ms:.0f}ms"
    )
    print(f"catastrophic rollback: restore best if hit/full drops by >{args.rollback_drop:.2f}")
    print("curriculum: " + " -> ".join(phase.name for phase in phases))
    print()

    for phase_pos in range(resume_phase, len(phases)):
        phase = phases[phase_pos]
        phase_index = phase_pos + 1
        previous_notes = (
            tuple(notes for notes in note_curriculum(phase.notes) if notes < phase.notes)
            if phase.notes > 1
            else ()
        )
        print(
            f"--- phase {phase_index}/{len(phases)} {phase.name}: {phase.notes} note(s), "
            f"BPM={phase.bpm_min:g}..{phase.bpm_max:g}, jitter=±{phase.train_phase_jitter_ms:g}ms ---"
        )
        baseline = deterministic_probe(
            model,
            device,
            phase=phase,
            control_dt=args.control_dt,
            episodes=args.eval_episodes,
            eval_bpm_points=args.eval_bpm_points,
            seed_base=args.seed * 1000000 + phase_index * 10000,
        )
        retention = previous_note_probes(model, device, previous_notes=previous_notes, args=args)
        print_probe("  baseline: ", baseline)
        print_gate_status(baseline, retention, args, phase.notes)
        if passes(baseline, retention, args, phase.notes):
            print("  phase already passed")
            save_checkpoint(
                checkpoint,
                model,
                optimizer,
                args=args,
                phase_index=phase_index,
                phase=phase,
                global_update=global_update,
                probe=baseline,
            )
            print()
            continue

        best_probe = baseline
        best_retention = retention
        best_key = rank_key(baseline, retention)
        best_model = copy.deepcopy(model.state_dict())
        best_optimizer = copy.deepcopy(optimizer.state_dict())
        save_checkpoint(
            checkpoint,
            model,
            optimizer,
            args=args,
            phase_index=phase_index,
            phase=phase,
            global_update=global_update,
            probe=baseline,
        )
        phase_passed = False

        for phase_update in range(1, args.updates_per_phase + 1):
            global_update += 1
            rollouts: list[Rollout] = []
            bpm_schedule = training_bpm_schedule(
                phase,
                episodes=args.rollout_episodes,
                points=args.train_bpm_points,
                rng=rng,
            )
            for rollout_index in range(args.rollout_episodes):
                rollout_phase = phase
                bpm_override: float | None = bpm_schedule[rollout_index]
                if previous_notes and rng.random() < args.previous_stage_replay:
                    rollout_phase = full_vision_phase(args, rng.choice(previous_notes))
                    bpm_override = None
                rollouts.append(
                    collect_rollout(
                        model,
                        device,
                        phase=rollout_phase,
                        control_dt=args.control_dt,
                        gamma=args.gamma,
                        rng=rng,
                        bpm_override=bpm_override,
                    )
                )

            policy_loss, value_loss, entropy, approx_kl, epochs_done = ppo_update(
                model, optimizer, rollouts, args
            )
            rollout_hit = sum(r.hits for r in rollouts) / max(1, sum(r.targets for r in rollouts))
            rollout_reward = sum(r.reward for r in rollouts) / len(rollouts)
            rollout_overload = sum(int(r.overloaded) for r in rollouts)
            print(
                f"  upd {phase_update:3d}/{args.updates_per_phase} global={global_update:4d} "
                f"rollout-hit={rollout_hit:.3f} reward={rollout_reward:+.3f} "
                f"ovl={rollout_overload}/{len(rollouts)} "
                f"sigma={model.log_std.detach().exp().mean().item():.3f} "
                f"pi={policy_loss:+.4f} v={value_loss:.4f} H={entropy:.3f} "
                f"KL={approx_kl:.4f} epochs={epochs_done}"
            )

            if phase_update % args.eval_every_updates != 0 and phase_update != args.updates_per_phase:
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
            retention = previous_note_probes(model, device, previous_notes=previous_notes, args=args)
            print_probe("    eval: ", probe)
            print_gate_status(probe, retention, args, phase.notes)
            key = rank_key(probe, retention)

            if key > best_key:
                best_key = key
                best_probe = probe
                best_retention = retention
                best_model = copy.deepcopy(model.state_dict())
                best_optimizer = copy.deepcopy(optimizer.state_dict())
                save_checkpoint(
                    checkpoint,
                    model,
                    optimizer,
                    args=args,
                    phase_index=phase_index,
                    phase=phase,
                    global_update=global_update,
                    probe=probe,
                )
                print(f"    saved best -> {checkpoint}")

            catastrophic = (
                probe.hit_rate < best_probe.hit_rate - args.rollback_drop
                or probe.full_rate < best_probe.full_rate - args.rollback_drop
                or (best_probe.overloads == 0 and probe.overloads > 0)
            )
            if catastrophic:
                model.load_state_dict(best_model)
                optimizer.load_state_dict(best_optimizer)
                for group in optimizer.param_groups:
                    group["lr"] = max(args.min_lr, float(group["lr"]) * args.rollback_lr_factor)
                print(
                    f"    rollback -> best hit={best_probe.hit_rate:.3f} "
                    f"full={best_probe.full_rate:.3f}; lr={optimizer.param_groups[0]['lr']:.2e}"
                )
                continue

            if passes(probe, retention, args, phase.notes):
                phase_passed = True
                print("    phase passed (completion + precision)")
                print()
                break

        if not phase_passed:
            model.load_state_dict(best_model)
            optimizer.load_state_dict(best_optimizer)
            if completion_passes(best_probe, best_retention, args, phase.notes):
                print(
                    f"precision target not fully met at {phase.name}; advancing on stable completion gate "
                    f"(hit={best_probe.hit_rate:.3f}, full={best_probe.full_rate:.3f}, "
                    f"clean={best_probe.clean_rate:.3f}, maxBias={_fmt_ms(best_probe.max_abs_bpm_bias_ms)}ms)"
                )
                print()
                continue
            print(
                f"curriculum stopped at {phase.name}; best hit={best_probe.hit_rate:.3f}, "
                f"full={best_probe.full_rate:.3f}"
            )
            break

    print()
    print(f"best checkpoint: {checkpoint}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the recurrent planet-geometry policy with conservative PPO."
    )
    parser.add_argument("--checkpoint", default="checkpoints/planet_geometry_v03_ppo.pt")
    parser.add_argument("--warm-start", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--notes", type=int, default=16)
    parser.add_argument("--bpm-min", type=float, default=120.0)
    parser.add_argument("--bpm-max", type=float, default=300.0)
    parser.add_argument("--control-dt", type=float, default=0.010)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--rollout-episodes", type=int, default=16)
    parser.add_argument("--updates-per-phase", type=int, default=40)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-clip", type=float, default=0.15)
    parser.add_argument("--target-kl", type=float, default=0.020)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--rollback-drop", type=float, default=0.20)
    parser.add_argument("--rollback-lr-factor", type=float, default=0.70)
    parser.add_argument("--initial-log-std", type=float, default=-0.70)
    parser.add_argument("--eval-every-updates", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-bpm-points", type=int, default=5)
    parser.add_argument("--train-bpm-points", type=int, default=5)
    parser.add_argument("--retention-episodes", type=int, default=20)
    parser.add_argument("--train-phase-jitter-ms", type=float, default=100.0)
    parser.add_argument("--eval-phase-jitter-ms", type=float, default=100.0)
    parser.add_argument("--previous-stage-replay", type=float, default=0.20)
    parser.add_argument("--stage1-hit-rate", type=float, default=0.95)
    parser.add_argument("--stage1-full-rate", type=float, default=0.95)
    parser.add_argument("--stage1-clean-rate", type=float, default=0.80)
    parser.add_argument("--precision-max-bpm-bias-ms", type=float, default=60.0)
    parser.add_argument("--advance-hit-rate", type=float, default=0.90)
    parser.add_argument("--advance-full-rate", type=float, default=0.80)
    parser.add_argument("--retention-hit-rate", type=float, default=0.70)
    parser.add_argument("--retention-full-rate", type=float, default=0.60)
    parser.add_argument("--vision-latency-ms", type=float, default=50.0)
    parser.add_argument("--vision-latency-jitter-ms", type=float, default=15.0)
    parser.add_argument("--vision-hz", type=float, default=60.0)
    parser.add_argument("--vision-noise-std", type=float, default=0.015)
    parser.add_argument("--vision-dropout", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.notes <= 0 or args.rollout_episodes <= 0 or args.updates_per_phase <= 0:
        parser.error("notes, rollout-episodes and updates-per-phase must be positive")
    if args.bpm_min <= 0.0 or args.bpm_max < args.bpm_min:
        parser.error("BPM domain must satisfy 0 < bpm-min <= bpm-max")
    if args.ppo_epochs <= 0 or not 0.0 < args.ppo_clip < 1.0:
        parser.error("invalid PPO epoch/clip settings")
    if (
        args.eval_every_updates <= 0
        or args.eval_episodes <= 0
        or args.eval_bpm_points <= 0
        or args.train_bpm_points <= 0
    ):
        parser.error("evaluation/training BPM settings must be positive")
    if args.retention_episodes <= 0:
        parser.error("retention-episodes must be positive")
    if not 0.0 <= args.previous_stage_replay <= 1.0:
        parser.error("previous-stage-replay must be between 0 and 1")
    if not 0.0 <= args.vision_dropout <= 1.0:
        parser.error("vision-dropout must be between 0 and 1")
    if args.rollback_drop < 0.0 or not 0.0 < args.rollback_lr_factor <= 1.0:
        parser.error("invalid rollback settings")
    if args.precision_max_bpm_bias_ms <= 0.0:
        parser.error("precision-max-bpm-bias-ms must be positive")
    for name in (
        "stage1_hit_rate",
        "stage1_full_rate",
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
