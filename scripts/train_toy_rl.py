from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - user-facing dependency message
    raise SystemExit(
        'PyTorch is required. Run: uv sync --extra dev --extra rl --inexact'
    ) from exc

from dmdod.adofai_rules import normal_timing_windows
from dmdod.curriculum import curriculum_start_s, note_curriculum
from dmdod.perception import VisualCueConfig
from dmdod.rhythm_env import RhythmMotorEnv, make_regular_targets
from dmdod.toy_policy import ActorCritic, discounted_returns, observation_tensor


@dataclass(frozen=True)
class DeterministicProbe:
    episodes: int
    total_hits: int
    total_targets: int
    total_too_early: int
    overload_runs: int
    full_hit_runs: int
    clean_clear_runs: int
    timing_errors_ms: tuple[float, ...]

    @property
    def hit_rate(self) -> float:
        return self.total_hits / max(self.total_targets, 1)

    @property
    def full_hit_rate(self) -> float:
        return self.full_hit_runs / max(self.episodes, 1)

    @property
    def clean_clear_rate(self) -> float:
        return self.clean_clear_runs / max(self.episodes, 1)

    @property
    def mean_too_early(self) -> float:
        return self.total_too_early / max(self.episodes, 1)

    @property
    def mean_abs_error_ms(self) -> float | None:
        if not self.timing_errors_ms:
            return None
        return sum(abs(value) for value in self.timing_errors_ms) / len(self.timing_errors_ms)


@dataclass(frozen=True)
class RetentionSummary:
    probes: tuple[tuple[int, DeterministicProbe], ...]

    @property
    def overload_runs(self) -> int:
        return sum(probe.overload_runs for _, probe in self.probes)

    @property
    def min_hit_rate(self) -> float:
        if not self.probes:
            return 1.0
        return min(probe.hit_rate for _, probe in self.probes)

    @property
    def min_full_rate(self) -> float:
        if not self.probes:
            return 1.0
        return min(probe.full_hit_rate for _, probe in self.probes)

    @property
    def min_clean_rate(self) -> float:
        if not self.probes:
            return 1.0
        return min(probe.clean_clear_rate for _, probe in self.probes)


def resolved_bpm_range(args: argparse.Namespace) -> tuple[float, float]:
    low = args.bpm if args.bpm_min is None else args.bpm_min
    high = args.bpm if args.bpm_max is None else args.bpm_max
    return float(low), float(high)


def evaluation_bpms(args: argparse.Namespace) -> tuple[float, ...]:
    low, high = resolved_bpm_range(args)
    if abs(high - low) < 1e-12:
        return (low,)
    count = max(2, args.eval_bpm_points)
    return tuple(low + (high - low) * i / (count - 1) for i in range(count))


def resolved_perception_config(args: argparse.Namespace) -> VisualCueConfig:
    humanized = args.humanized_perception
    jitter_ms = args.perception_latency_jitter_ms
    noise_std = args.perception_noise_std
    dropout = args.perception_dropout
    hz = args.perception_hz

    if jitter_ms is None:
        jitter_ms = 15.0 if humanized else 0.0
    if noise_std is None:
        noise_std = 0.03 if humanized else 0.0
    if dropout is None:
        dropout = 0.01 if humanized else 0.0
    if hz is None:
        hz = 60.0 if humanized else 0.0

    return VisualCueConfig(
        latency_s=args.perception_latency_ms / 1000.0,
        width_s=args.perception_width_ms / 1000.0,
        horizon_s=args.perception_horizon_ms / 1000.0,
        latency_jitter_s=jitter_ms / 1000.0,
        sample_period_s=(1.0 / hz if hz > 0.0 else 0.0),
        amplitude_noise_std=noise_std,
        dropout_probability=dropout,
    )


def perception_config_dict(config: VisualCueConfig) -> dict[str, float]:
    return {
        "latency_s": config.latency_s,
        "width_s": config.width_s,
        "horizon_s": config.horizon_s,
        "latency_jitter_s": config.latency_jitter_s,
        "sample_period_s": config.sample_period_s,
        "amplitude_noise_std": config.amplitude_noise_std,
        "dropout_probability": config.dropout_probability,
    }


def make_env(
    *,
    bpm: float,
    notes: int,
    start_s: float,
    pattern: str,
    same_hand: bool,
    control_dt: float,
    cue_config: VisualCueConfig,
    perception_seed: int,
) -> RhythmMotorEnv:
    targets = make_regular_targets(
        bpm=bpm,
        count=notes,
        start_s=start_s,
        pattern=pattern,
    )
    return RhythmMotorEnv(
        targets,
        bpm=bpm,
        same_hand=same_hand,
        control_dt_s=control_dt,
        cue_config=cue_config,
        perception_seed=perception_seed,
    )


def deterministic_probe(
    model: ActorCritic,
    device: torch.device,
    *,
    bpms: tuple[float, ...],
    notes: int,
    base_start_s: float,
    pattern: str,
    same_hand: bool,
    control_dt: float,
    cue_config: VisualCueConfig,
    episodes: int,
    start_jitter_ms: float,
    seed_base: int,
) -> DeterministicProbe:
    """Evaluate actor means across deterministic phase/BPM/sensor probes."""

    if episodes == 1:
        offsets = (0.0,)
    else:
        jitter_s = start_jitter_ms / 1000.0
        offsets = tuple(
            -jitter_s + 2.0 * jitter_s * i / (episodes - 1)
            for i in range(episodes)
        )

    total_hits = 0
    total_targets = 0
    total_too_early = 0
    overload_runs = 0
    full_hit_runs = 0
    clean_clear_runs = 0
    errors: list[float] = []

    was_training = model.training
    model.eval()
    for episode_index, offset in enumerate(offsets):
        bpm = bpms[episode_index % len(bpms)]
        start_s = max(0.050, base_start_s + offset)
        env = make_env(
            bpm=bpm,
            notes=notes,
            start_s=start_s,
            pattern=pattern,
            same_hand=same_hand,
            control_dt=control_dt,
            cue_config=cue_config,
            perception_seed=seed_base + episode_index * 1009,
        )
        observation = env.reset()
        while True:
            x = observation_tensor(observation, device)
            action = model.deterministic_action(x)
            transition = env.step(action)
            observation = transition.observation
            if transition.done:
                break

        stats = env.stats
        full_hit = stats.hits == stats.targets and not stats.overloaded
        clean_clear = full_hit and stats.too_early_presses == 0
        total_hits += stats.hits
        total_targets += stats.targets
        total_too_early += stats.too_early_presses
        overload_runs += int(stats.overloaded)
        full_hit_runs += int(full_hit)
        clean_clear_runs += int(clean_clear)
        errors.extend(env.timing_errors_ms)

    if was_training:
        model.train()

    return DeterministicProbe(
        episodes=episodes,
        total_hits=total_hits,
        total_targets=total_targets,
        total_too_early=total_too_early,
        overload_runs=overload_runs,
        full_hit_runs=full_hit_runs,
        clean_clear_runs=clean_clear_runs,
        timing_errors_ms=tuple(errors),
    )


def probe_previous_stages(
    model: ActorCritic,
    device: torch.device,
    *,
    previous_stages: tuple[int, ...],
    args: argparse.Namespace,
    same_hand: bool,
    bpms: tuple[float, ...],
    cue_config: VisualCueConfig,
) -> RetentionSummary:
    probes: list[tuple[int, DeterministicProbe]] = []
    for index, notes in enumerate(previous_stages):
        probes.append(
            (
                notes,
                deterministic_probe(
                    model,
                    device,
                    bpms=bpms,
                    notes=notes,
                    base_start_s=(0.750 if args.no_curriculum else curriculum_start_s(notes)),
                    pattern=args.pattern,
                    same_hand=same_hand,
                    control_dt=args.control_dt,
                    cue_config=cue_config,
                    episodes=args.retention_episodes,
                    start_jitter_ms=args.eval_start_jitter_ms,
                    seed_base=args.seed * 100000 + index * 10000 + notes,
                ),
            )
        )
    return RetentionSummary(tuple(probes))


def retention_passes(summary: RetentionSummary, args: argparse.Namespace) -> bool:
    return (
        summary.overload_runs == 0
        and summary.min_hit_rate >= args.retention_hit_rate
        and summary.min_full_rate >= args.retention_full_rate
    )


def probe_rank_key(
    probe: DeterministicProbe,
    retention: RetentionSummary,
) -> tuple[float, ...]:
    """Prefer retained older skills, then current-stage performance."""

    error = probe.mean_abs_error_ms
    return (
        1.0 if retention.overload_runs == 0 else 0.0,
        -float(retention.overload_runs),
        retention.min_hit_rate,
        retention.min_full_rate,
        1.0 if probe.overload_runs == 0 else 0.0,
        -float(probe.overload_runs),
        probe.hit_rate,
        probe.clean_clear_rate,
        probe.full_hit_rate,
        -probe.mean_too_early,
        -(error if error is not None else float("inf")),
    )


def passes_stage(
    probe: DeterministicProbe,
    retention: RetentionSummary,
    args: argparse.Namespace,
    *,
    stage_notes: int,
) -> bool:
    if probe.overload_runs != 0 or not retention_passes(retention, args):
        return False
    if stage_notes == 1:
        return probe.clean_clear_rate >= args.stage1_clean_rate
    return (
        probe.hit_rate >= args.advance_hit_rate
        and probe.full_hit_rate >= args.advance_full_rate
    )


def stage_criterion_text(args: argparse.Namespace, stage_notes: int) -> str:
    current = (
        f"clean>={args.stage1_clean_rate:.2f}"
        if stage_notes == 1
        else f"hit>={args.advance_hit_rate:.2f}, full>={args.advance_full_rate:.2f}"
    )
    return (
        f"{current}, no OVERLOAD; retention hit>={args.retention_hit_rate:.2f}, "
        f"full>={args.retention_full_rate:.2f}"
    )


def save_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
    same_hand: bool,
    stage_index: int,
    stage_notes: int,
    stage_start_s: float,
    global_episode: int,
    stage_episode: int,
    probe: DeterministicProbe,
    retention: RetentionSummary,
    bpm_low: float,
    bpm_high: float,
    cue_config: VisualCueConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 6,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "bpm": args.bpm,
            "bpm_min": bpm_low,
            "bpm_max": bpm_high,
            "notes": stage_notes,
            "target_notes": args.notes,
            "pattern": args.pattern,
            "same_hand": same_hand,
            "control_dt": args.control_dt,
            "start_s": stage_start_s,
            "global_episode": global_episode,
            "stage_episode": stage_episode,
            "hidden_dim": 64,
            "initial_log_std": args.initial_log_std,
            "timing_option": "normal",
            "game_rules": "adofai-wiki-v0.1",
            "body_rules": "zero-separated-reversal-counted-v0.2",
            "perception": "active-next-target-sampled-noisy-v0.3",
            "perception_config": perception_config_dict(cue_config),
            "curriculum": not args.no_curriculum,
            "curriculum_stage_index": stage_index,
            "curriculum_stage_notes": stage_notes,
            "previous_stage_replay": args.previous_stage_replay,
            "deterministic_probe": {
                "episodes": probe.episodes,
                "hit_rate": probe.hit_rate,
                "full_hit_rate": probe.full_hit_rate,
                "clean_clear_rate": probe.clean_clear_rate,
                "clean_clear_runs": probe.clean_clear_runs,
                "mean_too_early": probe.mean_too_early,
                "overload_runs": probe.overload_runs,
                "mean_abs_error_ms": probe.mean_abs_error_ms,
            },
            "retention_probe": {
                str(notes): {
                    "episodes": old_probe.episodes,
                    "hit_rate": old_probe.hit_rate,
                    "full_hit_rate": old_probe.full_hit_rate,
                    "clean_clear_rate": old_probe.clean_clear_rate,
                    "overload_runs": old_probe.overload_runs,
                }
                for notes, old_probe in retention.probes
            },
        },
        path,
    )


def _same_perception_config(saved: object, current: VisualCueConfig) -> bool:
    if not isinstance(saved, dict):
        saved = {}
    expected = perception_config_dict(current)
    legacy = perception_config_dict(VisualCueConfig())
    actual = {key: float(saved.get(key, legacy[key])) for key in expected}
    return all(abs(actual[key] - expected[key]) <= 1e-12 for key in expected)


def _validate_resume_checkpoint(
    saved: dict[str, object],
    args: argparse.Namespace,
    *,
    same_hand: bool,
    bpm_low: float,
    bpm_high: float,
    cue_config: VisualCueConfig,
) -> None:
    target_notes = int(saved.get("target_notes", saved.get("notes", args.notes)))
    if target_notes != args.notes:
        raise SystemExit(
            f"resume checkpoint target_notes={target_notes}, but --notes={args.notes}"
        )
    if str(saved.get("pattern", args.pattern)) != args.pattern:
        raise SystemExit("resume checkpoint pattern does not match --pattern")
    if bool(saved.get("same_hand", same_hand)) != same_hand:
        raise SystemExit("resume checkpoint same_hand setting does not match")
    saved_bpm = float(saved.get("bpm", args.bpm))
    saved_low = float(saved.get("bpm_min", saved_bpm))
    saved_high = float(saved.get("bpm_max", saved_bpm))
    if abs(saved_low - bpm_low) > 1e-9 or abs(saved_high - bpm_high) > 1e-9:
        raise SystemExit("resume checkpoint BPM domain differs; use --warm-start to change the task")
    if abs(float(saved.get("control_dt", args.control_dt)) - args.control_dt) > 1e-12:
        raise SystemExit("resume checkpoint control_dt does not match --control-dt")
    if not _same_perception_config(saved.get("perception_config"), cue_config):
        raise SystemExit("resume checkpoint perception differs; use --warm-start to change the sensor model")


def print_probe(prefix: str, probe: DeterministicProbe) -> None:
    print(
        f"{prefix}hit={probe.hit_rate:.3f}  "
        f"full={probe.full_hit_runs}/{probe.episodes}  "
        f"clean={probe.clean_clear_runs}/{probe.episodes}  "
        f"early={probe.mean_too_early:.2f}/run  "
        f"OVERLOAD={probe.overload_runs}/{probe.episodes}  "
        f"MAE={'--' if probe.mean_abs_error_ms is None else f'{probe.mean_abs_error_ms:.1f}ms'}"
    )


def print_retention(summary: RetentionSummary) -> None:
    if not summary.probes:
        return
    pieces = []
    for notes, probe in summary.probes:
        pieces.append(
            f"{notes}n hit={probe.hit_rate:.2f} full={probe.full_hit_rate:.2f} "
            f"ovl={probe.overload_runs}/{probe.episodes}"
        )
    print("  retention: " + " | ".join(pieces))


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    same_hand = args.pattern != "alternate" or args.same_hand
    bpm_low, bpm_high = resolved_bpm_range(args)
    probe_bpms = evaluation_bpms(args)
    cue_config = resolved_perception_config(args)

    stages = (args.notes,) if args.no_curriculum else note_curriculum(args.notes)
    model = ActorCritic(initial_log_std=args.initial_log_std).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint = Path(args.checkpoint)

    global_episode = 0
    resume_stage_pos = 0
    resumed = False

    if args.resume and args.warm_start:
        raise SystemExit("--resume and --warm-start are mutually exclusive")

    if args.warm_start:
        warm_path = Path(args.warm_start)
        if not warm_path.exists():
            raise SystemExit(f"warm-start checkpoint not found: {warm_path}")
        saved = torch.load(warm_path, map_location=device)
        model.load_state_dict(saved["model"])
        print(f"warm-start: loaded policy weights from {warm_path}; optimizer/curriculum restart")

    if args.resume:
        if not checkpoint.exists():
            raise SystemExit(f"resume checkpoint not found: {checkpoint}")
        saved = torch.load(checkpoint, map_location=device)
        _validate_resume_checkpoint(
            saved,
            args,
            same_hand=same_hand,
            bpm_low=bpm_low,
            bpm_high=bpm_high,
            cue_config=cue_config,
        )
        model.load_state_dict(saved["model"])
        if "optimizer" in saved:
            optimizer.load_state_dict(saved["optimizer"])
            optimizer_restored = True
        else:
            optimizer_restored = False
        global_episode = int(saved.get("global_episode", 0))
        saved_stage_notes = int(saved.get("curriculum_stage_notes", saved.get("notes", stages[0])))
        if saved_stage_notes not in stages:
            raise SystemExit(
                f"resume checkpoint stage {saved_stage_notes} is not in current curriculum {stages}"
            )
        resume_stage_pos = stages.index(saved_stage_notes)
        resumed = True
        print(
            f"resume: {checkpoint} at {saved_stage_notes} note(s), "
            f"global episode {global_episode}"
        )
        print(
            "resume: optimizer state restored"
            if optimizer_restored
            else "resume: legacy checkpoint has no optimizer state; Adam state starts fresh"
        )

    low_windows = normal_timing_windows(bpm_low)
    high_windows = normal_timing_windows(bpm_high)

    print("=== Toy RL with ADOFAI Normal Timing ===")
    if abs(bpm_high - bpm_low) < 1e-12:
        print(
            f"{bpm_low:g} BPM: Perfect ±{low_windows.perfect_s*1000:.2f} ms, "
            f"E/L Perfect ±{low_windows.early_late_perfect_s*1000:.2f} ms, "
            f"Pass ±{low_windows.pass_s*1000:.2f} ms"
        )
    else:
        print(
            f"BPM domain: {bpm_low:g}..{bpm_high:g} (uniform training); "
            f"Pass ±{low_windows.pass_s*1000:.2f}..{high_windows.pass_s*1000:.2f} ms"
        )
        print("eval BPM probes: " + ", ".join(f"{bpm:g}" for bpm in probe_bpms))
    print("OVERLOAD: Too Early +2, valid hit -1, fail at 6")
    print("perception: only the next unresolved target remains visible")
    print(
        "sensor: "
        f"latency={cue_config.latency_s*1000:.1f}±{cue_config.latency_jitter_s*1000:.1f} ms, "
        f"sample={'continuous' if cue_config.sample_period_s <= 0 else f'{1.0/cue_config.sample_period_s:.0f} Hz'}, "
        f"noise={cue_config.amplitude_noise_std:.3f}, "
        f"dropout={cue_config.dropout_probability*100:.1f}%"
    )
    print(
        f"exploration: initial log_std={args.initial_log_std:.2f} "
        f"(current sigma={model.log_std.detach().exp().mean().item():.3f})"
    )
    if args.no_curriculum:
        print(f"curriculum: disabled ({args.notes} notes)")
    else:
        print("curriculum: " + " -> ".join(str(stage) for stage in stages) + " notes")
        print(
            f"anti-forgetting: {args.previous_stage_replay*100:.0f}% rollouts replay a random previous stage; "
            f"retention probes={args.retention_episodes}/stage"
        )
    print(
        f"deterministic policy eval: every {args.eval_every} episodes, "
        f"{args.eval_episodes} phase/BPM/sensor probes"
    )
    print(
        f"stage 1 advance: clean>={args.stage1_clean_rate:.2f}; later: "
        f"hit>={args.advance_hit_rate:.2f}, full>={args.advance_full_rate:.2f}; "
        f"retention: hit>={args.retention_hit_rate:.2f}, full>={args.retention_full_rate:.2f}"
    )
    print()

    saved_probe: DeterministicProbe | None = None
    saved_stage_notes = stages[resume_stage_pos]
    reached_stage_notes = stages[resume_stage_pos]

    active_stages = stages[resume_stage_pos:]
    for stage_offset, stage_notes in enumerate(active_stages):
        stage_pos = resume_stage_pos + stage_offset
        stage_index = stage_pos + 1
        previous_stages = tuple(stages[:stage_pos])
        stage_start_s = 0.750 if args.no_curriculum else curriculum_start_s(stage_notes)
        train_jitter_s = args.train_start_jitter_ms / 1000.0
        stage_best_key: tuple[float, ...] | None = None
        stage_passed = False
        reached_stage_notes = stage_notes

        print(
            f"--- stage {stage_index}/{len(stages)}: {stage_notes} note(s), "
            f"base start={stage_start_s*1000:.0f} ms ---"
        )

        # Always test the incoming policy before changing it. This lets a strong
        # warm-start skip easy curriculum stages and avoids one unnecessary
        # optimizer step destabilizing an already-good policy.
        baseline = deterministic_probe(
            model,
            device,
            bpms=probe_bpms,
            notes=stage_notes,
            base_start_s=stage_start_s,
            pattern=args.pattern,
            same_hand=same_hand,
            control_dt=args.control_dt,
            cue_config=cue_config,
            episodes=args.eval_episodes,
            start_jitter_ms=args.eval_start_jitter_ms,
            seed_base=args.seed * 1000000 + stage_index * 10000,
        )
        retention = probe_previous_stages(
            model,
            device,
            previous_stages=previous_stages,
            args=args,
            same_hand=same_hand,
            bpms=probe_bpms,
            cue_config=cue_config,
        )
        print_probe("  stage baseline: ", baseline)
        print_retention(retention)
        stage_best_key = probe_rank_key(baseline, retention)
        saved_probe = baseline
        saved_stage_notes = stage_notes
        save_checkpoint(
            checkpoint,
            model,
            optimizer,
            args=args,
            same_hand=same_hand,
            stage_index=stage_index,
            stage_notes=stage_notes,
            stage_start_s=stage_start_s,
            global_episode=global_episode,
            stage_episode=0,
            probe=baseline,
            retention=retention,
            bpm_low=bpm_low,
            bpm_high=bpm_high,
            cue_config=cue_config,
        )
        print(f"  saved stage baseline -> {checkpoint}")

        if passes_stage(baseline, retention, args, stage_notes=stage_notes):
            stage_passed = True
            print(f"  stage already passed: {stage_criterion_text(args, stage_notes)}")
            print()
            if stage_pos == len(stages) - 1:
                break
            continue

        for stage_episode in range(1, args.episodes + 1):
            global_episode += 1

            rollout_notes = stage_notes
            replayed = False
            if previous_stages and rng.random() < args.previous_stage_replay:
                rollout_notes = rng.choice(previous_stages)
                replayed = True
            rollout_base_start = 0.750 if args.no_curriculum else curriculum_start_s(rollout_notes)
            offset = rng.uniform(-train_jitter_s, train_jitter_s) if train_jitter_s > 0.0 else 0.0
            rollout_start_s = max(0.050, rollout_base_start + offset)
            rollout_bpm = (
                bpm_low
                if abs(bpm_high - bpm_low) < 1e-12
                else rng.uniform(bpm_low, bpm_high)
            )
            env = make_env(
                bpm=rollout_bpm,
                notes=rollout_notes,
                start_s=rollout_start_s,
                pattern=args.pattern,
                same_hand=same_hand,
                control_dt=args.control_dt,
                cue_config=cue_config,
                perception_seed=rng.randrange(0, 2**31),
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
            entropy_t = torch.stack(entropies)
            advantages = returns - values_t.detach()
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)

            policy_loss = -(log_probs_t * advantages).mean()
            value_loss = torch.mean((values_t - returns) ** 2)
            entropy_bonus = entropy_t.mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_bonus

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            stats = env.stats
            if stage_episode == 1 or stage_episode % args.log_every == 0 or stage_episode == args.episodes:
                error_text = "--" if stats.mean_abs_error_ms is None else f"{stats.mean_abs_error_ms:6.1f}ms"
                overload_text = "OVERLOAD" if stats.overloaded else f"ovl={stats.overload_counter}"
                std = model.log_std.detach().exp().mean().item()
                replay_text = f" replay={rollout_notes}n" if replayed else ""
                print(
                    f"ep {global_episode:4d}  stage={stage_episode:3d}/{args.episodes}{replay_text}  "
                    f"bpm={rollout_bpm:6.1f}  reward={stats.total_reward:8.3f}  "
                    f"hits={stats.hits:2d}/{stats.targets:2d}  miss={stats.misses:2d}  "
                    f"early={stats.too_early_presses:2d}  {overload_text:8s}  "
                    f"MAE={error_text}  sigma={std:.3f}  loss={loss.item():8.4f}"
                )

            should_probe = (
                stage_episode == 1
                or stage_episode % args.eval_every == 0
                or stage_episode == args.episodes
            )
            if not should_probe:
                continue

            probe = deterministic_probe(
                model,
                device,
                bpms=probe_bpms,
                notes=stage_notes,
                base_start_s=stage_start_s,
                pattern=args.pattern,
                same_hand=same_hand,
                control_dt=args.control_dt,
                cue_config=cue_config,
                episodes=args.eval_episodes,
                start_jitter_ms=args.eval_start_jitter_ms,
                seed_base=args.seed * 1000000 + stage_index * 10000,
            )
            retention = probe_previous_stages(
                model,
                device,
                previous_stages=previous_stages,
                args=args,
                same_hand=same_hand,
                bpms=probe_bpms,
                cue_config=cue_config,
            )
            print_probe("  eval mean-policy: ", probe)
            print_retention(retention)

            key = probe_rank_key(probe, retention)
            if stage_best_key is None or key > stage_best_key:
                stage_best_key = key
                saved_probe = probe
                saved_stage_notes = stage_notes
                save_checkpoint(
                    checkpoint,
                    model,
                    optimizer,
                    args=args,
                    same_hand=same_hand,
                    stage_index=stage_index,
                    stage_notes=stage_notes,
                    stage_start_s=stage_start_s,
                    global_episode=global_episode,
                    stage_episode=stage_episode,
                    probe=probe,
                    retention=retention,
                    bpm_low=bpm_low,
                    bpm_high=bpm_high,
                    cue_config=cue_config,
                )
                print(f"  saved deterministic best -> {checkpoint}")

            if passes_stage(probe, retention, args, stage_notes=stage_notes):
                stage_passed = True
                print(f"  stage passed: {stage_criterion_text(args, stage_notes)}")
                print()
                break

        if not stage_passed:
            print(
                f"curriculum stopped at {stage_notes} note(s): "
                f"advance criterion was not reached within {args.episodes} episodes"
            )
            break
        if stage_pos == len(stages) - 1:
            break

    print()
    print(f"best checkpoint: {checkpoint}")
    print(f"furthest stage reached: {reached_stage_notes}/{args.notes} notes")
    print(f"checkpoint stage: {saved_stage_notes} notes")
    if saved_probe is not None:
        print(f"deterministic hit rate: {saved_probe.hit_rate:.4f}")
        print(f"deterministic full-hit runs: {saved_probe.full_hit_runs}/{saved_probe.episodes}")
        print(f"deterministic clean clears: {saved_probe.clean_clear_runs}/{saved_probe.episodes}")
        print(f"deterministic OVERLOAD runs: {saved_probe.overload_runs}/{saved_probe.episodes}")
        print(f"deterministic mean Too Early: {saved_probe.mean_too_early:.3f}")
        if saved_probe.mean_abs_error_ms is not None:
            print(f"deterministic mean abs timing error: {saved_probe.mean_abs_error_ms:.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the toy motor-control rhythm policy.")
    parser.add_argument("--episodes", type=int, default=250, help="maximum episodes per stage")
    parser.add_argument("--bpm", type=float, default=180.0, help="nominal BPM / fixed BPM when no range is given")
    parser.add_argument("--bpm-min", type=float, default=None, help="uniform training BPM lower bound")
    parser.add_argument("--bpm-max", type=float, default=None, help="uniform training BPM upper bound")
    parser.add_argument("--eval-bpm-points", type=int, default=5, help="BPM points spanning the training domain during probes")
    parser.add_argument("--notes", type=int, default=16)
    parser.add_argument("--pattern", choices=("left", "alternate"), default="left")
    parser.add_argument("--same-hand", action="store_true", help="use same-hand body for alternate pattern")
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
    parser.add_argument("--train-start-jitter-ms", type=float, default=100.0)
    parser.add_argument("--eval-start-jitter-ms", type=float, default=100.0)
    parser.add_argument("--previous-stage-replay", type=float, default=0.20)
    parser.add_argument("--stage1-clean-rate", type=float, default=0.80)
    parser.add_argument("--advance-hit-rate", type=float, default=0.90)
    parser.add_argument("--advance-full-rate", type=float, default=0.80)
    parser.add_argument("--retention-hit-rate", type=float, default=0.70)
    parser.add_argument("--retention-full-rate", type=float, default=0.60)
    parser.add_argument(
        "--humanized-perception",
        action="store_true",
        help="enable 15 ms latency jitter, 60 Hz sampling, 0.03 cue noise and dropout probability 0.01 defaults",
    )
    parser.add_argument("--perception-latency-ms", type=float, default=50.0)
    parser.add_argument("--perception-width-ms", type=float, default=90.0)
    parser.add_argument("--perception-horizon-ms", type=float, default=450.0)
    parser.add_argument("--perception-latency-jitter-ms", type=float, default=None)
    parser.add_argument("--perception-noise-std", type=float, default=None)
    parser.add_argument("--perception-dropout", type=float, default=None)
    parser.add_argument("--perception-hz", type=float, default=None)
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an identical task from --checkpoint",
    )
    parser.add_argument(
        "--warm-start",
        default=None,
        help="load policy weights only from another checkpoint, then restart optimizer/curriculum",
    )
    parser.add_argument("--checkpoint", default="checkpoints/toy_policy.pt")
    args = parser.parse_args()

    bpm_low, bpm_high = resolved_bpm_range(args)
    if args.episodes <= 0 or args.notes <= 0 or args.bpm <= 0.0:
        parser.error("episodes, notes, and bpm must be positive")
    if bpm_low <= 0.0 or bpm_high <= 0.0 or bpm_low > bpm_high:
        parser.error("BPM range must satisfy 0 < bpm-min <= bpm-max")
    if args.eval_bpm_points <= 0:
        parser.error("eval-bpm-points must be positive")
    if args.eval_every <= 0 or args.eval_episodes <= 0 or args.retention_episodes <= 0 or args.log_every <= 0:
        parser.error("log/eval intervals and probe episode counts must be positive")
    if args.train_start_jitter_ms < 0.0 or args.eval_start_jitter_ms < 0.0:
        parser.error("start jitter values must be non-negative")
    if args.perception_latency_ms < 0.0 or args.perception_width_ms <= 0.0 or args.perception_horizon_ms <= 0.0:
        parser.error("perception latency/horizon/width values are invalid")
    if args.perception_latency_jitter_ms is not None and args.perception_latency_jitter_ms < 0.0:
        parser.error("perception-latency-jitter-ms must be non-negative")
    if args.perception_noise_std is not None and args.perception_noise_std < 0.0:
        parser.error("perception-noise-std must be non-negative")
    if args.perception_dropout is not None and not 0.0 <= args.perception_dropout <= 1.0:
        parser.error("perception-dropout must be between 0 and 1")
    if args.perception_hz is not None and args.perception_hz < 0.0:
        parser.error("perception-hz must be non-negative")
    for name in (
        "previous_stage_replay",
        "stage1_clean_rate",
        "advance_hit_rate",
        "advance_full_rate",
        "retention_hit_rate",
        "retention_full_rate",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"{name.replace('_', '-')} must be between 0 and 1")
    train(args)


if __name__ == "__main__":
    main()
