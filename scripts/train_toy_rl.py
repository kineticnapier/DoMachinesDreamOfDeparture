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

from dmdod.curriculum import curriculum_start_s, note_curriculum
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


def make_env(
    *,
    bpm: float,
    notes: int,
    start_s: float,
    pattern: str,
    same_hand: bool,
    control_dt: float,
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
    )


def deterministic_probe(
    model: ActorCritic,
    device: torch.device,
    *,
    bpm: float,
    notes: int,
    base_start_s: float,
    pattern: str,
    same_hand: bool,
    control_dt: float,
    episodes: int,
    start_jitter_ms: float,
) -> DeterministicProbe:
    """Evaluate actor means on fixed phase offsets, with no exploration noise."""

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
    for offset in offsets:
        start_s = max(0.050, base_start_s + offset)
        env = make_env(
            bpm=bpm,
            notes=notes,
            start_s=start_s,
            pattern=pattern,
            same_hand=same_hand,
            control_dt=control_dt,
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


def probe_rank_key(probe: DeterministicProbe) -> tuple[float, ...]:
    """Rank deterministic policies: survive first, then actually hit cues."""

    error = probe.mean_abs_error_ms
    return (
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
    args: argparse.Namespace,
    *,
    stage_notes: int,
) -> bool:
    if probe.overload_runs != 0:
        return False
    if stage_notes == 1:
        # One-note hit/full rates are heavily quantized.  Use the directly
        # meaningful criterion here: cue-driven clean clears across phase probes.
        return probe.clean_clear_rate >= args.stage1_clean_rate
    return (
        probe.hit_rate >= args.advance_hit_rate
        and probe.full_hit_rate >= args.advance_full_rate
    )


def stage_criterion_text(args: argparse.Namespace, stage_notes: int) -> str:
    if stage_notes == 1:
        return f"clean>={args.stage1_clean_rate:.2f}, no OVERLOAD"
    return (
        f"hit>={args.advance_hit_rate:.2f}, "
        f"full>={args.advance_full_rate:.2f}, no OVERLOAD"
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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 4,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "bpm": args.bpm,
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
            "curriculum": not args.no_curriculum,
            "curriculum_stage_index": stage_index,
            "curriculum_stage_notes": stage_notes,
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
        },
        path,
    )


def _validate_resume_checkpoint(
    saved: dict[str, object],
    args: argparse.Namespace,
    *,
    same_hand: bool,
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
    if abs(float(saved.get("bpm", args.bpm)) - args.bpm) > 1e-9:
        raise SystemExit("resume checkpoint BPM does not match --bpm")
    if abs(float(saved.get("control_dt", args.control_dt)) - args.control_dt) > 1e-12:
        raise SystemExit("resume checkpoint control_dt does not match --control-dt")


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    same_hand = args.pattern != "alternate" or args.same_hand

    stages = (args.notes,) if args.no_curriculum else note_curriculum(args.notes)
    model = ActorCritic(initial_log_std=args.initial_log_std).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint = Path(args.checkpoint)

    global_episode = 0
    resume_stage_pos = 0
    resumed = False
    resume_probe: DeterministicProbe | None = None

    if args.resume:
        if not checkpoint.exists():
            raise SystemExit(f"resume checkpoint not found: {checkpoint}")
        saved = torch.load(checkpoint, map_location=device)
        _validate_resume_checkpoint(saved, args, same_hand=same_hand)
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
        if optimizer_restored:
            print("resume: optimizer state restored")
        else:
            print("resume: legacy checkpoint has no optimizer state; Adam state starts fresh")

    probe_env = make_env(
        bpm=args.bpm,
        notes=max(1, stages[resume_stage_pos]),
        start_s=(
            0.750
            if args.no_curriculum
            else curriculum_start_s(stages[resume_stage_pos])
        ),
        pattern=args.pattern,
        same_hand=same_hand,
        control_dt=args.control_dt,
    )
    windows = probe_env.timing_windows

    print("=== Toy RL with ADOFAI Normal Timing ===")
    print(
        f"{args.bpm:g} BPM: Perfect ±{windows.perfect_s*1000:.2f} ms, "
        f"E/L Perfect ±{windows.early_late_perfect_s*1000:.2f} ms, "
        f"Pass ±{windows.pass_s*1000:.2f} ms"
    )
    print("OVERLOAD: Too Early +2, valid hit -1, fail at 6")
    print(
        f"exploration: initial log_std={args.initial_log_std:.2f} "
        f"(current sigma={model.log_std.detach().exp().mean().item():.3f})"
    )
    if args.no_curriculum:
        print(f"curriculum: disabled ({args.notes} notes)")
    else:
        print("curriculum: " + " -> ".join(str(stage) for stage in stages) + " notes")
    print(
        f"deterministic checkpoint eval: every {args.eval_every} episodes, "
        f"{args.eval_episodes} phase probes ±{args.eval_start_jitter_ms:g} ms"
    )
    print(
        f"stage 1 advance: clean>={args.stage1_clean_rate:.2f}; "
        f"later stages: hit>={args.advance_hit_rate:.2f}, "
        f"full>={args.advance_full_rate:.2f}; all require no OVERLOAD"
    )
    print()

    saved_probe: DeterministicProbe | None = None
    saved_stage_notes = stages[resume_stage_pos]
    reached_stage_notes = stages[resume_stage_pos]

    active_stages = stages[resume_stage_pos:]
    for stage_offset, stage_notes in enumerate(active_stages):
        stage_pos = resume_stage_pos + stage_offset
        stage_index = stage_pos + 1
        stage_start_s = (
            0.750 if args.no_curriculum else curriculum_start_s(stage_notes)
        )
        train_jitter_s = args.train_start_jitter_ms / 1000.0
        stage_best_key: tuple[float, ...] | None = None
        stage_passed = False
        reached_stage_notes = stage_notes

        print(
            f"--- stage {stage_index}/{len(stages)}: {stage_notes} note(s), "
            f"base start={stage_start_s*1000:.0f} ms ---"
        )

        if resumed and stage_offset == 0:
            resume_probe = deterministic_probe(
                model,
                device,
                bpm=args.bpm,
                notes=stage_notes,
                base_start_s=stage_start_s,
                pattern=args.pattern,
                same_hand=same_hand,
                control_dt=args.control_dt,
                episodes=args.eval_episodes,
                start_jitter_ms=args.eval_start_jitter_ms,
            )
            stage_best_key = probe_rank_key(resume_probe)
            saved_probe = resume_probe
            saved_stage_notes = stage_notes
            print(
                f"  resume baseline: hit={resume_probe.hit_rate:.3f}  "
                f"full={resume_probe.full_hit_runs}/{resume_probe.episodes}  "
                f"clean={resume_probe.clean_clear_runs}/{resume_probe.episodes}  "
                f"early={resume_probe.mean_too_early:.2f}/run  "
                f"OVERLOAD={resume_probe.overload_runs}/{resume_probe.episodes}  "
                f"MAE={'--' if resume_probe.mean_abs_error_ms is None else f'{resume_probe.mean_abs_error_ms:.1f}ms'}"
            )
            if stage_pos < len(stages) - 1 and passes_stage(
                resume_probe, args, stage_notes=stage_notes
            ):
                print(f"  stage already passed on resume: {stage_criterion_text(args, stage_notes)}")
                print()
                continue

        for stage_episode in range(1, args.episodes + 1):
            global_episode += 1
            offset = rng.uniform(-train_jitter_s, train_jitter_s) if train_jitter_s > 0.0 else 0.0
            rollout_start_s = max(0.050, stage_start_s + offset)
            env = make_env(
                bpm=args.bpm,
                notes=stage_notes,
                start_s=rollout_start_s,
                pattern=args.pattern,
                same_hand=same_hand,
                control_dt=args.control_dt,
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
                print(
                    f"ep {global_episode:4d}  stage={stage_episode:3d}/{args.episodes}  "
                    f"reward={stats.total_reward:8.3f}  "
                    f"hits={stats.hits:2d}/{stats.targets:2d}  "
                    f"miss={stats.misses:2d}  early={stats.too_early_presses:2d}  "
                    f"{overload_text:8s}  MAE={error_text}  "
                    f"sigma={std:.3f}  loss={loss.item():8.4f}"
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
                bpm=args.bpm,
                notes=stage_notes,
                base_start_s=stage_start_s,
                pattern=args.pattern,
                same_hand=same_hand,
                control_dt=args.control_dt,
                episodes=args.eval_episodes,
                start_jitter_ms=args.eval_start_jitter_ms,
            )
            probe_error = probe.mean_abs_error_ms
            print(
                f"  eval mean-policy: hit={probe.hit_rate:.3f}  "
                f"full={probe.full_hit_runs}/{probe.episodes}  "
                f"clean={probe.clean_clear_runs}/{probe.episodes}  "
                f"early={probe.mean_too_early:.2f}/run  "
                f"OVERLOAD={probe.overload_runs}/{probe.episodes}  "
                f"MAE={'--' if probe_error is None else f'{probe_error:.1f}ms'}"
            )

            key = probe_rank_key(probe)
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
                )
                print(f"  saved deterministic best -> {checkpoint}")

            if stage_pos < len(stages) - 1 and passes_stage(
                probe, args, stage_notes=stage_notes
            ):
                stage_passed = True
                print(f"  stage passed: {stage_criterion_text(args, stage_notes)}")
                print()
                break

        if stage_pos < len(stages) - 1 and not stage_passed:
            print(
                f"curriculum stopped at {stage_notes} note(s): "
                f"advance criterion was not reached within {args.episodes} episodes"
            )
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
    parser.add_argument(
        "--episodes",
        type=int,
        default=250,
        help="maximum training episodes per curriculum stage",
    )
    parser.add_argument("--bpm", type=float, default=180.0)
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
    parser.add_argument("--train-start-jitter-ms", type=float, default=100.0)
    parser.add_argument("--eval-start-jitter-ms", type=float, default=100.0)
    parser.add_argument("--stage1-clean-rate", type=float, default=0.80)
    parser.add_argument("--advance-hit-rate", type=float, default=0.90)
    parser.add_argument("--advance-full-rate", type=float, default=0.80)
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from --checkpoint; older checkpoints resume with fresh optimizer state",
    )
    parser.add_argument("--checkpoint", default="checkpoints/toy_policy.pt")
    args = parser.parse_args()

    if args.episodes <= 0 or args.notes <= 0 or args.bpm <= 0.0:
        parser.error("episodes, notes, and bpm must be positive")
    if args.eval_every <= 0 or args.eval_episodes <= 0 or args.log_every <= 0:
        parser.error("log/eval intervals and eval episodes must be positive")
    if args.train_start_jitter_ms < 0.0 or args.eval_start_jitter_ms < 0.0:
        parser.error("start jitter values must be non-negative")
    if not 0.0 <= args.stage1_clean_rate <= 1.0:
        parser.error("stage1-clean-rate must be between 0 and 1")
    if not 0.0 <= args.advance_hit_rate <= 1.0:
        parser.error("advance-hit-rate must be between 0 and 1")
    if not 0.0 <= args.advance_full_rate <= 1.0:
        parser.error("advance-full-rate must be between 0 and 1")
    train(args)


if __name__ == "__main__":
    main()
