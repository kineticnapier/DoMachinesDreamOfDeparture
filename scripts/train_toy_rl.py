from __future__ import annotations

import argparse
from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - user-facing dependency message
    raise SystemExit(
        'PyTorch is required. Run: uv sync --extra dev --extra rl --inexact'
    ) from exc

from dmdod.rhythm_env import RhythmMotorEnv, make_regular_targets
from dmdod.toy_policy import ActorCritic, discounted_returns, observation_tensor


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    targets = make_regular_targets(
        bpm=args.bpm,
        count=args.notes,
        start_s=0.750,
        pattern=args.pattern,
    )
    same_hand = args.pattern != "alternate" or args.same_hand
    env = RhythmMotorEnv(
        targets,
        bpm=args.bpm,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
    )

    windows = env.timing_windows
    print("=== Toy RL with ADOFAI Normal Timing ===")
    print(
        f"{args.bpm:g} BPM: Perfect ±{windows.perfect_s*1000:.2f} ms, "
        f"E/L Perfect ±{windows.early_late_perfect_s*1000:.2f} ms, "
        f"Pass ±{windows.pass_s*1000:.2f} ms"
    )
    print("OVERLOAD: Too Early +2, valid hit -1, fail at 6")
    print()

    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_key: tuple[int, int, int, float] | None = None
    best_hits = -1
    best_error = float("inf")
    best_too_early = 0
    checkpoint = Path(args.checkpoint)

    for episode in range(1, args.episodes + 1):
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

        stats = env.stats
        error = stats.mean_abs_error_ms if stats.mean_abs_error_ms is not None else float("inf")
        # OVERLOAD is a failed run, so any non-overloaded policy ranks above an
        # overloaded one.  Among surviving runs prefer hits, fewer Too Early
        # presses, then tighter timing.
        candidate_key = (
            0 if stats.overloaded else 1,
            stats.hits,
            -stats.too_early_presses,
            -error,
        )
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_hits = stats.hits
            best_error = error
            best_too_early = stats.too_early_presses
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            # Save the exact policy that generated this rollout, before the
            # optimizer update changes its parameters.
            torch.save(
                {
                    "format_version": 2,
                    "model": model.state_dict(),
                    "bpm": args.bpm,
                    "notes": args.notes,
                    "pattern": args.pattern,
                    "same_hand": args.same_hand,
                    "control_dt": args.control_dt,
                    "episode": episode,
                    "hidden_dim": 64,
                    "timing_option": "normal",
                    "game_rules": "adofai-wiki-v0.1",
                },
                checkpoint,
            )

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

        if episode == 1 or episode % args.log_every == 0 or episode == args.episodes:
            error_text = "--" if stats.mean_abs_error_ms is None else f"{stats.mean_abs_error_ms:6.1f}ms"
            overload_text = "OVERLOAD" if stats.overloaded else f"ovl={stats.overload_counter}"
            print(
                f"ep {episode:4d}/{args.episodes}  "
                f"reward={stats.total_reward:8.3f}  "
                f"hits={stats.hits:2d}/{stats.targets:2d}  "
                f"miss={stats.misses:2d}  early={stats.too_early_presses:2d}  "
                f"{overload_text:8s}  MAE={error_text}  loss={loss.item():8.4f}"
            )

    print()
    print(f"best checkpoint: {checkpoint}")
    print(f"best hits: {best_hits}/{args.notes}")
    print(f"best Too Early inputs: {best_too_early}")
    if best_error != float("inf"):
        print(f"best mean abs timing error: {best_error:.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the first toy motor-control rhythm policy.")
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--bpm", type=float, default=180.0)
    parser.add_argument("--notes", type=int, default=16)
    parser.add_argument("--pattern", choices=("left", "alternate"), default="left")
    parser.add_argument("--same-hand", action="store_true", help="use same-hand body for alternate pattern")
    parser.add_argument("--control-dt", type=float, default=0.010)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint", default="checkpoints/toy_policy.pt")
    args = parser.parse_args()

    if args.episodes <= 0 or args.notes <= 0 or args.bpm <= 0.0:
        parser.error("episodes, notes, and bpm must be positive")
    train(args)


if __name__ == "__main__":
    main()
