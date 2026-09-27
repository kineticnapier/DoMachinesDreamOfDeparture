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
    env = RhythmMotorEnv(targets, same_hand=same_hand, control_dt_s=args.control_dt)

    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_hits = -1
    best_error = float("inf")
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
        is_best = stats.hits > best_hits or (stats.hits == best_hits and error < best_error)
        if is_best:
            # Save the exact policy that generated this rollout.  Saving after
            # optimizer.step() would associate the episode score with a policy
            # that had never actually produced that trajectory.
            best_hits = stats.hits
            best_error = error
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "format_version": 1,
                    "model": model.state_dict(),
                    "bpm": args.bpm,
                    "notes": args.notes,
                    "pattern": args.pattern,
                    "same_hand": args.same_hand,
                    "control_dt": args.control_dt,
                    "episode": episode,
                    "hidden_dim": 64,
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
            print(
                f"ep {episode:4d}/{args.episodes}  "
                f"reward={stats.total_reward:8.3f}  "
                f"hits={stats.hits:2d}/{stats.targets:2d}  "
                f"miss={stats.misses:2d}  stray={stats.stray_presses:3d}  "
                f"MAE={error_text}  loss={loss.item():8.4f}"
            )

    print()
    print(f"best checkpoint: {checkpoint}")
    print(f"best hits: {best_hits}/{args.notes}")
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
