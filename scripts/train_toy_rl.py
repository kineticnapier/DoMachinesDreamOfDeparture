from __future__ import annotations

import argparse
from pathlib import Path

try:
    import torch
    from torch import nn
    from torch.distributions import Normal
except ImportError as exc:  # pragma: no cover - user-facing dependency message
    raise SystemExit('PyTorch is required. Run: pip install -e ".[rl]"') from exc

from dmdod.motor_env import MotorAction
from dmdod.rhythm_env import RhythmMotorEnv, RhythmObservation, make_regular_targets


def observation_tensor(observation: RhythmObservation, device: torch.device) -> torch.Tensor:
    m = observation.motor
    # Fixed normalization constants are unit conversions / broad physical scales,
    # not chart timing information.
    values = [
        m.left_position_m / 0.006,
        m.right_position_m / 0.006,
        m.left_velocity_m_s / 1.0,
        m.right_velocity_m_s / 1.0,
        1.0 if m.left_pressed else 0.0,
        1.0 if m.right_pressed else 0.0,
        observation.cue.left,
        observation.cue.right,
    ]
    return torch.tensor(values, dtype=torch.float32, device=device)


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, 2)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((2,), -0.35))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mean = self.actor_mean(h)
        value = self.critic(h).squeeze(-1)
        std = self.log_std.exp().clamp(0.08, 1.5)
        return mean, std, value

    def sample_action(self, x: torch.Tensor) -> tuple[MotorAction, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std, value = self(x)
        dist = Normal(mean, std)
        latent = dist.sample()
        squashed = torch.tanh(latent)
        # Change-of-variables correction for tanh. latent/action are sampled
        # without a reparameterized path; policy gradients flow through log_prob.
        log_prob = dist.log_prob(latent).sum() - torch.log(1.0 - squashed.square() + 1e-6).sum()
        entropy = dist.entropy().sum()
        action = MotorAction(float(squashed[0].item()), float(squashed[1].item()))
        return action, log_prob, value, entropy


def discounted_returns(rewards: list[float], gamma: float, device: torch.device) -> torch.Tensor:
    running = 0.0
    result: list[float] = []
    for reward in reversed(rewards):
        running = reward + gamma * running
        result.append(running)
    result.reverse()
    return torch.tensor(result, dtype=torch.float32, device=device)


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
        error = stats.mean_abs_error_ms if stats.mean_abs_error_ms is not None else float("inf")
        is_best = stats.hits > best_hits or (stats.hits == best_hits and error < best_error)
        if is_best:
            best_hits = stats.hits
            best_error = error
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "bpm": args.bpm,
                    "notes": args.notes,
                    "pattern": args.pattern,
                    "control_dt": args.control_dt,
                    "episode": episode,
                },
                checkpoint,
            )

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
