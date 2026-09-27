from __future__ import annotations

try:
    import torch
    from torch import nn
    from torch.distributions import Normal
except ImportError as exc:  # pragma: no cover - optional RL dependency
    raise ImportError(
        "PyTorch is required for dmdod.toy_policy. "
        "Install the rl extra before importing this module."
    ) from exc

from .motor_env import MotorAction
from .rhythm_env import RhythmObservation


def observation_tensor(observation: RhythmObservation, device: torch.device) -> torch.Tensor:
    """Convert only agent-visible observations into the toy policy input tensor."""

    m = observation.motor
    # Fixed values here are unit conversions / broad physical scales, not chart
    # timing information.  Exact simulator/chart time remains unavailable.
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
    """Small continuous-action actor-critic used by the first toy RL task."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        initial_log_std: float = -1.20,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, 2)
        self.critic = nn.Linear(hidden_dim, 1)
        # The first version started at log_std=-0.35 (sigma ~= 0.70), which
        # produced so much random motion that many episodes hit OVERLOAD before
        # the first cue.  Start at sigma ~= 0.30 instead; log_std remains
        # learnable and entropy regularization may still increase exploration.
        self.log_std = nn.Parameter(torch.full((2,), float(initial_log_std)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mean = self.actor_mean(h)
        value = self.critic(h).squeeze(-1)
        std = self.log_std.exp().clamp(0.08, 1.5)
        return mean, std, value

    def sample_action(self, x: torch.Tensor) -> tuple[MotorAction, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a stochastic action for policy-gradient training."""

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

    @torch.no_grad()
    def deterministic_action(self, x: torch.Tensor) -> MotorAction:
        """Use tanh(actor mean) with no sampling/exploration noise."""

        mean, _, _ = self(x)
        squashed = torch.tanh(mean)
        return MotorAction(float(squashed[0].item()), float(squashed[1].item()))


def discounted_returns(rewards: list[float], gamma: float, device: torch.device) -> torch.Tensor:
    running = 0.0
    result: list[float] = []
    for reward in reversed(rewards):
        running = reward + gamma * running
        result.append(running)
    result.reverse()
    return torch.tensor(result, dtype=torch.float32, device=device)
