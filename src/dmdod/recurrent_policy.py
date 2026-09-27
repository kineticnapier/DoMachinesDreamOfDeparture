from __future__ import annotations

try:
    import torch
    from torch import nn
    from torch.distributions import Normal
except ImportError as exc:  # pragma: no cover - optional RL dependency
    raise ImportError(
        "PyTorch is required for dmdod.recurrent_policy. "
        "Install the rl extra before importing this module."
    ) from exc

from .motor_env import MotorAction


class RecurrentActorCritic(nn.Module):
    """GRU-based continuous actor-critic for frame-sequence control.

    The recurrent state is reset at episode boundaries and is never populated
    from privileged simulator state. It can therefore infer quantities such as
    apparent angular velocity, rotation direction and cycle count only from the
    sequence of agent-visible observations.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 64,
        initial_log_std: float = -0.70,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.post = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.actor_mean = nn.Linear(self.hidden_dim, 2)
        self.critic = nn.Linear(self.hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((2,), float(initial_log_std)))

    def initial_state(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.hidden_dim, dtype=torch.float32, device=device)

    def forward_step(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = torch.tanh(self.input_layer(x))
        next_state = self.gru(encoded, state)
        features = torch.tanh(self.post(next_state))
        mean = self.actor_mean(features)
        value = self.critic(features).squeeze(-1)
        std = self.log_std.exp().clamp(0.08, 1.5)
        return mean, std, value, next_state

    def sample_action(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[MotorAction, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std, value, next_state = self.forward_step(x, state)
        dist = Normal(mean, std)
        latent = dist.sample()
        squashed = torch.tanh(latent)
        log_prob = dist.log_prob(latent).sum() - torch.log(
            1.0 - squashed.square() + 1e-6
        ).sum()
        entropy = dist.entropy().sum()
        action = MotorAction(float(squashed[0].item()), float(squashed[1].item()))
        return action, log_prob, value, entropy, next_state

    @torch.no_grad()
    def deterministic_action(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[MotorAction, torch.Tensor]:
        mean, _, _, next_state = self.forward_step(x, state)
        squashed = torch.tanh(mean)
        action = MotorAction(float(squashed[0].item()), float(squashed[1].item()))
        return action, next_state
