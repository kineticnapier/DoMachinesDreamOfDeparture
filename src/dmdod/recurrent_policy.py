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

    @staticmethod
    def _latent_log_prob(dist: Normal, latent: torch.Tensor) -> torch.Tensor:
        squashed = torch.tanh(latent)
        return dist.log_prob(latent).sum() - torch.log(
            1.0 - squashed.square() + 1e-6
        ).sum()

    def sample_action(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[MotorAction, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compatibility sampler used by the earlier one-episode A2C trainer."""

        action, latent, log_prob, value, entropy, next_state = self.sample_action_latent(
            x, state
        )
        del latent
        return action, log_prob, value, entropy, next_state

    def sample_action_latent(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[
        MotorAction,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Sample an action and expose its pre-tanh latent for PPO replay."""

        mean, std, value, next_state = self.forward_step(x, state)
        dist = Normal(mean, std)
        latent = dist.sample()
        squashed = torch.tanh(latent)
        log_prob = self._latent_log_prob(dist, latent)
        entropy = dist.entropy().sum()
        action = MotorAction(float(squashed[0].item()), float(squashed[1].item()))
        return action, latent, log_prob, value, entropy, next_state

    def evaluate_latent_sequence(
        self,
        observations: torch.Tensor,
        latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replay one complete episode through the GRU for recurrent PPO.

        Hidden state starts at zero exactly as it does at an environment reset.
        The caller supplies only the sequence of agent-visible observations and
        the actions sampled from the old policy; no privileged simulator state
        participates in the replay.
        """

        if observations.ndim != 2 or observations.shape[1] != self.input_dim:
            raise ValueError(
                f"observations must have shape [T, {self.input_dim}], got "
                f"{tuple(observations.shape)}"
            )
        if latents.ndim != 2 or latents.shape != (observations.shape[0], 2):
            raise ValueError(
                f"latents must have shape [T, 2], got {tuple(latents.shape)}"
            )

        state = self.initial_state(observations.device)
        log_probs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for x, latent in zip(observations, latents):
            mean, std, value, state = self.forward_step(x, state)
            dist = Normal(mean, std)
            log_probs.append(self._latent_log_prob(dist, latent))
            values.append(value)
            entropies.append(dist.entropy().sum())
        return (
            torch.stack(log_probs),
            torch.stack(values),
            torch.stack(entropies),
        )

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
