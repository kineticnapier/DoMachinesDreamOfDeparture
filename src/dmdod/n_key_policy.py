from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - optional RL dependency
    raise ImportError(
        "PyTorch is required for dmdod.n_key_policy. Install the rl extra before importing this module."
    ) from exc

from .n_key_motor import NKeyAction, n_key_names


N_KEY_POLICY_VERSION = "n-key-gru-v1"


class NKeyRecurrentActorCritic(nn.Module):
    """Configurable-output GRU policy for 2K/4K/6K/8K Level A bodies."""

    def __init__(
        self,
        *,
        input_dim: int,
        key_count: int,
        hidden_dim: int = 128,
        initial_log_std: float = -1.20,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.key_count = int(key_count)
        self.key_names = n_key_names(self.key_count)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = self.key_count

        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        # Use a real nn.GRU module so CUDA can maintain its packed contiguous
        # weight layout.  This avoids the direct torch._VF.gru warning emitted
        # by the first four-key bootstrap implementation.
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.post = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.actor_mean = nn.Linear(self.hidden_dim, self.action_dim)
        self.critic = nn.Linear(self.hidden_dim, 1)
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), float(initial_log_std))
        )

    def initial_state(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.hidden_dim, dtype=torch.float32, device=device)

    def forward_step(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 1 or x.shape[0] != self.input_dim:
            raise ValueError(
                f"observation must have shape [{self.input_dim}], got {tuple(x.shape)}"
            )
        if state.ndim != 1 or state.shape[0] != self.hidden_dim:
            raise ValueError(
                f"state must have shape [{self.hidden_dim}], got {tuple(state.shape)}"
            )

        encoded = torch.tanh(self.input_layer(x)).reshape(1, 1, self.hidden_dim)
        hx = state.reshape(1, 1, self.hidden_dim)
        recurrent, next_state = self.gru(encoded, hx)
        features = torch.tanh(self.post(recurrent.reshape(self.hidden_dim)))
        mean = self.actor_mean(features)
        value = self.critic(features).squeeze(-1)
        std = self.log_std.exp().clamp(0.08, 1.5)
        return mean, std, value, next_state.reshape(self.hidden_dim)

    def forward_sequence(
        self,
        observations: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observations.ndim != 2 or observations.shape[1] != self.input_dim:
            raise ValueError(
                f"observations must have shape [T, {self.input_dim}], got "
                f"{tuple(observations.shape)}"
            )
        if initial_state.ndim != 1 or initial_state.shape[0] != self.hidden_dim:
            raise ValueError(
                f"initial_state must have shape [{self.hidden_dim}], got "
                f"{tuple(initial_state.shape)}"
            )
        if observations.shape[0] == 0:
            return (
                observations.new_empty((0, self.action_dim)),
                observations.new_empty((0,)),
                initial_state,
            )

        encoded = torch.tanh(self.input_layer(observations)).unsqueeze(1)
        hx = initial_state.reshape(1, 1, self.hidden_dim)
        recurrent, final_state = self.gru(encoded, hx)
        recurrent = recurrent.squeeze(1)
        features = torch.tanh(self.post(recurrent))
        means = self.actor_mean(features)
        values = self.critic(features).squeeze(-1)
        return means, values, final_state.reshape(self.hidden_dim)

    @torch.no_grad()
    def deterministic_action(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[NKeyAction, torch.Tensor]:
        mean, _, _, next_state = self.forward_step(x, state)
        squashed = torch.tanh(mean)
        return NKeyAction(tuple(float(value.item()) for value in squashed)), next_state

    def checkpoint_metadata(self) -> dict[str, int | str | tuple[str, ...]]:
        return {
            "n_key_policy_version": N_KEY_POLICY_VERSION,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "action_dim": self.action_dim,
            "key_count": self.key_count,
            "key_names": self.key_names,
        }
