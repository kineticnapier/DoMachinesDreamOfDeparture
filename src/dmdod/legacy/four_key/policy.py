from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - optional RL dependency
    raise ImportError(
        "PyTorch is required for dmdod.four_key_policy. Install the rl extra before importing this module."
    ) from exc

from dmdod.legacy.four_key.motor import FOUR_KEY_ACTION_DIM, FourKeyAction


FOUR_KEY_POLICY_VERSION = "four-key-gru-v1"


class FourKeyRecurrentActorCritic(nn.Module):
    """Four-output sibling of the existing recurrent motor policy.

    Parameter names intentionally mirror ``RecurrentActorCritic`` so the later
    BC/CUDA training path can be generalized without inventing a second network
    layout.  Existing two-key checkpoints are untouched.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 128,
        initial_log_std: float = -1.20,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = FOUR_KEY_ACTION_DIM
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
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
        encoded = torch.tanh(self.input_layer(x))
        next_state = self.gru(encoded, state)
        features = torch.tanh(self.post(next_state))
        mean = self.actor_mean(features)
        value = self.critic(features).squeeze(-1)
        std = self.log_std.exp().clamp(0.08, 1.5)
        return mean, std, value, next_state

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
        recurrent, final_state = torch._VF.gru(
            encoded,
            hx,
            [
                self.gru.weight_ih,
                self.gru.weight_hh,
                self.gru.bias_ih,
                self.gru.bias_hh,
            ],
            True,
            1,
            0.0,
            self.training,
            False,
            False,
        )
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
    ) -> tuple[FourKeyAction, torch.Tensor]:
        mean, _, _, next_state = self.forward_step(x, state)
        squashed = torch.tanh(mean)
        return FourKeyAction(*(float(value.item()) for value in squashed)), next_state

    def checkpoint_metadata(self) -> dict[str, int | str]:
        return {
            "four_key_policy_version": FOUR_KEY_POLICY_VERSION,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "action_dim": self.action_dim,
        }
