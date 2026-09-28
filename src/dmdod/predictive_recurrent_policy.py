from __future__ import annotations

try:
    import torch
    from torch.distributions import Normal
except ImportError as exc:  # pragma: no cover - optional RL dependency
    raise ImportError(
        "PyTorch is required for dmdod.predictive_recurrent_policy. "
        "Install the rl extra before importing this module."
    ) from exc

from .recurrent_policy import RecurrentActorCritic


class PredictiveRecurrentActorCritic(RecurrentActorCritic):
    """Recurrent actor-critic with a self-supervised visual-motion head.

    The auxiliary head predicts the next visible frame delta from the current
    recurrent representation.  Its training target is constructed only from
    consecutive policy observations; no BPM, chart timestamp, target angle or
    timing error is used.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 64,
        initial_log_std: float = -0.70,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            initial_log_std=initial_log_std,
        )
        self.motion_predictor = torch.nn.Linear(self.hidden_dim, 4)

    def evaluate_latent_sequence_predictive(
        self,
        observations: torch.Tensor,
        latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replay an episode and predict the next visible motion at each step."""

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
        predictions: list[torch.Tensor] = []

        for x, latent in zip(observations, latents):
            mean, std, value, state = self.forward_step(x, state)
            dist = Normal(mean, std)
            log_probs.append(self._latent_log_prob(dist, latent))
            values.append(value)
            entropies.append(dist.entropy().sum())
            features = torch.tanh(self.post(state))
            predictions.append(self.motion_predictor(features))

        return (
            torch.stack(log_probs),
            torch.stack(values),
            torch.stack(entropies),
            torch.stack(predictions),
        )
