from __future__ import annotations

import math

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - optional RL dependency
    raise ImportError(
        "PyTorch is required for dmdod.n_key_policy. Install the rl extra before importing this module."
    ) from exc

from dmdod.motor.n_key import NKeyAction, n_key_names


N_KEY_POLICY_VERSION = "n-key-gru-v1"
N_KEY_SPARSE_RESERVOIR_VERSION = "n-key-fixed-sparse-reservoir-v1"
N_KEY_POLICY_BACKEND_GRU = "gru"
N_KEY_POLICY_BACKEND_SPARSE_RESERVOIR = "sparse_reservoir"
N_KEY_POLICY_BACKENDS = (
    N_KEY_POLICY_BACKEND_GRU,
    N_KEY_POLICY_BACKEND_SPARSE_RESERVOIR,
)
DEFAULT_SPARSE_RESERVOIR_DENSITY = 0.10
DEFAULT_SPARSE_RESERVOIR_GAIN = 0.90
DEFAULT_SPARSE_RESERVOIR_SEED = 1701


class NKeyPolicyBase(nn.Module):
    """Backend-neutral contract used by N-key training and evaluation code.

    Future recurrent cores (for example a fixed sparse reservoir or a
    connectome-derived front end) should implement the same state/forward
    contract instead of making the training loop depend on one concrete GRU.
    """

    backend_name = "base"
    policy_version = "n-key-policy-base-v1"

    def __init__(
        self,
        *,
        input_dim: int,
        key_count: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.key_count = int(key_count)
        self.key_names = n_key_names(self.key_count)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = self.key_count

    def prepare_recurrent_runtime(self) -> None:
        """Allow a backend to prepare optimized recurrent runtime state."""

    def initial_state(self, device: torch.device) -> torch.Tensor:
        raise NotImplementedError

    def forward_step(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward_sequence(
        self,
        observations: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @torch.no_grad()
    def deterministic_action(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[NKeyAction, torch.Tensor]:
        mean, _, _, next_state = self.forward_step(x, state)
        squashed = torch.tanh(mean)
        return NKeyAction(tuple(float(value.item()) for value in squashed)), next_state

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "n_key_policy_backend": self.backend_name,
            "n_key_policy_version": self.policy_version,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "action_dim": self.action_dim,
            "key_count": self.key_count,
            "key_names": self.key_names,
        }


class NKeyRecurrentActorCritic(NKeyPolicyBase):
    """Configurable-output GRU policy for 2K/4K/6K/8K Level A bodies."""

    backend_name = N_KEY_POLICY_BACKEND_GRU
    policy_version = N_KEY_POLICY_VERSION

    def __init__(
        self,
        *,
        input_dim: int,
        key_count: int,
        hidden_dim: int = 128,
        initial_log_std: float = -1.20,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            key_count=key_count,
            hidden_dim=hidden_dim,
        )

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

    def prepare_recurrent_runtime(self) -> None:
        self.gru.flatten_parameters()

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


class NKeyFixedSparseReservoirActorCritic(NKeyPolicyBase):
    """Trainable sensory/readout layers around a fixed sparse recurrent core.

    The recurrent topology and recurrent weights never learn.  This is an
    artificial sparse-reservoir baseline for later connectome-derived backends,
    not a biological model.  A trainable sensory adapter feeds a fixed tanh RNN
    whose input transform is identity and whose recurrent matrix is sparsified
    once at construction time.  Only the sensory adapter and readout heads are
    optimized by the existing BC/DAgger pipeline.
    """

    backend_name = N_KEY_POLICY_BACKEND_SPARSE_RESERVOIR
    policy_version = N_KEY_SPARSE_RESERVOIR_VERSION

    def __init__(
        self,
        *,
        input_dim: int,
        key_count: int,
        hidden_dim: int = 128,
        initial_log_std: float = -1.20,
        reservoir_density: float = DEFAULT_SPARSE_RESERVOIR_DENSITY,
        reservoir_gain: float = DEFAULT_SPARSE_RESERVOIR_GAIN,
        reservoir_seed: int = DEFAULT_SPARSE_RESERVOIR_SEED,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            key_count=key_count,
            hidden_dim=hidden_dim,
        )
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not (0.0 < reservoir_density <= 1.0):
            raise ValueError("reservoir_density must be in (0, 1]")
        if reservoir_gain <= 0.0:
            raise ValueError("reservoir_gain must be positive")

        self.reservoir_density = float(reservoir_density)
        self.reservoir_gain = float(reservoir_gain)
        self.reservoir_seed = int(reservoir_seed)

        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.reservoir = nn.RNN(
            self.hidden_dim,
            self.hidden_dim,
            nonlinearity="tanh",
            bias=False,
        )
        self._initialize_fixed_reservoir()
        self.post = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.actor_mean = nn.Linear(self.hidden_dim, self.action_dim)
        self.critic = nn.Linear(self.hidden_dim, 1)
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), float(initial_log_std))
        )

    def _initialize_fixed_reservoir(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.reservoir_seed)
        density = self.reservoir_density
        fan_in = max(1.0, density * float(self.hidden_dim))

        with torch.no_grad():
            self.reservoir.weight_ih_l0.copy_(torch.eye(self.hidden_dim))
            mask = torch.rand(
                self.hidden_dim,
                self.hidden_dim,
                generator=generator,
            ) < density
            weights = torch.randn(
                self.hidden_dim,
                self.hidden_dim,
                generator=generator,
            )
            weights.mul_(self.reservoir_gain / math.sqrt(fan_in))
            weights.mul_(mask)
            self.reservoir.weight_hh_l0.copy_(weights)

        self.reservoir.weight_ih_l0.requires_grad_(False)
        self.reservoir.weight_hh_l0.requires_grad_(False)

    def prepare_recurrent_runtime(self) -> None:
        self.reservoir.flatten_parameters()

    def initial_state(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.hidden_dim, dtype=torch.float32, device=device)

    def _validate_state(self, state: torch.Tensor) -> None:
        if state.ndim != 1 or state.shape[0] != self.hidden_dim:
            raise ValueError(
                f"state must have shape [{self.hidden_dim}], got {tuple(state.shape)}"
            )

    def forward_step(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 1 or x.shape[0] != self.input_dim:
            raise ValueError(
                f"observation must have shape [{self.input_dim}], got {tuple(x.shape)}"
            )
        self._validate_state(state)

        encoded = torch.tanh(self.input_layer(x)).reshape(1, 1, self.hidden_dim)
        hx = state.reshape(1, 1, self.hidden_dim)
        recurrent, next_state = self.reservoir(encoded, hx)
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
        self._validate_state(initial_state)
        if observations.shape[0] == 0:
            return (
                observations.new_empty((0, self.action_dim)),
                observations.new_empty((0,)),
                initial_state,
            )

        encoded = torch.tanh(self.input_layer(observations)).unsqueeze(1)
        hx = initial_state.reshape(1, 1, self.hidden_dim)
        recurrent, final_state = self.reservoir(encoded, hx)
        recurrent = recurrent.squeeze(1)
        features = torch.tanh(self.post(recurrent))
        means = self.actor_mean(features)
        values = self.critic(features).squeeze(-1)
        return means, values, final_state.reshape(self.hidden_dim)

    def checkpoint_metadata(self) -> dict[str, object]:
        metadata = super().checkpoint_metadata()
        metadata.update(
            {
                "reservoir_density": self.reservoir_density,
                "reservoir_gain": self.reservoir_gain,
                "reservoir_seed": self.reservoir_seed,
            }
        )
        return metadata


def build_n_key_policy(
    *,
    backend: str,
    input_dim: int,
    key_count: int,
    hidden_dim: int = 128,
    initial_log_std: float = -1.20,
    reservoir_density: float = DEFAULT_SPARSE_RESERVOIR_DENSITY,
    reservoir_gain: float = DEFAULT_SPARSE_RESERVOIR_GAIN,
    reservoir_seed: int = DEFAULT_SPARSE_RESERVOIR_SEED,
) -> NKeyPolicyBase:
    """Construct one registered N-key policy backend."""

    backend = str(backend)
    if backend == N_KEY_POLICY_BACKEND_GRU:
        return NKeyRecurrentActorCritic(
            input_dim=input_dim,
            key_count=key_count,
            hidden_dim=hidden_dim,
            initial_log_std=initial_log_std,
        )
    if backend == N_KEY_POLICY_BACKEND_SPARSE_RESERVOIR:
        return NKeyFixedSparseReservoirActorCritic(
            input_dim=input_dim,
            key_count=key_count,
            hidden_dim=hidden_dim,
            initial_log_std=initial_log_std,
            reservoir_density=reservoir_density,
            reservoir_gain=reservoir_gain,
            reservoir_seed=reservoir_seed,
        )
    choices = ", ".join(N_KEY_POLICY_BACKENDS)
    raise ValueError(f"unknown N-key policy backend {backend!r}; expected one of: {choices}")


def n_key_policy_backend_from_checkpoint(checkpoint: dict) -> str:
    """Read backend metadata while keeping legacy GRU checkpoints loadable."""

    backend = str(checkpoint.get("n_key_policy_backend", N_KEY_POLICY_BACKEND_GRU))
    if backend not in N_KEY_POLICY_BACKENDS:
        choices = ", ".join(N_KEY_POLICY_BACKENDS)
        raise ValueError(
            f"checkpoint uses unsupported N-key policy backend {backend!r}; "
            f"expected one of: {choices}"
        )
    return backend
