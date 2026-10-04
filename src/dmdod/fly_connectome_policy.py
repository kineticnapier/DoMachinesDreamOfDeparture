from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from .malecns_connectome import MALECNS_DATASET, load_malecns_core
from .n_key_policy import NKeyPolicyBase


N_KEY_POLICY_BACKEND_FLY_CONNECTOME = "fly_connectome"
N_KEY_FLY_CONNECTOME_VERSION = "n-key-malecns-fixed-connectome-v1"
DEFAULT_FLY_CONNECTOME_GAIN = 0.90
DEFAULT_FLY_CONNECTOME_SENSORY_DIM = 128
DEFAULT_FLY_CONNECTOME_PROJECTION_SEED = 1701


class _FixedSparseMv(torch.autograd.Function):
    """Sparse matvec with a fixed weight and an explicit fixed transpose backward.

    PyTorch's generic CSR ``mv`` backward computes gradients for a sparse matrix
    input even when that matrix is a non-trainable buffer. On CUDA that path can
    repeatedly convert CSR/COO layouts and sort indices. The MaleCNS recurrent
    graph is fixed, so only the dense state gradient is required:

        y = W @ x        =>        dL/dx = W^T @ dL/dy

    Both CSR matrices are prepared once by the policy and are intentionally
    excluded from checkpoint state.
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        weight_transpose: torch.Tensor,
        vector: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(weight_transpose)
        return torch.mv(weight, vector)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (weight_transpose,) = ctx.saved_tensors
        grad_vector = torch.mv(weight_transpose, grad_output)
        return None, None, grad_vector


class NKeyFlyConnectomeActorCritic(NKeyPolicyBase):
    """N-key actor-critic around a fixed MaleCNS recurrent topology.

    The policy keeps one recurrent state value per selected MaleCNS neuron.
    Trainable observations first enter a small sensory bottleneck; a fixed,
    seeded projection fans that bottleneck into the connectome. Recurrent
    weights preserve the artifact topology and relative raw synapse strengths,
    but are row-normalized and globally scaled for stable tanh dynamics.

    This first backend is intentionally topology-only: the current MaleCNS
    artifact is unsigned, so neurotransmitter-derived excitation/inhibition is
    not modeled yet. The connectome and input projection are fixed; only the
    sensory bottleneck, actor/critic readouts, and log standard deviation learn.
    """

    backend_name = N_KEY_POLICY_BACKEND_FLY_CONNECTOME
    policy_version = N_KEY_FLY_CONNECTOME_VERSION

    def __init__(
        self,
        *,
        input_dim: int,
        key_count: int,
        core_path: str | Path,
        sensory_dim: int = DEFAULT_FLY_CONNECTOME_SENSORY_DIM,
        recurrent_gain: float = DEFAULT_FLY_CONNECTOME_GAIN,
        projection_seed: int = DEFAULT_FLY_CONNECTOME_PROJECTION_SEED,
        initial_log_std: float = -1.20,
    ) -> None:
        artifact = load_malecns_core(core_path)
        metadata = dict(artifact["metadata"])
        node_count = int(metadata["node_count"])
        if node_count <= 0:
            raise ValueError("MaleCNS core node_count must be positive")
        if sensory_dim <= 0:
            raise ValueError("sensory_dim must be positive")
        if recurrent_gain <= 0.0:
            raise ValueError("recurrent_gain must be positive")

        super().__init__(
            input_dim=input_dim,
            key_count=key_count,
            hidden_dim=node_count,
        )
        self.core_path = str(Path(core_path))
        self.sensory_dim = int(sensory_dim)
        self.recurrent_gain = float(recurrent_gain)
        self.projection_seed = int(projection_seed)
        self.core_metadata = metadata

        self.sensory = nn.Linear(self.input_dim, self.sensory_dim)
        self.actor_mean = nn.Linear(self.hidden_dim, self.action_dim)
        self.critic = nn.Linear(self.hidden_dim, 1)
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), float(initial_log_std))
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.projection_seed)
        projection = torch.randn(
            self.hidden_dim,
            self.sensory_dim,
            generator=generator,
            dtype=torch.float32,
        ) / math.sqrt(float(self.sensory_dim))
        self.register_buffer("input_projection", projection)

        edge_index = torch.as_tensor(
            artifact["edge_index"], dtype=torch.int64, device="cpu"
        )
        raw_weight = torch.as_tensor(
            artifact["edge_weight"], dtype=torch.float32, device="cpu"
        )
        target = edge_index[0]
        incoming = torch.zeros(self.hidden_dim, dtype=torch.float32)
        incoming.scatter_add_(0, target, raw_weight)
        denominator = incoming[target].clamp_min(1.0)
        normalized_weight = raw_weight * (self.recurrent_gain / denominator)
        recurrent = torch.sparse_coo_tensor(
            edge_index,
            normalized_weight,
            (self.hidden_dim, self.hidden_dim),
            dtype=torch.float32,
            check_invariants=True,
        ).coalesce()
        self.register_buffer("recurrent_weight", recurrent)
        # Runtime-only CSR acceleration structures. Keep only canonical COO in
        # checkpoints so the existing checkpoint schema remains unchanged.
        self.register_buffer(
            "_recurrent_weight_runtime",
            recurrent.to_sparse_csr(),
            persistent=False,
        )
        self.register_buffer(
            "_recurrent_weight_transpose_runtime",
            recurrent.transpose(0, 1).coalesce().to_sparse_csr(),
            persistent=False,
        )
        self.register_buffer(
            "body_ids",
            torch.as_tensor(artifact["body_ids"], dtype=torch.int64, device="cpu"),
        )

    def _refresh_recurrent_runtime_weight(self) -> None:
        """Rebuild fixed CSR forward/backward views from the canonical COO graph."""
        self._recurrent_weight_runtime = self.recurrent_weight.to_sparse_csr()
        self._recurrent_weight_transpose_runtime = (
            self.recurrent_weight.transpose(0, 1).coalesce().to_sparse_csr()
        )

    def prepare_recurrent_runtime(self) -> None:
        # Rebuild after ``to(device)`` / checkpoint loading so both runtime
        # matrices exactly match the canonical recurrent graph on that device.
        self._refresh_recurrent_runtime_weight()

    def initial_state(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.hidden_dim, dtype=torch.float32, device=device)

    def _validate_state(self, state: torch.Tensor) -> None:
        if state.ndim != 1 or state.shape[0] != self.hidden_dim:
            raise ValueError(
                f"state must have shape [{self.hidden_dim}], got {tuple(state.shape)}"
            )

    def _advance_injected(
        self,
        injected: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        # Evaluation does not need a backward graph, so keep the direct CSR mv
        # fast path. During training, use the fixed-weight custom backward to
        # avoid PyTorch's generic sparse-weight gradient machinery.
        if torch.is_grad_enabled():
            recurrent = _FixedSparseMv.apply(
                self._recurrent_weight_runtime,
                self._recurrent_weight_transpose_runtime,
                state,
            )
        else:
            recurrent = torch.mv(self._recurrent_weight_runtime, state)
        return torch.tanh(injected + recurrent)

    def _advance(self, encoded: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        injected = torch.mv(self.input_projection, encoded)
        return self._advance_injected(injected, state)

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
        encoded = torch.tanh(self.sensory(x))
        next_state = self._advance(encoded, state)
        mean = self.actor_mean(next_state)
        value = self.critic(next_state).squeeze(-1)
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
        self._validate_state(initial_state)
        if observations.shape[0] == 0:
            return (
                observations.new_empty((0, self.action_dim)),
                observations.new_empty((0,)),
                initial_state,
            )

        encoded = torch.tanh(self.sensory(observations))
        injected_sequence = torch.matmul(encoded, self.input_projection.transpose(0, 1))
        state = initial_state
        states: list[torch.Tensor] = []
        for injected in injected_sequence:
            state = self._advance_injected(injected, state)
            states.append(state)
        recurrent = torch.stack(states)
        means = self.actor_mean(recurrent)
        values = self.critic(recurrent).squeeze(-1)
        return means, values, state

    def checkpoint_metadata(self) -> dict[str, object]:
        metadata = super().checkpoint_metadata()
        metadata.update(
            {
                "fly_connectome_dataset": MALECNS_DATASET,
                "fly_connectome_core_path": self.core_path,
                "fly_connectome_node_count": self.hidden_dim,
                "fly_connectome_edge_count": int(self.recurrent_weight._nnz()),
                "fly_connectome_signed": bool(self.core_metadata.get("signed", False)),
                "fly_connectome_sensory_dim": self.sensory_dim,
                "fly_connectome_recurrent_gain": self.recurrent_gain,
                "fly_connectome_projection_seed": self.projection_seed,
            }
        )
        return metadata
