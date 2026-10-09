from __future__ import annotations

"""Human-visible residual controller around the fixed MaleCNS reservoir.

The legacy fly-connectome policy compresses the entire 263D observation through
one sensory bottleneck before the fixed recurrent core. This backend keeps that
path intact, but also lets a trainable controller read the same human-visible
observation in its natural structure:

- motor position / velocity / pressed state plus orbiting-planet XY;
- the ordered visible-floor slots, encoded with a small 1D convolutional stack;
- the human-visible HUD slice (BPM and transient timing feedback);
- the fixed MaleCNS recurrent state.

No privileged chart time, target timestamp, absolute floor index, future
invisible floor, teacher action, evaluator overload value, or other hidden
state is introduced.

The action head is residual. actor_mean remains the inherited baseline readout
and controller_delta starts at exactly zero, so a model warm-started from a
fly-connectome checkpoint initially reproduces the parent's actions while the
new controller state is free to learn.
"""

from pathlib import Path

import torch
from torch import nn

from dmdod.features.hud_features import HUD_FEATURE_DIM
from dmdod.features.real_chart import (
    DEFAULT_REAL_CHART_FEATURE_CONFIG,
    REAL_CHART_FLOOR_FEATURE_DIM,
)
from dmdod.motor.n_key import NKeyAction

from .fly_policy import (
    DEFAULT_FLY_CONNECTOME_GAIN,
    DEFAULT_FLY_CONNECTOME_PROJECTION_SEED,
    DEFAULT_FLY_CONNECTOME_SENSORY_DIM,
    NKeyFlyConnectomeActorCritic,
)


N_KEY_POLICY_BACKEND_HUMAN_VISIBLE_CONTROLLER = "human_visible_controller"
N_KEY_HUMAN_VISIBLE_CONTROLLER_VERSION = "n-key-human-visible-residual-controller-v1"
DEFAULT_CONTROLLER_HIDDEN_DIM = 128
DEFAULT_CONNECTOME_CONTEXT_DIM = 64
DEFAULT_FLOOR_CONTEXT_DIM = 64
DEFAULT_MOTOR_CONTEXT_DIM = 48
DEFAULT_HUD_CONTEXT_DIM = 24


class NKeyHumanVisibleControllerActorCritic(NKeyFlyConnectomeActorCritic):
    """Fixed MaleCNS reservoir plus a trainable human-visible residual controller."""

    backend_name = N_KEY_POLICY_BACKEND_HUMAN_VISIBLE_CONTROLLER
    policy_version = N_KEY_HUMAN_VISIBLE_CONTROLLER_VERSION

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
        controller_hidden_dim: int = DEFAULT_CONTROLLER_HIDDEN_DIM,
        connectome_context_dim: int = DEFAULT_CONNECTOME_CONTEXT_DIM,
        floor_context_dim: int = DEFAULT_FLOOR_CONTEXT_DIM,
        motor_context_dim: int = DEFAULT_MOTOR_CONTEXT_DIM,
        hud_context_dim: int = DEFAULT_HUD_CONTEXT_DIM,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            key_count=key_count,
            core_path=core_path,
            sensory_dim=sensory_dim,
            recurrent_gain=recurrent_gain,
            projection_seed=projection_seed,
            initial_log_std=initial_log_std,
        )
        if controller_hidden_dim <= 0:
            raise ValueError("controller_hidden_dim must be positive")
        if min(
            connectome_context_dim,
            floor_context_dim,
            motor_context_dim,
            hud_context_dim,
        ) <= 0:
            raise ValueError("controller context dimensions must be positive")

        self.connectome_hidden_dim = int(self.hidden_dim)
        self.controller_hidden_dim = int(controller_hidden_dim)
        self.connectome_context_dim = int(connectome_context_dim)
        self.floor_context_dim = int(floor_context_dim)
        self.motor_context_dim = int(motor_context_dim)
        self.hud_context_dim = int(hud_context_dim)
        # One extra bit records whether the legacy fly no-grad state has
        # advanced at least once. The parent backend reuses its projection
        # scratch buffer as the returned state, so from the second inference
        # step onward that state is overwritten by the current projection
        # before recurrence. Preserve that established rollout behaviour here
        # without relying on tensor aliasing.
        self.policy_state_dim = (
            self.connectome_hidden_dim + self.controller_hidden_dim + 1
        )

        self.floor_slots = int(DEFAULT_REAL_CHART_FEATURE_CONFIG.floor_slots)
        self.floor_feature_dim = int(REAL_CHART_FLOOR_FEATURE_DIM)
        self.motor_feature_dim = 3 * int(self.key_count)
        self.orbit_feature_dim = 2
        self.hud_feature_dim = int(HUD_FEATURE_DIM)

        expected = (
            self.motor_feature_dim
            + self.orbit_feature_dim
            + self.floor_slots * self.floor_feature_dim
            + self.hud_feature_dim
        )
        if self.input_dim != expected:
            raise ValueError(
                f"human-visible controller expects structured HUD input {expected}D "
                f"for {self.key_count}K, got {self.input_dim}D"
            )

        self.connectome_context = nn.Sequential(
            nn.Linear(self.connectome_hidden_dim, self.connectome_context_dim),
            nn.Tanh(),
        )
        self.floor_encoder = nn.Sequential(
            nn.Conv1d(self.floor_feature_dim, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * self.floor_slots, self.floor_context_dim),
            nn.GELU(),
        )
        self.motor_encoder = nn.Sequential(
            nn.Linear(
                self.motor_feature_dim + self.orbit_feature_dim,
                self.motor_context_dim,
            ),
            nn.GELU(),
        )
        self.hud_encoder = nn.Sequential(
            nn.Linear(self.hud_feature_dim, self.hud_context_dim),
            nn.GELU(),
        )

        controller_input_dim = (
            self.connectome_context_dim
            + self.floor_context_dim
            + self.motor_context_dim
            + self.hud_context_dim
        )
        self.controller = nn.GRUCell(controller_input_dim, self.controller_hidden_dim)
        self.controller_post = nn.Sequential(
            nn.LayerNorm(self.controller_hidden_dim),
            nn.Linear(self.controller_hidden_dim, self.controller_hidden_dim),
            nn.GELU(),
        )
        self.controller_delta = nn.Linear(self.controller_hidden_dim, self.action_dim)
        self.controller_value_delta = nn.Linear(self.controller_hidden_dim, 1)

        # Exact behavioural warm start: until the residual head learns, actions
        # and values are those of the parent fly-connectome policy.
        nn.init.zeros_(self.controller_delta.weight)
        nn.init.zeros_(self.controller_delta.bias)
        nn.init.zeros_(self.controller_value_delta.weight)
        nn.init.zeros_(self.controller_value_delta.bias)

    def initial_state(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.policy_state_dim, dtype=torch.float32, device=device)

    def _split_state(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.ndim != 1 or state.shape[0] != self.policy_state_dim:
            raise ValueError(
                f"state must have shape [{self.policy_state_dim}], got {tuple(state.shape)}"
            )
        controller_end = self.connectome_hidden_dim + self.controller_hidden_dim
        return (
            state[: self.connectome_hidden_dim],
            state[self.connectome_hidden_dim : controller_end],
            state[controller_end:],
        )

    def _advance_parent_compatible(
        self,
        encoded: torch.Tensor,
        connectome_state: torch.Tensor,
        started: bool,
    ) -> torch.Tensor:
        """Match the established parent no-grad rollout semantics exactly.

        NKeyFlyConnectomeActorCritic returns its reusable projection buffer as
        the recurrent state during inference. On the next step that same buffer
        is filled with the new input projection before the recurrent multiply,
        so after the first step the recurrent input is the *current* projection.
        The relaxed controller stores state in a concatenated tensor and would
        otherwise silently change the inherited policy's behaviour.
        """

        injected = torch.mv(self.input_projection, encoded)
        recurrent_state = injected if started else connectome_state
        return self._advance_injected(injected, recurrent_state)

    def _split_observation(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 1 or x.shape[0] != self.input_dim:
            raise ValueError(
                f"observation must have shape [{self.input_dim}], got {tuple(x.shape)}"
            )

        motor_end = self.motor_feature_dim
        orbit_end = motor_end + self.orbit_feature_dim
        floors_end = orbit_end + self.floor_slots * self.floor_feature_dim

        motor_orbit = x[:orbit_end]
        floors = x[orbit_end:floors_end].reshape(
            self.floor_slots,
            self.floor_feature_dim,
        )
        hud = x[floors_end:]
        return motor_orbit, floors, hud

    def _direct_context(
        self,
        x: torch.Tensor,
        connectome_state: torch.Tensor,
    ) -> torch.Tensor:
        motor_orbit, floors, hud = self._split_observation(x)
        # Conv1d consumes [N, C, L]: floor features are channels and ordered
        # relative floor slots are the sequence axis.
        floor_context = self.floor_encoder(
            floors.transpose(0, 1).unsqueeze(0)
        ).squeeze(0)
        return torch.cat(
            (
                self.connectome_context(connectome_state),
                floor_context,
                self.motor_encoder(motor_orbit),
                self.hud_encoder(hud),
            ),
            dim=0,
        )

    def forward_step(self, x: torch.Tensor, state: torch.Tensor):
        connectome_state, controller_state, started_flag = self._split_state(state)
        if x.ndim != 1 or x.shape[0] != self.input_dim:
            raise ValueError(
                f"observation must have shape [{self.input_dim}], got {tuple(x.shape)}"
            )

        # The inherited sensory adapter and MaleCNS path are deliberately
        # frozen for this backend. Keep their recurrence outside autograd so
        # controller training does not build a 4096D graph it cannot update.
        with torch.no_grad():
            encoded = torch.tanh(self.sensory(x))
            next_connectome = self._advance_parent_compatible(
                encoded,
                connectome_state,
                bool(started_flag.item() >= 0.5),
            )

        context = self._direct_context(x, next_connectome)
        next_controller = self.controller(context, controller_state)
        controller_features = self.controller_post(next_controller)

        mean = self.actor_mean(next_connectome) + self.controller_delta(controller_features)
        value = (
            self.critic(next_connectome).squeeze(-1)
            + self.controller_value_delta(controller_features).squeeze(-1)
        )
        std = self.log_std.exp().clamp(0.08, 1.5)
        next_state = torch.cat(
            (
                next_connectome,
                next_controller,
                torch.ones(1, dtype=x.dtype, device=x.device),
            ),
            dim=0,
        )
        return mean, std, value, next_state

    def forward_sequence(
        self,
        observations: torch.Tensor,
        initial_state: torch.Tensor,
    ):
        if observations.ndim != 2 or observations.shape[1] != self.input_dim:
            raise ValueError(
                f"observations must have shape [T, {self.input_dim}], got "
                f"{tuple(observations.shape)}"
            )
        connectome_state, controller_state, started_flag = self._split_state(initial_state)
        if observations.shape[0] == 0:
            return (
                observations.new_empty((0, self.action_dim)),
                observations.new_empty((0,)),
                initial_state,
            )

        # Batch the sensory transform, then reproduce the parent's established
        # no-grad rollout semantics explicitly. This keeps exact warm-start
        # behaviour while avoiding dependence on scratch-buffer aliasing.
        with torch.no_grad():
            encoded_sequence = torch.tanh(self.sensory(observations))
            connectome_states: list[torch.Tensor] = []
            started = bool(started_flag.item() >= 0.5)
            for encoded in encoded_sequence:
                connectome_state = self._advance_parent_compatible(
                    encoded,
                    connectome_state,
                    started,
                )
                connectome_states.append(connectome_state)
                started = True
            connectome_stack = torch.stack(connectome_states)

        motor_end = self.motor_feature_dim
        orbit_end = motor_end + self.orbit_feature_dim
        floors_end = orbit_end + self.floor_slots * self.floor_feature_dim
        motor_orbit = observations[:, :orbit_end]
        floors = observations[:, orbit_end:floors_end].reshape(
            observations.shape[0],
            self.floor_slots,
            self.floor_feature_dim,
        ).transpose(1, 2)
        hud = observations[:, floors_end:]

        contexts = torch.cat(
            (
                self.connectome_context(connectome_stack),
                self.floor_encoder(floors),
                self.motor_encoder(motor_orbit),
                self.hud_encoder(hud),
            ),
            dim=1,
        )

        controller_states: list[torch.Tensor] = []
        for context in contexts:
            controller_state = self.controller(context, controller_state)
            controller_states.append(controller_state)
        controller_stack = torch.stack(controller_states)
        controller_features = self.controller_post(controller_stack)

        means = (
            self.actor_mean(connectome_stack)
            + self.controller_delta(controller_features)
        )
        values = (
            self.critic(connectome_stack).squeeze(-1)
            + self.controller_value_delta(controller_features).squeeze(-1)
        )
        final_state = torch.cat(
            (
                connectome_state,
                controller_state,
                torch.ones(
                    1,
                    dtype=observations.dtype,
                    device=observations.device,
                ),
            ),
            dim=0,
        )
        return means, values, final_state

    @torch.no_grad()
    def deterministic_action(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[NKeyAction, torch.Tensor]:
        mean, _, _, next_state = self.forward_step(x, state)
        squashed = torch.tanh(mean)
        return NKeyAction(tuple(float(value.item()) for value in squashed)), next_state

    def warm_start_from_fly_checkpoint(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load the inherited fly path while keeping new residual modules fresh."""

        result = self.load_state_dict(state_dict, strict=False)
        if result.unexpected_keys:
            raise ValueError(
                "unexpected keys in fly checkpoint: " + ", ".join(result.unexpected_keys)
            )

        new_prefixes = (
            "connectome_context.",
            "floor_encoder.",
            "motor_encoder.",
            "hud_encoder.",
            "controller.",
            "controller_post.",
            "controller_delta.",
            "controller_value_delta.",
        )
        invalid_missing = [
            key for key in result.missing_keys if not key.startswith(new_prefixes)
        ]
        if invalid_missing:
            raise ValueError(
                "fly checkpoint is missing inherited parameters: "
                + ", ".join(invalid_missing)
            )

        # load_state_dict leaves these at construction values, but state the
        # invariant explicitly in case initialization changes later.
        nn.init.zeros_(self.controller_delta.weight)
        nn.init.zeros_(self.controller_delta.bias)
        nn.init.zeros_(self.controller_value_delta.weight)
        nn.init.zeros_(self.controller_value_delta.bias)

    def checkpoint_metadata(self) -> dict[str, object]:
        metadata = super().checkpoint_metadata()
        metadata.update(
            {
                "n_key_policy_backend": self.backend_name,
                "n_key_policy_version": self.policy_version,
                "human_visible_controller_state_dim": self.policy_state_dim,
                "human_visible_controller_hidden_dim": self.controller_hidden_dim,
                "human_visible_connectome_context_dim": self.connectome_context_dim,
                "human_visible_floor_context_dim": self.floor_context_dim,
                "human_visible_motor_context_dim": self.motor_context_dim,
                "human_visible_hud_context_dim": self.hud_context_dim,
                "human_visible_floor_slots": self.floor_slots,
                "human_visible_floor_feature_dim": self.floor_feature_dim,
                "human_visible_motor_feature_dim": self.motor_feature_dim,
                "human_visible_hud_feature_dim": self.hud_feature_dim,
                "human_visible_information_contract": (
                    "visible-floor-geometry+motor+orbit+hud+fixed-connectome-state-v1"
                ),
                "human_visible_privileged_inputs": False,
                "human_visible_residual_warm_start": True,
            }
        )
        return metadata
