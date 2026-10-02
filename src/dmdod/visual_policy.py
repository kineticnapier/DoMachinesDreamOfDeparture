from __future__ import annotations

"""Level B pixel policy core.

The chart-derived policy input is RGB only.  ``frame_sequence_id`` is control
metadata used solely to decide whether the CNN result can be reused; it is never
concatenated into the neural-network input.
"""

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - optional RL dependency
    raise ImportError(
        "PyTorch is required for dmdod.visual_policy. Install the rl extra before importing this module."
    ) from exc

from .recurrent_policy import RecurrentActorCritic
from .visual_observation import (
    PROPRIOCEPTION_DIM,
    RgbFrame,
    VISUAL_OBSERVATION_VERSION,
)


VISUAL_POLICY_VERSION = "level-b-cnn128-gru128-v1"
DEFAULT_VISUAL_FEATURE_DIM = 128
DEFAULT_VISUAL_HIDDEN_DIM = 128


def rgb_frame_to_tensor(
    frame: RgbFrame,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert tightly packed RGB888 into a contiguous ``[3,H,W]`` uint8 tensor."""

    # bytearray gives torch writable storage and avoids the warning emitted for
    # a read-only ``bytes`` buffer.  Renderer IPC can later replace this copy
    # with shared memory without changing the policy-facing tensor layout.
    raw = torch.frombuffer(bytearray(frame.data), dtype=torch.uint8)
    tensor = raw.reshape(frame.height, frame.width, 3).permute(2, 0, 1).contiguous()
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


class SmallVisualEncoder(nn.Module):
    """Small CNN that maps an RGB frame to a fixed 128D visual feature by default."""

    def __init__(self, *, feature_dim: int = DEFAULT_VISUAL_FEATURE_DIM) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.feature_dim = int(feature_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 5)),
        )
        self.projection = nn.Linear(64 * 3 * 5, self.feature_dim)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        single = frame.ndim == 3
        if single:
            frame = frame.unsqueeze(0)
        if frame.ndim != 4 or frame.shape[1] != 3:
            raise ValueError(
                f"visual frame must have shape [3,H,W] or [B,3,H,W], got {tuple(frame.shape)}"
            )

        if frame.dtype == torch.uint8:
            x = frame.to(dtype=torch.float32) / 255.0
        elif frame.is_floating_point():
            x = frame.to(dtype=torch.float32)
        else:
            raise TypeError(f"visual frame dtype must be uint8 or floating point, got {frame.dtype}")

        x = self.conv(x)
        x = x.flatten(start_dim=1)
        feature = torch.tanh(self.projection(x))
        return feature.squeeze(0) if single else feature


@dataclass(slots=True)
class VisualFeatureCache:
    """Cache one encoded visual frame by renderer/policy frame sequence ID.

    The ID is intentionally metadata, not model input.  Reusing an ID means the
    caller promises that the visible RGB frame is the same sample and therefore
    the cached CNN feature is valid.
    """

    frame_sequence_id: int | None = None
    feature: torch.Tensor | None = None
    hits: int = 0
    misses: int = 0

    def reset(self) -> None:
        self.frame_sequence_id = None
        self.feature = None
        self.hits = 0
        self.misses = 0

    def get_or_encode(
        self,
        frame_sequence_id: int,
        frame: torch.Tensor,
        encoder: nn.Module,
    ) -> torch.Tensor:
        frame_sequence_id = int(frame_sequence_id)
        if self.feature is not None and self.frame_sequence_id == frame_sequence_id:
            self.hits += 1
            return self.feature

        feature = encoder(frame)
        if feature.ndim != 1:
            raise ValueError(
                f"cached visual encoder must return one feature vector, got {tuple(feature.shape)}"
            )
        self.frame_sequence_id = frame_sequence_id
        self.feature = feature
        self.misses += 1
        return feature


class LevelBVisualPolicy(nn.Module):
    """CNN + 6D proprioception + recurrent motor policy for Level B.

    The CNN may run at 60 Hz through ``VisualFeatureCache`` while the recurrent
    core still runs on every 100 Hz policy tick.  The recurrent model remains
    the existing actor-critic implementation so motor action semantics and GRU
    behavior stay aligned with the structured-input policy family.
    """

    def __init__(
        self,
        *,
        visual_feature_dim: int = DEFAULT_VISUAL_FEATURE_DIM,
        hidden_dim: int = DEFAULT_VISUAL_HIDDEN_DIM,
        initial_log_std: float = -0.70,
    ) -> None:
        super().__init__()
        self.visual_feature_dim = int(visual_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.visual_encoder = SmallVisualEncoder(feature_dim=self.visual_feature_dim)
        self.recurrent = RecurrentActorCritic(
            input_dim=self.visual_feature_dim + PROPRIOCEPTION_DIM,
            hidden_dim=self.hidden_dim,
            initial_log_std=initial_log_std,
        )

    def initial_state(self, device: torch.device) -> torch.Tensor:
        return self.recurrent.initial_state(device)

    def forward_step(
        self,
        frame: torch.Tensor,
        proprioception: torch.Tensor,
        state: torch.Tensor,
        *,
        frame_sequence_id: int | None = None,
        visual_cache: VisualFeatureCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if proprioception.ndim != 1 or proprioception.shape[0] != PROPRIOCEPTION_DIM:
            raise ValueError(
                f"proprioception must have shape [{PROPRIOCEPTION_DIM}], got {tuple(proprioception.shape)}"
            )

        if visual_cache is None:
            visual = self.visual_encoder(frame)
        else:
            if frame_sequence_id is None:
                raise ValueError("frame_sequence_id is required when visual_cache is used")
            visual = visual_cache.get_or_encode(
                frame_sequence_id,
                frame,
                self.visual_encoder,
            )

        if visual.ndim != 1 or visual.shape[0] != self.visual_feature_dim:
            raise ValueError(
                f"visual feature must have shape [{self.visual_feature_dim}], got {tuple(visual.shape)}"
            )
        if visual.device != proprioception.device:
            raise ValueError("visual feature and proprioception must be on the same device")

        combined = torch.cat((visual, proprioception.to(dtype=visual.dtype)), dim=0)
        return self.recurrent.forward_step(combined, state)

    def checkpoint_metadata(self) -> dict[str, int | str]:
        """Version fields that training checkpoints should persist beside state dicts."""

        return {
            "visual_policy_version": VISUAL_POLICY_VERSION,
            "visual_observation_version": VISUAL_OBSERVATION_VERSION,
            "visual_feature_dim": self.visual_feature_dim,
            "proprioception_dim": PROPRIOCEPTION_DIM,
            "hidden_dim": self.hidden_dim,
        }
