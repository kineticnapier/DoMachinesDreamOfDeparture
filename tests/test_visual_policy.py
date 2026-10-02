from __future__ import annotations

import torch
from torch import nn

from dmdod.visual_policy import (
    DEFAULT_VISUAL_FEATURE_DIM,
    LevelBVisualPolicy,
    SmallVisualEncoder,
    VisualFeatureCache,
)


def _frame(value: int = 0) -> torch.Tensor:
    return torch.full((3, 180, 320), value, dtype=torch.uint8)


def _proprioception() -> torch.Tensor:
    return torch.tensor((0.1, -0.1, 0.2, -0.2, 1.0, 0.0), dtype=torch.float32)


def test_small_visual_encoder_maps_320x180_rgb_to_128d_feature() -> None:
    encoder = SmallVisualEncoder()
    feature = encoder(_frame(127))

    assert feature.shape == (DEFAULT_VISUAL_FEATURE_DIM,)
    assert feature.dtype == torch.float32


class _CountingEncoder(nn.Module):
    def __init__(self, feature_dim: int = DEFAULT_VISUAL_FEATURE_DIM) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.calls = 0

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        value = frame.to(dtype=torch.float32).mean() / 255.0
        return torch.full(
            (self.feature_dim,),
            float(value.item()),
            dtype=torch.float32,
            device=frame.device,
        )


def test_visual_feature_cache_is_keyed_by_frame_sequence_id_not_rgb_comparison() -> None:
    encoder = _CountingEncoder()
    cache = VisualFeatureCache()

    first = cache.get_or_encode(10, _frame(0), encoder)
    repeated_id_different_pixels = cache.get_or_encode(10, _frame(255), encoder)
    next_frame = cache.get_or_encode(11, _frame(255), encoder)

    assert encoder.calls == 2
    assert cache.hits == 1
    assert cache.misses == 2
    assert repeated_id_different_pixels is first
    assert not torch.equal(next_frame, first)


def test_level_b_policy_reuses_cnn_feature_but_runs_recurrent_core_every_tick() -> None:
    policy = LevelBVisualPolicy()
    counting = _CountingEncoder(policy.visual_feature_dim)
    policy.visual_encoder = counting
    cache = VisualFeatureCache()
    state = policy.initial_state(torch.device("cpu"))

    mean1, std1, value1, state1 = policy.forward_step(
        _frame(32),
        _proprioception(),
        state,
        frame_sequence_id=7,
        visual_cache=cache,
    )
    mean2, std2, value2, state2 = policy.forward_step(
        _frame(200),
        _proprioception(),
        state1,
        frame_sequence_id=7,
        visual_cache=cache,
    )

    assert counting.calls == 1
    assert cache.misses == 1
    assert cache.hits == 1
    assert mean1.shape == mean2.shape == (2,)
    assert std1.shape == std2.shape == (2,)
    assert value1.ndim == value2.ndim == 0
    assert state1.shape == state2.shape == (policy.hidden_dim,)
    assert not torch.equal(state1, state2)


def test_level_b_policy_state_dict_and_checkpoint_metadata_cover_visual_and_recurrent_parts() -> None:
    policy = LevelBVisualPolicy()
    keys = tuple(policy.state_dict().keys())
    metadata = policy.checkpoint_metadata()

    assert any(key.startswith("visual_encoder.") for key in keys)
    assert any(key.startswith("recurrent.gru.") for key in keys)
    assert any(key.startswith("recurrent.actor_mean.") for key in keys)
    assert metadata["visual_feature_dim"] == 128
    assert metadata["proprioception_dim"] == 6
    assert metadata["hidden_dim"] == 128
