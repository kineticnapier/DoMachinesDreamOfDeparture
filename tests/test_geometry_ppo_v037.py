from __future__ import annotations

import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_geometry_ppo_v037 as trainer  # noqa: E402
from dmdod.evaluator import TargetHit  # noqa: E402
from dmdod.pattern_geometry_env import PatternMemoryGeometryEnv  # noqa: E402
from dmdod.planet_perception import PlanetVisionConfig  # noqa: E402
from dmdod.toy_policy import PATTERN_GEOMETRY_INPUT_DIM, observation_tensor  # noqa: E402


def test_recurrent_checkpoint_expansion_preserves_old_input_columns_and_zeroes_new_ones():
    old = trainer.base.core.RecurrentActorCritic(input_dim=10, hidden_dim=64)
    new = trainer.base.core.RecurrentActorCritic(
        input_dim=PATTERN_GEOMETRY_INPUT_DIM,
        hidden_dim=64,
    )

    with torch.no_grad():
        old.input_layer.weight.copy_(
            torch.arange(old.input_layer.weight.numel(), dtype=torch.float32).reshape_as(
                old.input_layer.weight
            )
            / 1000.0
        )
        old.actor_mean.bias.fill_(0.123)

    copied = trainer._copy_recurrent_weights(new, old.state_dict())

    assert "input_layer.weight" in copied
    assert torch.equal(new.input_layer.weight[:, :10], old.input_layer.weight)
    assert torch.count_nonzero(new.input_layer.weight[:, 10:]).item() == 0
    assert torch.equal(new.actor_mean.bias, old.actor_mean.bias)


def test_pattern_environment_produces_22_agent_visible_features():
    memory = trainer.PatternMemory(history_frames=3)
    env = PatternMemoryGeometryEnv(
        [TargetHit(0.75, "left")],
        bpm=180.0,
        pattern_memory=memory,
        chart_id="test-chart",
        control_dt_s=0.010,
        vision_config=PlanetVisionConfig(
            latency_s=0.0,
            latency_jitter_s=0.0,
            sample_period_s=0.0,
            position_noise_std=0.0,
            dropout_probability=0.0,
        ),
        perception_seed=1,
    )

    observation = env.reset()
    tensor = observation_tensor(observation, torch.device("cpu"))

    assert tensor.shape == (PATTERN_GEOMETRY_INPUT_DIM,)
    assert PATTERN_GEOMETRY_INPUT_DIM == 22


def test_chart_identity_ignores_phase_start_jitter_but_separates_visual_conditions():
    clean = trainer.base.core.clean_vision_config()
    sampled = PlanetVisionConfig(
        latency_s=0.05,
        latency_jitter_s=0.0,
        sample_period_s=1.0 / 60.0,
        position_noise_std=0.0,
        dropout_probability=0.0,
    )

    clean_id = trainer._chart_id(bpm=180.0, notes=1, config=clean)
    same_clean_id = trainer._chart_id(bpm=180.0, notes=1, config=clean)
    sampled_id = trainer._chart_id(bpm=180.0, notes=1, config=sampled)

    assert clean_id == same_clean_id
    assert clean_id != sampled_id
