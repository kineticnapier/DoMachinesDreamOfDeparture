import torch

from dmdod import PlanetVisionConfig, TargetHit
from dmdod.geometry_rhythm_env import GeometryRhythmEnv
from dmdod.toy_policy import GEOMETRY_INPUT_DIM, observation_tensor


def _clean_vision() -> PlanetVisionConfig:
    return PlanetVisionConfig(
        latency_s=0.0,
        latency_jitter_s=0.0,
        sample_period_s=0.0,
        position_noise_std=0.0,
        dropout_probability=0.0,
    )


def test_geometry_env_exposes_geometry_not_gaussian_cue():
    env = GeometryRhythmEnv(
        [TargetHit(0.5, "left")],
        bpm=120.0,
        vision_config=_clean_vision(),
    )
    observation = env.reset()

    assert hasattr(observation, "geometry")
    assert not hasattr(observation, "cue")
    assert observation.geometry.next_x == 1.0
    assert observation.geometry.next_y == 0.0


def test_geometry_policy_tensor_has_expected_dimension():
    env = GeometryRhythmEnv(
        [TargetHit(0.5, "left")],
        bpm=120.0,
        vision_config=_clean_vision(),
    )
    tensor = observation_tensor(env.reset(), torch.device("cpu"))
    assert tensor.shape == (GEOMETRY_INPUT_DIM,)
