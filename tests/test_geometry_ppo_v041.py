from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_geometry_ppo_v041 as trainer  # noqa: E402
from dmdod.parallel_rollout import ParallelRolloutSpec, collect_rollout_batch  # noqa: E402


def _spec(index: int, bpm: float) -> ParallelRolloutSpec:
    return ParallelRolloutSpec(
        index=index,
        notes=1,
        bpm=bpm,
        start_s=0.450,
        control_dt=0.010,
        vision={
            "latency_s": 0.0,
            "latency_jitter_s": 0.0,
            "sample_period_s": 0.0,
            "position_noise_std": 0.0,
            "dropout_probability": 0.0,
        },
        env_seed=100 + index,
        policy_seed=200 + index,
        gamma=0.995,
    )


def test_parallel_worker_batch_returns_ppo_payloads_in_requested_indices():
    model = trainer.v038.PredictiveRecurrentActorCritic(
        input_dim=trainer.v038.MOTION_GEOMETRY_INPUT_DIM,
        hidden_dim=16,
    )
    payloads = collect_rollout_batch(
        {name: value.detach().cpu() for name, value in model.state_dict().items()},
        16,
        [_spec(2, 240.0), _spec(0, 145.0)],
    )

    assert [item["index"] for item in payloads] == [2, 0]
    for item in payloads:
        steps = len(item["observations"])
        assert steps > 0
        assert len(item["observations"][0]) == trainer.v038.MOTION_GEOMETRY_INPUT_DIM
        assert len(item["latents"]) == steps
        assert len(item["old_log_probs"]) == steps
        assert len(item["old_values"]) == steps
        assert len(item["returns"]) == steps
        assert len(item["advantages"]) == steps
        assert math.isfinite(float(item["reward"]))


def test_parallel_chunks_balance_round_robin_and_preserve_all_specs():
    specs = [_spec(i, 145.0 + i) for i in range(10)]
    chunks = trainer._chunks(specs, workers=3)

    assert [len(chunk) for chunk in chunks] == [4, 3, 3]
    assert sorted(spec.index for chunk in chunks for spec in chunk) == list(range(10))


def test_payload_conversion_restores_rollout_tensors():
    model = trainer.v038.PredictiveRecurrentActorCritic(
        input_dim=trainer.v038.MOTION_GEOMETRY_INPUT_DIM,
        hidden_dim=16,
    )
    payload = collect_rollout_batch(
        {name: value.detach().cpu() for name, value in model.state_dict().items()},
        16,
        [_spec(0, 180.0)],
    )[0]
    rollout = trainer._payload_to_rollout(payload, torch.device("cpu"))

    assert rollout.observations.ndim == 2
    assert rollout.observations.shape[1] == trainer.v038.MOTION_GEOMETRY_INPUT_DIM
    assert rollout.latents.shape == (rollout.observations.shape[0], 2)
    assert rollout.old_log_probs.shape[0] == rollout.observations.shape[0]
