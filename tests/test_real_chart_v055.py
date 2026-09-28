from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v055 as trainer  # noqa: E402
from dmdod.recurrent_policy import RecurrentActorCritic  # noqa: E402


def _sequence(frames: int, source: str) -> trainer.DAggerSequence:
    observations = torch.zeros(
        (frames, trainer.REAL_CHART_INPUT_DIM),
        dtype=torch.float32,
    )
    actions = torch.zeros((frames, 2), dtype=torch.float32)
    if frames >= 2:
        actions[1, 0] = 1.0
    return trainer.DAggerSequence(observations, actions, source)


def test_dagger_sequence_validates_shapes_and_counts_frames():
    sequence = _sequence(5, "student")
    assert sequence.frames == 5
    assert trainer._dataset_frames([sequence, _sequence(3, "expert")]) == 8

    with pytest.raises(ValueError):
        trainer.DAggerSequence(
            torch.zeros((2, trainer.REAL_CHART_INPUT_DIM - 1)),
            torch.zeros((2, 2)),
            "bad-observation",
        )
    with pytest.raises(ValueError):
        trainer.DAggerSequence(
            torch.zeros((2, trainer.REAL_CHART_INPUT_DIM)),
            torch.zeros((3, 2)),
            "bad-action",
        )


def test_aggregate_bc_resets_recurrent_state_at_each_trajectory_boundary():
    class CountingPolicy(RecurrentActorCritic):
        def __init__(self) -> None:
            super().__init__(input_dim=trainer.REAL_CHART_INPUT_DIM, hidden_dim=8)
            self.initial_state_calls = 0

        def initial_state(self, device: torch.device) -> torch.Tensor:
            self.initial_state_calls += 1
            return super().initial_state(device)

    model = CountingPolicy()
    sequences = [_sequence(4, "expert"), _sequence(3, "student-round-1")]
    trainer._train_aggregate_bc(
        model,
        sequences,
        epochs=1,
        learning_rate=1e-4,
        chunk_steps=2,
    )
    assert model.initial_state_calls == len(sequences)


def test_v054_bootstrap_checkpoint_loads_only_matching_model_shape(tmp_path: Path):
    source = RecurrentActorCritic(
        input_dim=trainer.REAL_CHART_INPUT_DIM,
        hidden_dim=8,
    )
    path = tmp_path / "bootstrap.pt"
    torch.save(
        {
            "format_version": trainer.v054.CHECKPOINT_FORMAT_VERSION,
            "input_dim": trainer.REAL_CHART_INPUT_DIM,
            "hidden_dim": 8,
            "model_state": source.state_dict(),
        },
        path,
    )

    target = RecurrentActorCritic(
        input_dim=trainer.REAL_CHART_INPUT_DIM,
        hidden_dim=8,
    )
    payload = trainer._load_checkpoint_model(
        target,
        path,
        hidden_dim=8,
        expected_format=trainer.v054.CHECKPOINT_FORMAT_VERSION,
        device=torch.device("cpu"),
    )
    assert payload["format_version"] == trainer.v054.CHECKPOINT_FORMAT_VERSION
    for key, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[key])
