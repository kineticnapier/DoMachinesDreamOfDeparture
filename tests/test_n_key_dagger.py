from __future__ import annotations

from pathlib import Path
import sys

import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_real_chart_v161_n_key_dagger as dagger_trainer
from dmdod.adofai_chart import parse_adofai_text
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import compile_adofai
from dmdod.n_key_policy import NKeyRecurrentActorCritic
from dmdod.n_key_training import (
    collect_n_key_dagger_sequence,
    discretize_n_key_action,
)


def _segment():
    chart = parse_adofai_text(
        """
        {
          "angleData": [0, 90, 180, 270, 0, 90, 180],
          "settings": {
            "bpm": 120,
            "pitch": 100,
            "countdownTicks": 0,
            "separateCountdownTime": false
          },
          "actions": []
        }
        """
    )
    compiled = compile_adofai(chart)
    return build_playable_segment(compiled, start_s=0.0, end_s=compiled.duration_s)


def test_discretize_n_key_action_uses_separate_press_release_thresholds() -> None:
    action = discretize_n_key_action(
        (0.24, 0.25, 0.80, -0.44, -0.45, 0.0, -0.90, 0.10),
        press_threshold=0.25,
        release_threshold=-0.45,
    )

    assert action.values == (0.0, 1.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0)


def test_n_key_dagger_collects_teacher_labels_on_student_state_trajectory() -> None:
    torch.manual_seed(7)
    model = NKeyRecurrentActorCritic(input_dim=263, key_count=8, hidden_dim=16)

    rollout = collect_n_key_dagger_sequence(
        model,
        _segment(),
        lead_s=0.044,
        press_threshold=0.25,
        release_threshold=-0.45,
        control_dt_s=0.010,
        physics_dt_s=0.001,
        device=torch.device("cpu"),
        source="unit-dagger",
    )

    assert rollout.sequence.frames > 0
    assert rollout.sequence.observations.shape == (rollout.sequence.frames, 263)
    assert rollout.sequence.teacher_actions.shape == (rollout.sequence.frames, 8)
    assert rollout.sequence.key_count == 8
    assert rollout.sequence.source == "unit-dagger"
    assert rollout.physical_keydowns >= 0


def test_v161_checkpoint_records_one_dagger_round_without_finalizing() -> None:
    model = NKeyRecurrentActorCritic(input_dim=263, key_count=8, hidden_dim=16)
    parent = {
        "format_version": 17,
        "trainer_version": "1.6.0-n-key-bootstrap",
        "key_count": 8,
        "input_dim": 263,
        "hidden_dim": 16,
        "final_used_for_selection": False,
        "finalized": False,
    }

    payload = dagger_trainer._dagger_checkpoint_payload(
        parent,
        model=model,
        source_checkpoint=Path("bootstrap.pt"),
        output_checkpoint=Path("dagger1.pt"),
        round_index=1,
        dagger_epoch=2,
        dagger_epochs=4,
        press_threshold=0.25,
        release_threshold=-0.45,
        lr=3e-4,
        expert_frames=100,
        dagger_frames=80,
        losses=[0.8, 0.6],
    )

    assert dagger_trainer.TRAINER_VERSION == "1.6.1-n-key-dagger"
    assert dagger_trainer.CHECKPOINT_FORMAT_VERSION == 18
    assert payload["dagger_round"] == 1
    assert payload["completed_dagger_epoch"] == 2
    assert payload["dagger_press_threshold"] == 0.25
    assert payload["dagger_release_threshold"] == -0.45
    assert payload["dagger_expert_frames"] == 100
    assert payload["dagger_student_state_frames"] == 80
    assert payload["final_used_for_selection"] is False
    assert payload["finalized"] is False
    assert "model_state" in payload
