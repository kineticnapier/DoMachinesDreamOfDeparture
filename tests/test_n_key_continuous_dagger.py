from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_real_chart_v162_n_key_continuous_dagger as trainer
from dmdod.adofai_chart import parse_adofai_text
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import compile_adofai
from dmdod.n_key_policy import NKeyRecurrentActorCritic
from dmdod.n_key_training import collect_n_key_dagger_sequence


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


def test_n_key_dagger_collects_on_continuous_student_trajectory() -> None:
    torch.manual_seed(13)
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
        source="unit-continuous-dagger",
        action_mode="continuous",
    )

    assert rollout.sequence.frames > 0
    assert rollout.sequence.observations.shape == (rollout.sequence.frames, 263)
    assert rollout.sequence.teacher_actions.shape == (rollout.sequence.frames, 8)
    assert rollout.sequence.source == "unit-continuous-dagger"
    assert rollout.physical_keydowns >= 0


def test_n_key_dagger_rejects_unknown_action_mode() -> None:
    model = NKeyRecurrentActorCritic(input_dim=263, key_count=8, hidden_dim=16)

    with pytest.raises(ValueError, match="action_mode"):
        collect_n_key_dagger_sequence(
            model,
            _segment(),
            lead_s=0.044,
            press_threshold=0.25,
            release_threshold=-0.45,
            control_dt_s=0.010,
            physics_dt_s=0.001,
            device=torch.device("cpu"),
            action_mode="banana",
        )


def test_v162_checkpoint_records_continuous_round_without_thresholds() -> None:
    state = {"weight": torch.tensor([1.0])}
    parent = {
        "format_version": 18,
        "trainer_version": "1.6.1-n-key-dagger",
        "dagger_round": 1,
        "final_used_for_selection": False,
        "finalized": False,
    }

    payload = trainer._checkpoint_payload(
        parent,
        model_state=state,
        source_checkpoint=Path("dagger1.pt"),
        output_checkpoint=Path("dagger2.pt"),
        round_index=2,
        completed_epoch=4,
        requested_epochs=4,
        selected_epoch=3,
        lr=3e-4,
        expert_frames=100,
        dagger_frames=90,
        losses=[0.8, 0.6, 0.5, 0.4],
        selection_history=[{"epoch": 0}, {"epoch": 3}],
    )

    assert trainer.TRAINER_VERSION == "1.6.2-n-key-continuous-dagger"
    assert trainer.CHECKPOINT_FORMAT_VERSION == 19
    assert payload["dagger_round"] == 2
    assert payload["dagger_action_mode"] == "continuous"
    assert payload["dagger_press_threshold"] is None
    assert payload["dagger_release_threshold"] is None
    assert payload["dagger_selected_epoch"] == 3
    assert payload["dagger_selection_uses_validation"] is False
    assert payload["final_used_for_selection"] is False
    assert payload["finalized"] is False
