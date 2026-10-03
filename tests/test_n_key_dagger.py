from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

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


def _stats(
    *,
    hits: int,
    targets: int = 100,
    xacc: float = 50.0,
    early: int = 0,
    overloaded: bool = False,
):
    return SimpleNamespace(
        hits=hits,
        targets=targets,
        x_accuracy_percent=xacc,
        x_accuracy_points=xacc * targets / 100.0,
        x_accuracy_denominator=float(targets),
        too_early_presses=early,
        overloaded=overloaded,
    )


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


def test_train_safety_guard_rejects_safe_to_overload() -> None:
    references = [(_stats(hits=80, overloaded=False), 80)]
    candidates = [(_stats(hits=90, overloaded=True), 90)]

    accepted, reasons = dagger_trainer._train_safety_guard(references, candidates)

    assert accepted is False
    assert any("safe->overload" in reason for reason in reasons)


def test_train_safety_guard_rejects_large_anchor_hit_regression() -> None:
    references = [(_stats(hits=80, targets=100), 80)]
    candidates = [(_stats(hits=0, targets=100), 0)]

    accepted, reasons = dagger_trainer._train_safety_guard(references, candidates)

    assert accepted is False
    assert any("hits" in reason for reason in reasons)


def test_train_selection_prioritizes_aggregate_hits_before_xacc() -> None:
    more_hits = [(_stats(hits=91, xacc=10.0), 91)]
    less_hits_better_xacc = [(_stats(hits=90, xacc=99.0), 90)]

    assert dagger_trainer._selection_key(more_hits) > dagger_trainer._selection_key(
        less_hits_better_xacc
    )


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
        dagger_epoch=4,
        dagger_epochs=4,
        press_threshold=0.25,
        release_threshold=-0.45,
        lr=3e-4,
        expert_frames=100,
        dagger_frames=80,
        losses=[0.8, 0.6, 0.5, 0.4],
        selected_epoch=2,
        selection_history=[{"epoch": 0}, {"epoch": 2}],
    )

    assert dagger_trainer.TRAINER_VERSION == "1.6.1-n-key-dagger"
    assert dagger_trainer.CHECKPOINT_FORMAT_VERSION == 18
    assert payload["dagger_round"] == 1
    assert payload["completed_dagger_epoch"] == 4
    assert payload["dagger_selected_epoch"] == 2
    assert payload["dagger_press_threshold"] == 0.25
    assert payload["dagger_release_threshold"] == -0.45
    assert payload["dagger_expert_frames"] == 100
    assert payload["dagger_student_state_frames"] == 80
    assert payload["dagger_selection_uses_validation"] is False
    assert payload["dagger_train_selection_history"] == [{"epoch": 0}, {"epoch": 2}]
    assert payload["final_used_for_selection"] is False
    assert payload["finalized"] is False
    assert "model_state" in payload
