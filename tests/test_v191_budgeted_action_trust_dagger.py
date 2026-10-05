from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v191_n_key_connectome_budgeted_action_trust_dagger as v191


def test_format_duration() -> None:
    assert v191._format_duration(0) == "00:00:00"
    assert v191._format_duration(3661.9) == "01:01:01"


def test_budget_payload_records_wall_clock_state() -> None:
    class _FakeModel:
        def checkpoint_metadata(self):
            return {"n_key_policy_backend": "fly_connectome"}

        def named_parameters(self):
            yield "actor_mean.weight", nn.Parameter(torch.zeros(1), requires_grad=True)
            yield "actor_mean.bias", nn.Parameter(torch.zeros(1), requires_grad=True)

    payload = v191._budget_payload(
        {},
        model=_FakeModel(),
        model_state={"actor_mean.weight": torch.zeros(1)},
        source_checkpoint=Path("source.pt"),
        output_checkpoint=Path("output.pt"),
        round_index=4,
        requested_hours=8.0,
        elapsed_seconds=123.0,
        trial_count=7,
        accepted_steps=3,
        selected_step=2,
        stopped_reason="time-budget",
        actor_steps=8,
        lr=3e-4,
        stay_coef=20.0,
        max_action_rms=0.01,
        current_action_rms=0.0025,
        expert_frames=100,
        dagger_frames=120,
        student_frame_history=[120, 121],
        history=[{"trial": 1}],
    )

    assert payload["format_version"] == 32
    assert payload["trainer_version"] == v191.TRAINER_VERSION
    assert payload["budget_requested_hours"] == 8.0
    assert payload["budget_elapsed_seconds"] == 123.0
    assert payload["budget_trial_count"] == 7
    assert payload["budget_accepted_steps"] == 3
    assert payload["budget_selected_step"] == 2
    assert payload["budget_stopped_reason"] == "time-budget"
    assert payload["budget_initial_action_rms"] == 0.01
    assert payload["budget_current_action_rms"] == 0.0025
    assert payload["final_used_for_selection"] is False
