from __future__ import annotations

import pytest
import torch

from dmdod.adofai_chart import parse_adofai_text
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import compile_adofai
from dmdod.n_key_policy import NKeyRecurrentActorCritic
from dmdod.n_key_real_chart import (
    DiagnosticHudNKeyRealChartMotorEnv,
    encode_n_key_hud_real_chart_observation,
    n_key_hud_real_chart_input_dim,
)
from dmdod.n_key_training import (
    NKeyBCSequence,
    collect_n_key_expert_sequence,
    n_key_actuation_loss,
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


@pytest.mark.parametrize(
    ("key_count", "expected"),
    [(4, 251), (6, 257), (8, 263)],
)
def test_n_key_hud_input_dimension_scales_only_motor_slice(
    key_count: int,
    expected: int,
) -> None:
    assert n_key_hud_real_chart_input_dim(key_count) == expected


def test_eight_key_hud_environment_encodes_263d() -> None:
    env = DiagnosticHudNKeyRealChartMotorEnv(_segment(), key_count=8)
    observation = env.reset()
    encoded = encode_n_key_hud_real_chart_observation(observation)

    assert len(encoded) == 263
    assert observation.motor.key_names == (
        "left_4",
        "left_3",
        "left_2",
        "left_1",
        "right_1",
        "right_2",
        "right_3",
        "right_4",
    )


def test_n_key_expert_collection_produces_eight_action_labels() -> None:
    rollout = collect_n_key_expert_sequence(
        _segment(),
        key_count=8,
        lead_s=0.044,
        control_dt_s=0.010,
        physics_dt_s=0.001,
        device=torch.device("cpu"),
    )

    assert rollout.sequence.frames > 0
    assert rollout.sequence.observations.shape[1] == 263
    assert rollout.sequence.teacher_actions.shape == (rollout.sequence.frames, 8)
    assert rollout.sequence.key_count == 8
    assert rollout.physical_keydowns >= 0


def test_n_key_sequence_rejects_wrong_action_width() -> None:
    with pytest.raises(ValueError, match=r"\[T, 8\]"):
        NKeyBCSequence(
            observations=torch.zeros((3, 263)),
            teacher_actions=torch.zeros((3, 4)),
            key_count=8,
            source="bad",
        )


def test_n_key_actuation_loss_supports_eight_outputs() -> None:
    target = torch.zeros((4, 8), dtype=torch.float32)
    target[0, 3] = 1.0
    target[1, 4] = 1.0
    target[2, 3] = -1.0
    target[3, 4] = -1.0
    good = target.clone()
    bad = torch.zeros_like(target)

    assert n_key_actuation_loss(good, target) < n_key_actuation_loss(bad, target)


def test_n_key_actuation_loss_accepts_equivalent_free_finger_press() -> None:
    target = torch.zeros((1, 8), dtype=torch.float32)
    target[0, 0] = -1.0  # physically held key: release identity is fixed
    target[0, 3] = 1.0   # teacher routed the next target to this free key

    canonical = target.clone()
    equivalent = torch.zeros_like(target)
    equivalent[0, 0] = -1.0
    equivalent[0, 6] = 1.0  # another free finger can claim the same target

    assert n_key_actuation_loss(canonical, target) == pytest.approx(0.0, abs=1e-12)
    assert n_key_actuation_loss(equivalent, target) == pytest.approx(0.0, abs=1e-12)


def test_n_key_actuation_loss_penalizes_extra_press() -> None:
    target = torch.zeros((1, 8), dtype=torch.float32)
    target[0, 3] = 1.0

    exact_count = torch.zeros_like(target)
    exact_count[0, 6] = 1.0
    extra_press = exact_count.clone()
    extra_press[0, 5] = 1.0

    assert n_key_actuation_loss(exact_count, target) == pytest.approx(0.0, abs=1e-12)
    assert n_key_actuation_loss(extra_press, target) > 0.0


def test_n_key_actuation_loss_keeps_release_identity_specific() -> None:
    target = torch.zeros((1, 8), dtype=torch.float32)
    target[0, 0] = -1.0
    target[0, 3] = 1.0

    correct = target.clone()
    wrong_release = torch.zeros_like(target)
    wrong_release[0, 1] = -1.0
    wrong_release[0, 3] = 1.0

    assert n_key_actuation_loss(correct, target) == pytest.approx(0.0, abs=1e-12)
    assert n_key_actuation_loss(wrong_release, target) > 0.0


def test_n_key_policy_maps_263d_to_eight_actions_without_vf_gru_path() -> None:
    model = NKeyRecurrentActorCritic(input_dim=263, key_count=8, hidden_dim=16)
    state = model.initial_state(torch.device("cpu"))
    observations = torch.zeros((5, 263), dtype=torch.float32)

    means, values, final_state = model.forward_sequence(observations, state)
    action, next_state = model.deterministic_action(observations[0], state)

    assert means.shape == (5, 8)
    assert values.shape == (5,)
    assert final_state.shape == (16,)
    assert len(action.values) == 8
    assert next_state.shape == (16,)
