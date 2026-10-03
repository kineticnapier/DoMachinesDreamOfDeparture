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


def test_n_key_actuation_loss_macro_balances_rare_key_press() -> None:
    # key 0 has one press while key 1 presses on every frame.  Missing the rare
    # key must remain a first-class error rather than being diluted by the 100
    # common-key press labels.
    target = torch.zeros((100, 2), dtype=torch.float32)
    target[0, 0] = 1.0
    target[:, 1] = 1.0
    predicted = target.clone()
    predicted[0, 0] = 0.0

    loss = n_key_actuation_loss(predicted, target)

    # press margin miss on key 0 is 0.7^2.  Per-key macro averaging contributes
    # half of that before PRESS_MARGIN_COEF=6, i.e. about 1.47 by itself.
    assert loss > 1.4


def test_n_key_actuation_loss_macro_balances_neutral_keys() -> None:
    # Inner-like key 0 has only two neutral frames, outer-like key 1 is neutral
    # almost everywhere.  Unsafe pushes on key 0 should not disappear in the
    # large global neutral denominator.
    target = torch.ones((100, 2), dtype=torch.float32)
    target[:2, 0] = 0.0
    target[:, 1] = 0.0
    predicted = target.clone()
    predicted[:2, 0] = 0.30

    loss = n_key_actuation_loss(predicted, target)

    # neutral violation on key 0 is (0.30 - 0.05)^2 = 0.0625; macro averaging
    # across two neutral-bearing keys then NEUTRAL_PUSH_COEF=12 gives 0.375,
    # plus the small MSE term.
    assert loss > 0.37


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
