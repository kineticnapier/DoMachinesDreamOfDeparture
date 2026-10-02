from dataclasses import replace

import pytest
import torch

from dmdod.adofai_chart import parse_adofai_text
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import compile_adofai
from dmdod.four_key_motor import FourKeyAction, FourKeyMotorEnv
from dmdod.four_key_real_chart import FOUR_KEY_HUD_REAL_CHART_INPUT_DIM
from dmdod.four_key_training import (
    CenterFirstFourKeyTeacher,
    FourKeyDAggerSequence,
    collect_four_key_expert_sequence,
    four_key_actuation_loss,
)


def _idle_motor():
    return FourKeyMotorEnv().reset()


def _segment():
    chart = parse_adofai_text(
        """
        {
          "angleData": [0, 90, 180, 270, 0],
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


def _dense_segment():
    chart = parse_adofai_text(
        """
        {
          "angleData": [0, 90, 180, 270, 0, 90, 180, 270, 0],
          "settings": {
            "bpm": 1200,
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


def test_center_first_teacher_latches_key_until_physical_press() -> None:
    teacher = CenterFirstFourKeyTeacher()
    observation = _idle_motor()

    first = teacher.action(
        observation,
        now_s=0.0,
        target_time_s=0.04,
        lead_s=0.05,
        target_token=1,
    )
    second = teacher.action(
        observation,
        now_s=0.01,
        target_time_s=0.04,
        lead_s=0.05,
        target_token=1,
    )

    assert first.left_inner == 1.0
    assert first.right_inner == 0.0
    assert second.left_inner == 1.0
    assert second.right_inner == 0.0


def test_center_first_teacher_student_preference_cannot_skip_inner_tier() -> None:
    teacher = CenterFirstFourKeyTeacher()
    observation = _idle_motor()
    preferred = FourKeyAction(
        left_outer=1.0,
        left_inner=0.1,
        right_inner=0.8,
        right_outer=0.9,
    )

    action = teacher.action(
        observation,
        now_s=0.0,
        target_time_s=0.0,
        lead_s=0.05,
        target_token=1,
        preferred_action=preferred,
    )

    assert action.right_inner == 1.0
    assert action.left_outer == 0.0
    assert action.right_outer == 0.0


def test_center_first_teacher_uses_outer_only_when_both_inner_keys_are_held() -> None:
    teacher = CenterFirstFourKeyTeacher()
    observation = replace(
        _idle_motor(),
        left_inner_pressed=True,
        right_inner_pressed=True,
    )
    preferred = FourKeyAction(
        left_outer=0.2,
        left_inner=1.0,
        right_inner=1.0,
        right_outer=0.9,
    )

    action = teacher.action(
        observation,
        now_s=0.0,
        target_time_s=0.0,
        lead_s=0.05,
        target_token=1,
        preferred_action=preferred,
    )

    assert action.left_inner == -1.0
    assert action.right_inner == -1.0
    assert action.right_outer == 1.0


def test_new_target_releases_latch_and_advances_center_alternation() -> None:
    teacher = CenterFirstFourKeyTeacher()
    observation = _idle_motor()

    first = teacher.action(
        observation,
        now_s=0.0,
        target_time_s=0.0,
        lead_s=0.05,
        target_token=1,
    )
    second = teacher.action(
        observation,
        now_s=0.01,
        target_time_s=0.01,
        lead_s=0.05,
        target_token=2,
    )

    assert first.left_inner == 1.0
    assert second.right_inner == 1.0


def test_pipelined_teacher_launches_second_inner_before_first_keydown() -> None:
    teacher = CenterFirstFourKeyTeacher()
    observation = _idle_motor()

    action = teacher.pipeline_action(
        observation,
        now_s=0.0,
        targets=((1, 0.04), (2, 0.05)),
        lead_s=0.05,
    )

    assert action.left_inner == 1.0
    assert action.right_inner == 1.0
    assert action.left_outer == 0.0
    assert action.right_outer == 0.0


def test_pipelined_teacher_unlocks_outer_after_both_inners_are_reserved() -> None:
    teacher = CenterFirstFourKeyTeacher()
    observation = _idle_motor()

    action = teacher.pipeline_action(
        observation,
        now_s=0.0,
        targets=((1, 0.03), (2, 0.04), (3, 0.05)),
        lead_s=0.05,
    )

    assert action.left_inner == 1.0
    assert action.right_inner == 1.0
    assert action.left_outer == 1.0
    assert action.right_outer == 0.0


def test_pipelined_teacher_keeps_inflight_target_reserved_until_it_resolves() -> None:
    teacher = CenterFirstFourKeyTeacher()
    idle = _idle_motor()
    targets = ((1, 0.04), (2, 0.05))

    teacher.pipeline_action(
        idle,
        now_s=0.0,
        targets=targets,
        lead_s=0.05,
    )
    first_pressed = replace(idle, left_inner_pressed=True)
    action = teacher.pipeline_action(
        first_pressed,
        now_s=0.04,
        targets=targets,
        lead_s=0.05,
    )

    assert action.left_inner == -1.0
    assert action.right_inner == 1.0
    assert action.left_outer == 0.0
    assert action.right_outer == 0.0


def test_four_key_actuation_loss_accepts_only_four_output_batches() -> None:
    target = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    good = target.clone()
    bad = torch.zeros_like(target)

    assert four_key_actuation_loss(good, target) < four_key_actuation_loss(bad, target)

    with pytest.raises(ValueError):
        four_key_actuation_loss(torch.zeros((2, 2)), torch.zeros((2, 2)))


def test_four_key_dagger_sequence_enforces_251d_observation_and_4d_action() -> None:
    sequence = FourKeyDAggerSequence(
        observations=torch.zeros((3, FOUR_KEY_HUD_REAL_CHART_INPUT_DIM)),
        teacher_actions=torch.zeros((3, 4)),
        source="test",
    )
    assert sequence.frames == 3

    with pytest.raises(ValueError):
        FourKeyDAggerSequence(
            observations=torch.zeros((3, FOUR_KEY_HUD_REAL_CHART_INPUT_DIM - 1)),
            teacher_actions=torch.zeros((3, 4)),
            source="bad-observation",
        )

    with pytest.raises(ValueError):
        FourKeyDAggerSequence(
            observations=torch.zeros((3, FOUR_KEY_HUD_REAL_CHART_INPUT_DIM)),
            teacher_actions=torch.zeros((3, 2)),
            source="bad-action",
        )


def test_four_key_expert_collection_produces_251d_4d_training_sequence() -> None:
    rollout = collect_four_key_expert_sequence(
        _segment(),
        lead_s=0.05,
        control_dt_s=0.010,
        device=torch.device("cpu"),
    )

    assert rollout.sequence.frames > 0
    assert rollout.sequence.observations.shape[1] == FOUR_KEY_HUD_REAL_CHART_INPUT_DIM
    assert rollout.sequence.teacher_actions.shape == (rollout.sequence.frames, 4)
    assert rollout.teacher_fraction == 1.0
    assert rollout.physical_keydowns >= 0


def test_four_key_expert_collection_pipelines_dense_future_targets() -> None:
    rollout = collect_four_key_expert_sequence(
        _dense_segment(),
        lead_s=0.05,
        control_dt_s=0.010,
        device=torch.device("cpu"),
    )

    positive_per_frame = (rollout.sequence.teacher_actions > 0.25).sum(dim=1)
    assert int(positive_per_frame.max().item()) >= 2
