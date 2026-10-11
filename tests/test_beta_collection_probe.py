from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dmdod.adofai_chart import parse_adofai_text
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import compile_adofai
from dmdod.training.beta_collection_probe import (
    BETAS, format_beta_summary, run_beta_collection_probe, summarize_rollout,
)
from dmdod.training.config import load_training_config
from dmdod.training.dagger_continuation import (
    collect_n_key_intervention_dagger_sequence_with_continuation,
)
from dmdod.training.real_chart import NamedSegment, evaluate_role_continuous


class _AlwaysPressPolicy(torch.nn.Module):
    def __init__(self, key_count=4):
        super().__init__()
        self.key_count = key_count
        self.weight = torch.nn.Parameter(torch.tensor([6.0]))

    def initial_state(self, device):
        return torch.zeros(1, device=device)

    def forward_step(self, x, state):
        mean = self.weight.detach().repeat(self.key_count).to(x.device)
        return mean, None, None, state


def _segment():
    chart = parse_adofai_text(
        '{"angleData":[0,90,180,270,0,90,180,270,0,90,180,270,0,90,180,270],'
        '"settings":{"bpm":120,"pitch":100,"countdownTicks":0,'
        '"separateCountdownTime":false},"actions":[]}'
    )
    compiled = compile_adofai(chart)
    return build_playable_segment(compiled, start_s=0.0, end_s=compiled.duration_s)


def _collect(beta):
    model = _AlwaysPressPolicy()
    return collect_n_key_intervention_dagger_sequence_with_continuation(
        model, _segment(), lead_s=0.044,
        beta=beta, seed=7, press_threshold=0.25, release_threshold=-0.45,
        control_dt_s=0.010, physics_dt_s=0.001,
        device=torch.device("cpu"), action_mode="continuous",
        continue_after_failure=True,
    )


def test_first_terminal_boundary_is_separate_from_failure_continuation():
    failed = _collect(0.0)
    assert failed.stats.overloaded
    assert failed.terminal_frame_count > 0
    assert failed.sequence.frames > failed.terminal_frame_count
    assert failed.terminal_control_time_s is not None
    assert failed.terminal_control_time_s > 0
    row = summarize_rollout(failed, beta=0.0, anchor_index=1, chart_name="tiny")
    assert row.overloaded
    assert row.pre_terminal_frames == failed.terminal_frame_count
    assert row.post_terminal_frames > 0
    assert row.interventions == 0
    assert row.action_elements_pre == row.pre_terminal_frames * 4
    assert row.action_elements_post == row.post_terminal_frames * 4
    assert row.pressed_key_frames_pre >= 0
    assert "post reflects diagnostic continuation" in "\n".join(format_beta_summary((row,)))


def test_beta_one_uses_teacher_every_frame_and_beta_zero_uses_student():
    one = _collect(1.0)
    zero = _collect(0.0)
    assert bool(one.interventions.all())
    assert not bool(zero.interventions.any())
    assert torch.equal(one.executed_actions, one.sequence.teacher_actions)
    assert torch.equal(zero.executed_actions, zero.student_actions)
    assert one.terminal_frame_count > 0
    assert zero.terminal_frame_count > 0
    assert one.sequence.frames >= one.terminal_frame_count


def test_beta_zero_first_terminal_matches_normal_student_evaluation():
    model = _AlwaysPressPolicy()
    named = NamedSegment("train", "tiny", "sha", 0.0, _segment().duration_s, _segment())
    result = evaluate_role_continuous(
        model, [named], label="check",
        control_dt_s=0.01, physics_dt_s=0.001,
        device=torch.device("cpu"), verbose=False,
    )[0]
    zero = collect_n_key_intervention_dagger_sequence_with_continuation(
        model, named.segment, lead_s=0.044, beta=0.0, seed=7,
        press_threshold=0.25, release_threshold=-0.45,
        control_dt_s=0.01, physics_dt_s=0.001,
        device=torch.device("cpu"), action_mode="continuous",
        continue_after_failure=True,
    )
    assert zero.stats.hits == result[0].hits
    assert zero.stats.too_early_presses == result[0].too_early_presses
    assert zero.stats.overloaded == result[0].overloaded
    assert zero.physical_keydowns == result[1]


def test_probe_reuses_anchors_and_seeds_without_modifying_model(capsys):
    seg = _segment()
    named = NamedSegment("train", "tiny", "sha", 0.0, seg.duration_s, seg)
    prepared = SimpleNamespace(
        anchors=[named], lead_s=0.044, key_count=4, control_dt_s=0.01,
        physics_dt_s=0.001, device=torch.device("cpu"),
    )
    model = _AlwaysPressPolicy()
    initial = model.weight.detach().clone()
    calls = []

    def collector(model, segment, **kwargs):
        calls.append((kwargs["beta"], kwargs["seed"], kwargs["continue_after_failure"]))
        return collect_n_key_intervention_dagger_sequence_with_continuation(
            model, segment, **kwargs,
        )

    rows = run_beta_collection_probe(prepared, model, seed=2401, collect=collector)
    assert tuple(r.beta for r in rows) == BETAS
    assert len({seed for _, seed, _ in calls}) == 1
    assert all(cont for _, _, cont in calls)
    assert torch.equal(model.weight, initial)
    assert "not-overloaded=" in capsys.readouterr().out


def test_beta_probe_config_is_read_only(tmp_path):
    cfg = load_training_config("configs/training/human_visible_beta_collection_probe.toml")
    assert cfg.mode == "human_visible_curriculum_dagger"
    assert cfg.data.anchor_limit == 20
    assert cfg.data.validation_limit == 1
    assert cfg.run.output.endswith("UNUSED_beta_collection_probe_DO_NOT_WRITE.pt")
    assert not cfg.human_visible.alpha_sweep_enabled
    assert BETAS == (1.0, 0.5, 0.0)
