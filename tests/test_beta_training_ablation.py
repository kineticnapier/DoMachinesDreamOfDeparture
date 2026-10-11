from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dmdod.adofai_chart import parse_adofai_text
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import compile_adofai
from dmdod.training.beta_training_ablation import (
    ARMS, _assert_beta_reference, preflight_outputs,
)
from dmdod.training.config import load_training_config
from dmdod.training.dagger_continuation import (
    collect_n_key_intervention_dagger_sequence_with_continuation,
)


class _AlwaysPressPolicy(torch.nn.Module):
    key_count = 4

    def initial_state(self, device):
        return torch.zeros(1, device=device)

    def forward_step(self, x, state):
        mean = torch.full((4,), 6.0, dtype=x.dtype, device=x.device)
        return mean, None, None, state


def _segment():
    chart = parse_adofai_text(
        '{"angleData":[0,90,180,270,0,90,180,270,0,90,180,270,0,90,180,270],'
        '"settings":{"bpm":120,"pitch":100,"countdownTicks":0,'
        '"separateCountdownTime":false},"actions":[]}'
    )
    compiled = compile_adofai(chart)
    return build_playable_segment(compiled, start_s=0.0, end_s=compiled.duration_s)


def _collect(limit):
    return collect_n_key_intervention_dagger_sequence_with_continuation(
        _AlwaysPressPolicy(), _segment(), lead_s=0.044,
        beta=0.0, seed=2401, press_threshold=0.25,
        release_threshold=-0.45,
        control_dt_s=0.010, physics_dt_s=0.001,
        device=torch.device("cpu"),
        action_mode="continuous", continue_after_failure=True,
        max_post_failure_s=limit,
    )


def test_post_failure_cap_retains_exact_first_terminal_and_prefix():
    full = _collect(None)
    capped = _collect(1.0)
    assert full.stats.overloaded
    assert capped.stats.overloaded
    assert full.stats.hits == capped.stats.hits
    assert full.stats.too_early_presses == capped.stats.too_early_presses
    assert full.physical_keydowns == capped.physical_keydowns
    assert full.terminal_frame_count == capped.terminal_frame_count
    assert full.terminal_control_time_s == capped.terminal_control_time_s
    assert capped.sequence.frames == capped.terminal_frame_count + 100
    assert full.sequence.frames >= capped.sequence.frames
    assert torch.equal(
        capped.sequence.observations,
        full.sequence.observations[:capped.sequence.frames],
    )
    assert torch.equal(
        capped.sequence.teacher_actions,
        full.sequence.teacher_actions[:capped.sequence.frames],
    )
    assert torch.equal(
        capped.executed_actions,
        full.executed_actions[:capped.sequence.frames],
    )


def test_zero_post_failure_cap_disables_continuation():
    capped = _collect(0.0)
    assert capped.stats.overloaded
    assert capped.sequence.frames == capped.terminal_frame_count


@pytest.mark.parametrize("limit", [-1, float("nan"), float("inf")])
def test_invalid_post_failure_cap_rejected(limit):
    with pytest.raises(ValueError, match="max_post_failure_s"):
        _collect(limit)


def test_arm_definition_and_beta1_reproducibility_guard():
    assert [x.name for x in ARMS] == [
        "beta1_full", "beta05_full", "beta0_full", "beta0_post1s",
    ]
    assert [x.beta for x in ARMS] == [1.0, 0.5, 0.0, 0.0]
    assert ARMS[-1].max_post_failure_s == 1.0
    expected = (9, 1546, 250, 1796)
    actual = dict(zip(("safe", "hits", "early", "keydowns"), expected))
    _assert_beta_reference(actual, expected)
    with pytest.raises(RuntimeError, match="reference candidate mismatch"):
        _assert_beta_reference({**actual, "hits": 1545}, expected)


def test_preflight_never_overwrites_existing_outputs(tmp_path):
    report = tmp_path / "beta.json"
    source = tmp_path / "source.pt"
    paths = preflight_outputs(report, save_candidates=True, source_checkpoint=source)
    assert len(paths) == 4
    assert all(not path.exists() for path in paths.values())
    report.write_text("{}")
    with pytest.raises(FileExistsError):
        preflight_outputs(report, save_candidates=False, source_checkpoint=source)
    report.unlink()
    next(iter(paths.values())).write_text("occupied")
    with pytest.raises(FileExistsError):
        preflight_outputs(report, save_candidates=True, source_checkpoint=source)


def test_ablation_config_has_fixed_20_validation_anchors_and_no_alpha_sweep():
    cfg = load_training_config(
        "configs/training/human_visible_beta_training_ablation.toml"
    )
    assert cfg.data.anchor_limit == 20
    assert cfg.data.validation_limit == 20
    assert cfg.human_visible.updates_per_epoch == 2
    assert not cfg.human_visible.alpha_sweep_enabled
    assert "UNUSED_beta_training_ablation" in cfg.run.output
