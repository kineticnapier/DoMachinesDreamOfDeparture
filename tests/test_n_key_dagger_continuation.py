from __future__ import annotations

from pathlib import Path
import sys

import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_real_chart_v166_n_key_failure_continuation_dagger as trainer
from dmdod.adofai_chart import parse_adofai_text
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import compile_adofai
from dmdod.n_key_dagger_continuation import (
    collect_n_key_dagger_sequence_with_continuation,
)


class _AlwaysPressPolicy(torch.nn.Module):
    def __init__(self, key_count: int) -> None:
        super().__init__()
        self.key_count = key_count

    def initial_state(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.float32, device=device)

    def forward_step(self, x: torch.Tensor, state: torch.Tensor):
        mean = torch.full(
            (self.key_count,),
            6.0,
            dtype=x.dtype,
            device=x.device,
        )
        std = torch.ones_like(mean)
        value = torch.zeros((), dtype=x.dtype, device=x.device)
        return mean, std, value, state


def _segment():
    chart = parse_adofai_text(
        """
        {
          "angleData": [0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270],
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


def _collect(*, continue_after_failure: bool):
    return collect_n_key_dagger_sequence_with_continuation(
        _AlwaysPressPolicy(8),
        _segment(),
        lead_s=0.044,
        press_threshold=0.25,
        release_threshold=-0.45,
        control_dt_s=0.010,
        physics_dt_s=0.001,
        device=torch.device("cpu"),
        action_mode="continuous",
        continue_after_failure=continue_after_failure,
    )


def test_failure_continuation_preserves_terminal_score_and_extends_labels() -> None:
    ordinary = _collect(continue_after_failure=False)
    continued = _collect(continue_after_failure=True)

    assert ordinary.stats.overloaded is True
    assert continued.stats.overloaded is True
    assert continued.stats.hits == ordinary.stats.hits
    assert continued.stats.too_early_presses == ordinary.stats.too_early_presses
    assert continued.physical_keydowns == ordinary.physical_keydowns
    assert continued.sequence.frames > ordinary.sequence.frames + 50
    assert continued.sequence.teacher_actions.shape == (
        continued.sequence.frames,
        8,
    )


def test_v166_wrapper_records_failure_continuation_identity() -> None:
    assert trainer.TRAINER_VERSION == "1.6.6-n-key-failure-continuation-dagger"
    assert trainer.CHECKPOINT_FORMAT_VERSION == 24
