from __future__ import annotations

import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v070 as v070  # noqa: E402
import train_real_chart_v070_flat as flat  # noqa: E402
from dmdod.adofai_chart import parse_adofai_text  # noqa: E402
from dmdod.adofai_playable import build_playable_segment  # noqa: E402
from dmdod.adofai_timing import compile_adofai  # noqa: E402
from dmdod.flat_hud_eval import evaluate_hud_state_on_segment  # noqa: E402
from dmdod.parallel_hud_eval import evaluate_hud_state_on_segments  # noqa: E402
from dmdod.real_chart_hud_features import HUD_REAL_CHART_INPUT_DIM  # noqa: E402
from dmdod.recurrent_policy import RecurrentActorCritic  # noqa: E402


def _tiny_segment():
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


def test_flat_eval_defaults_to_all_5600g_logical_threads(monkeypatch):
    monkeypatch.delenv("DMDOD_EVAL_WORKERS", raising=False)
    monkeypatch.setattr(flat.os, "cpu_count", lambda: 12)
    assert flat._configured_flat_eval_workers() == 12

    monkeypatch.setattr(flat.os, "cpu_count", lambda: 32)
    assert flat._configured_flat_eval_workers() == 12


def test_flat_eval_worker_override_can_use_more_than_six(monkeypatch):
    monkeypatch.setenv("DMDOD_EVAL_WORKERS", "10")
    assert flat._configured_flat_eval_workers() == 10


def test_flat_evaluator_matches_existing_batched_evaluator_for_one_segment():
    torch.manual_seed(123)
    model = RecurrentActorCritic(
        input_dim=HUD_REAL_CHART_INPUT_DIM,
        hidden_dim=8,
        initial_log_std=-1.20,
    )
    state = model.state_dict()
    segment = _tiny_segment()

    old = evaluate_hud_state_on_segments(state, 8, [segment], True, 0.010)[0]
    new = evaluate_hud_state_on_segment(state, 8, segment, True, 0.010)

    old_stats, old_keydowns = old
    new_stats, new_keydowns = new
    assert new_keydowns == old_keydowns
    assert new_stats.hits == old_stats.hits
    assert new_stats.misses == old_stats.misses
    assert new_stats.too_early_presses == old_stats.too_early_presses
    assert new_stats.overloaded == old_stats.overloaded
    assert new_stats.x_accuracy_percent == old_stats.x_accuracy_percent


def test_flat_trainer_keeps_v070_checkpoint_format_for_resume():
    assert v070.CHECKPOINT_FORMAT_VERSION == 14
    assert flat.FLAT_PARALLEL_EVAL_VERSION == "alpha-segment-flat-pool-v2"
