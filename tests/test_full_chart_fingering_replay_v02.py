from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_oracle_fingering_v02 as teacher
import run_full_chart_fingering_replay_v02 as full

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.training import fingering_network_v01 as network


def _segment():
    chart = compile_adofai(parse_adofai_text(json.dumps({
        "angleData": [0, 0, 0, 0],
        "settings": {"bpm": 120, "pitch": 100, "countdownTicks": 0},
        "actions": [],
    })))
    return build_playable_segment(chart)


def test_network_execution_never_calls_teacher_physical_ranker(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Teacher rollout leaked into Network inference")
    monkeypatch.setattr(teacher, "rank_first_choices", forbidden)
    model = network.FingeringRanker()
    model.eval()
    result = full.run_network(_segment(), model, stop_after_s=.08)
    assert result["teacher_shadow_rollout_used"] is False
    assert result["teacher_candidate_scores_used"] is False
    assert result["failure_continuation"] is False
    assert result["trace"]


def test_choice_summary_counts_thumb_side_and_fast_disjoint_switch():
    rows = [
        {"event_index": 1, "due_ms": 10, "chosen": "left_5/tip", "axes": ["left_5"]},
        {"event_index": 2, "due_ms": 30, "chosen": "right_6/edge", "axes": ["right_6"]},
    ]
    result = full.summarize_choices(rows)
    assert result["thumb_uses"] == 1 and result["side_uses"] == 1
    assert result["extreme_switches_le_35ms"][0]["gap_ms"] == 20


def test_trace_contains_seek_reconstructable_cumulative_state():
    model = network.FingeringRanker()
    model.eval()
    result = full.run_network(_segment(), model, stop_after_s=.03)
    assert all(len(frame) == 7 for frame in result["trace"])
    assert all(set(frame[6]) == {"hits", "misses", "too_early", "overloaded"}
               for frame in result["trace"])
