from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_fingering_continuity_comparison_v02 as comparison
import run_full_chart_fingering_replay_v02 as runner

from dmdod.training import fingering_network_v01 as network


def _result(policy: str, chosen: str, outcome: str = "Hit") -> dict:
    return {
        "stats": {"targets": 1, "hits": int(outcome == "Hit"),
                  "misses": int(outcome != "Hit"), "too_early": 0,
                  "overloaded": False, "overload_counter": 0,
                  "x_accuracy_percent": 100.0, "timing_mae_ms": 1.0,
                  "physical_keydowns": 1},
        "decisions": [{"event_index": 0, "ordinals": [0], "due_ms": 500.0,
                       "decision_ms": 460.0, "chosen": chosen,
                       "axes": [chosen.split("/")[0]], "outcome": outcome,
                       "judgement": outcome, "actual_error_ms": 1.0,
                       "different_from_free": policy == "continuity",
                       "style_class": "ordinary_proxy", "style_alpha": 0.0,
                       "selection_reason": policy, "selector_mode": policy}],
        "keydowns": [{"episode_s": .501, "time_ms": 501.0,
                      "axis": chosen.split("/")[0], "contact": "left_1"}],
        "trace": [[0.0, [0.0] * 12, [0.0] * 12, [False] * 12,
                   ["tip", "tip"], [False, False],
                   {"hits": 0, "misses": 0, "too_early": 0, "overloaded": False}]],
        "replay_end_s": .6, "termination_reason": "diagnostic_prefix_complete",
        "failure_continuation": False, "teacher_shadow_rollout_used": False,
        "teacher_candidate_scores_used": False, "policy": policy,
        "style_alpha": 0.0, "selector_mode": policy,
    }


def test_compare_results_reports_changed_and_lost_baseline_hits():
    free = _result("free", "left_1/tip", "Hit")
    continuity = _result("continuity", "left_2/tip", "Miss")
    result = comparison.compare_results(free, continuity)
    assert result["different_event_count"] == 1
    assert result["lost_baseline_hit_events"] == [0]
    assert result["recovered_hit_events"] == []
    assert result["hit_non_regression"] is False
    assert result["free_continuity"]["decision_count"] == 1
    assert result["continuity_continuity"]["changed_from_free_count"] == 1


def test_ab_run_uses_independent_selector_modes_and_keeps_v03_overlay(
        tmp_path, monkeypatch):
    chart_path = tmp_path / "small.adofai"
    chart_path.write_text(json.dumps({
        "angleData": [0, 0],
        "settings": {"bpm": 120, "pitch": 100, "countdownTicks": 0},
        "actions": [],
    }), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"read-only checkpoint stub")
    model = network.FingeringRanker()
    model.eval()
    monkeypatch.setattr(runner, "load_model", lambda _path: model)
    calls = []

    def fake_run(_segment, _model, **kwargs):
        calls.append(kwargs)
        mode = kwargs["selector_mode"]
        return _result(mode, "left_1/tip" if mode == "free" else "left_2/tip")

    monkeypatch.setattr(runner, "run_network", fake_run)
    output_dir = tmp_path / "comparison"
    report = comparison.run(chart_path, checkpoint, output_dir,
                            stop_after_s=.6, wall_timeout_s=10)

    assert [call["selector_mode"] for call in calls] == ["free", "continuity"]
    assert report["training_performed"] is False
    assert report["teacher_shadow_rollout_used"] is False
    assert (output_dir / "comparison.json").is_file()
    assert (output_dir / "replay-v02.html").is_file()
    html = (output_dir / "replay-v02.html").read_text(encoding="utf-8")
    assert "16 Contact KeyDown Counter" in html
    assert "Continuity" in html

    try:
        comparison.run(chart_path, checkpoint, output_dir,
                       stop_after_s=.6, wall_timeout_s=10)
    except FileExistsError:
        pass
    else:
        raise AssertionError("comparison output was overwritten")
