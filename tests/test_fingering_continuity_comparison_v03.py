from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_fingering_continuity_comparison_v03 as comparison
import run_full_chart_fingering_replay_v02 as runner

from dmdod.training import fingering_network_v01 as network


def _result(policy: str, axis: str, error_ms: float) -> dict:
    return {
        "stats": {"targets": 1, "hits": 1, "misses": 0, "too_early": 0,
                  "overloaded": False, "overload_counter": 0,
                  "x_accuracy_percent": 100.0, "timing_mae_ms": abs(error_ms),
                  "signed_timing_error_ms": error_ms, "physical_keydowns": 1},
        "decisions": [{"event_index": 0, "ordinals": [0], "due_ms": 500.0,
                       "decision_ms": 460.0, "chosen": axis,
                       "axes": [axis], "outcome": "Hit", "judgement": "Hit",
                       "actual_error_ms": error_ms, "different_from_free": policy != "free",
                       "style_class": "ordinary_proxy", "style_alpha": 0.0,
                       "selection_reason": policy, "selector_mode": policy,
                       "fast_network_guard_applied": policy == "continuity_v03"}],
        "keydowns": [{"episode_s": .5 + error_ms * .001,
                      "time_ms": 500 + error_ms, "axis": axis, "contact": axis}],
        "trace": [[0.0, [0.0] * 12, [0.0] * 12, [False] * 12,
                   ["tip", "tip"], [False, False],
                   {"hits": 0, "misses": 0, "too_early": 0, "overloaded": False}]],
        "replay_end_s": .6, "termination_reason": "diagnostic_prefix_complete",
        "failure_continuation": False, "teacher_shadow_rollout_used": False,
        "teacher_candidate_scores_used": False, "policy": policy,
        "style_alpha": 0.0, "selector_mode": policy,
    }


def test_policy_summary_includes_signed_error_headroom_and_guard_count():
    result = _result("continuity_v03", "left_2", 7.0)
    summary = comparison.policy_summary(result)
    assert summary["signed_timing_error_ms"] == 7.0
    assert summary["mean_arrival_headroom_ms"] == -7.0
    assert summary["fast_network_guard_count"] == 1
    assert summary["continuity"]["decision_count"] == 1


def test_three_way_run_keeps_independent_arms_and_v03_viewer(tmp_path, monkeypatch):
    chart_path = tmp_path / "small.adofai"
    chart_path.write_text(json.dumps({
        "angleData": [0, 0],
        "settings": {"bpm": 120, "pitch": 100, "countdownTicks": 0},
        "actions": [],
    }), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"unchanged checkpoint")
    model = network.FingeringRanker()
    model.eval()
    monkeypatch.setattr(runner, "load_model", lambda _path: model)
    calls = []

    def fake_run(_segment, _model, **kwargs):
        mode = kwargs["selector_mode"]
        calls.append(mode)
        axes = {"free": "left_1", "continuity": "left_2",
                "continuity_v03": "left_3"}
        return _result(mode, axes[mode], 3.0)

    monkeypatch.setattr(runner, "run_network", fake_run)
    output = tmp_path / "three-way"
    report = comparison.run(chart_path, checkpoint, output,
                            stop_after_s=.6, wall_timeout_s=10)
    assert calls == ["free", "continuity", "continuity_v03"]
    assert set(report["policies"]) == {"free", "continuity", "continuity_v03"}
    assert report["training_performed"] is False
    assert (output / "comparison-v03.json").is_file()
    html = (output / "replay-v03.html").read_text(encoding="utf-8")
    assert "16 Contact KeyDown Counter" in html
    assert "Continuity v0.3" in html

    try:
        comparison.run(chart_path, checkpoint, output,
                       stop_after_s=.6, wall_timeout_s=10)
    except FileExistsError:
        pass
    else:
        raise AssertionError("v0.3 output was overwritten")
