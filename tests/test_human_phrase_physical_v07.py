from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_dmdogo_v06_pattern_first as runner
from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai


def _segment():
    return build_playable_segment(compile_adofai(parse_adofai_text(json.dumps({
        "angleData": [0] * 20,
        "settings": {"bpm": 1920, "pitch": 100, "countdownTicks": 0},
        "actions": [],
    }))))


def test_v07_physical_replay_uses_distinct_untrained_phrase_planner(monkeypatch):
    def leak(*args, **kwargs):
        raise AssertionError("Teacher candidate ranking must not be used")
    monkeypatch.setattr(runner.teacher, "rank_first_choices", leak)
    result = runner.run_pattern_arm(
        _segment(), planning_mode="phrase_v07", minimum_reuse_s=.080,
        stop_after_s=5.0, wall_timeout_s=30.0)
    assert result["planning_mode"] == "phrase_v07"
    assert result["abstract_schedule"]["feasible"]
    assert result["abstract_schedule"]["model_contract"]["human_demonstrations_used"] is False
    assert result["abstract_schedule"]["model_contract"]["motor_feasibility_verified"] is False
    assert result["network_used"] is False
    assert result["teacher_candidate_scores_used"] is False
    assert result["decisions"]
    assert all(row["gesture_template"] for row in result["decisions"])
    assert all(row["rhythm_class"] for row in result["decisions"])
    assert all(row["physical_eligible"] for row in result["decisions"])


def test_v07_and_default_v06_replay_output_names_do_not_collide(tmp_path):
    chart = tmp_path / "mini.adofai"
    chart.write_text(json.dumps({
        "angleData": [0] * 20,
        "settings": {"bpm": 1920, "pitch": 100, "countdownTicks": 0},
        "actions": [],
    }), encoding="utf-8")
    out = tmp_path / "phrase"
    report = runner.run(
        chart, out, planning_mode="phrase_v07",
        stop_after_s=5.0, wall_timeout_s=30.0)
    assert report["format"] == "dmdogo-human-phrase-physical-v0.7"
    assert report["planning_mode"] == "phrase_v07"
    payload = json.loads((out / "replay-v07.json").read_text(encoding="utf-8"))
    assert payload["format"] == "dmdogo-human-phrase-physical-v0.7"
    assert payload["default_policy"] == "human_phrase_v07"
    assert (out / "replay-v07.html").exists()
    assert (out / "telemetry-v07.json").exists()
    assert not (out / "replay-v06.json").exists()
