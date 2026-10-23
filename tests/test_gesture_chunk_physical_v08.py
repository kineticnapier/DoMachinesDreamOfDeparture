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


def chart_text():
    return json.dumps({
        "angleData": [0] * 20,
        "settings": {"bpm": 1920, "pitch": 100, "countdownTicks": 0},
        "actions": [],
    })


def _segment():
    return build_playable_segment(
        compile_adofai(parse_adofai_text(chart_text())))


def test_v08_physical_still_checks_live_motor_without_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Teacher candidate ranking called by v0.8")
    monkeypatch.setattr(runner.teacher, "rank_first_choices", forbidden)
    result = runner.run_pattern_arm(
        _segment(), planning_mode="gesture_v08", minimum_reuse_s=.080,
        stop_after_s=5., wall_timeout_s=30.)
    assert result["planning_mode"] == "gesture_v08"
    assert result["policy"] == "gesture_chunk_v08"
    assert result["abstract_schedule"]["feasible"] is True
    assert result["abstract_schedule"]["model_contract"]["human_likeness_verified"] is False
    assert result["network_used"] is False
    assert result["teacher_candidate_scores_used"] is False
    assert result["decisions"]
    assert all(row["physical_eligible"] for row in result["decisions"])
    assert all(row["gesture_chunk"] is not None for row in result["decisions"])
    assert result["physical"]["local_replans"] == sum(
        row["local_replan"] for row in result["decisions"])


def test_v08_replay_format_and_default_v06_unchanged(tmp_path):
    chart = tmp_path / "mini.adofai"
    chart.write_text(chart_text(), encoding="utf-8")
    out = tmp_path / "v08"
    result = runner.run(
        chart, out, planning_mode="gesture_v08",
        stop_after_s=5., wall_timeout_s=30.)
    assert result["format"] == "dmdogo-gesture-chunk-physical-v0.8"
    payload = json.loads((out / "replay-v08.json").read_text(encoding="utf-8"))
    assert payload["format"] == result["format"]
    assert payload["default_policy"] == "gesture_chunk_v08"
    assert "gesture_chunk_v08" in payload["policies"]
    assert (out / "replay-v08.html").is_file()
    assert (out / "telemetry-v08.json").is_file()
    assert not (out / "replay-v07.json").exists()
    assert not (out / "replay-v06.json").exists()
