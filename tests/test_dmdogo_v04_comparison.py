from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_oracle_fingering_v02 as teacher
import run_dmdogo_v04_comparison as v04
import run_full_chart_fingering_replay_v02 as baseline

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.training import fingering_network_v01 as network


def _segment(count=8, bpm=480):
    chart = compile_adofai(parse_adofai_text(json.dumps({
        "angleData": [0] * count,
        "settings": {"bpm": bpm, "pitch": 100, "countdownTicks": 0},
        "actions": [],
    })))
    return build_playable_segment(chart)


def test_latency_and_integrated_arms_are_causal_and_live_eligible(monkeypatch):
    monkeypatch.setattr(teacher, "rank_first_choices",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("Teacher rollout leaked")))
    model = network.FingeringRanker()
    model.eval()
    for name, planning in (("latency_only", False), ("integrated_v04", True)):
        result = v04.run_research_arm(
            _segment(), model, policy_name=name, planning=planning,
            compensation=True, stop_after_s=1.0, wall_timeout_s=10)
        assert result["teacher_shadow_rollout_used"] is False
        assert result["compensation_telemetry"]
        for row in result["decisions"]:
            selected = row["selected_candidate_index"]
            if selected is not None:
                assert row["eligible"][selected]


def test_planning_is_deterministic_and_reset_per_episode():
    model = network.FingeringRanker()
    model.eval()
    left = v04.run_research_arm(
        _segment(), model, policy_name="planning_only", planning=True,
        compensation=False, stop_after_s=1.0, wall_timeout_s=10)
    right = v04.run_research_arm(
        _segment(), model, policy_name="planning_only", planning=True,
        compensation=False, stop_after_s=1.0, wall_timeout_s=10)
    assert [row["chosen"] for row in left["decisions"]] == [
        row["chosen"] for row in right["decisions"]]
    assert left["planner"]["peak_beam"] <= 8


def test_six_arm_output_is_create_once_and_viewer_keeps_key_overlay(tmp_path, monkeypatch):
    chart = tmp_path / "small.adofai"
    chart.write_text(json.dumps({
        "angleData": [0, 0],
        "settings": {"bpm": 120, "pitch": 100, "countdownTicks": 0},
        "actions": [],
    }), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen")
    model = network.FingeringRanker()
    model.eval()
    monkeypatch.setattr(baseline, "load_model", lambda _path: model)
    output = tmp_path / "v04"
    report = v04.run(chart, checkpoint, output, stop_after_s=.2, wall_timeout_s=10)
    assert tuple(report["policies"]) == v04.POLICY_ORDER
    html = (output / "replay-v04.html").read_text(encoding="utf-8")
    assert "16 Contact KeyDown Counter" in html
    assert "Integrated v0.4" in html
    with pytest.raises(FileExistsError):
        v04.run(chart, checkpoint, output, stop_after_s=.2, wall_timeout_s=10)
