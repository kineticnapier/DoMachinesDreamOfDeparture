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
import run_dmdogo_v05_comparison as v05

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.training import fingering_network_v01 as network


def _segment(count: int = 8, bpm: float = 480):
    return build_playable_segment(compile_adofai(parse_adofai_text(json.dumps({
        "angleData": [0] * count,
        "settings": {"bpm": bpm, "pitch": 100, "countdownTicks": 0},
        "actions": [],
    }))))


def test_compensation_disabled_matches_v02_selector_history():
    model = network.FingeringRanker()
    model.eval()
    segment = _segment()
    baseline = v05.run_continuity_arm(
        segment, model, compensation=False, stop_after_s=1.0, wall_timeout_s=10)
    reference = v05.run_v02_reference(
        segment, model, stop_after_s=1.0, wall_timeout_s=10)
    assert [row["chosen"] for row in baseline["decisions"]] == [
        row["chosen"] for row in reference["decisions"]]
    assert baseline["keydowns"] == reference["keydowns"]


def test_compensated_continuity_is_causal_live_eligible_and_reset(monkeypatch):
    monkeypatch.setattr(teacher, "rank_first_choices",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("Teacher rollout leaked")))
    model = network.FingeringRanker()
    model.eval()
    left = v05.run_continuity_arm(
        _segment(), model, compensation=True, stop_after_s=1.0, wall_timeout_s=10)
    right = v05.run_continuity_arm(
        _segment(), model, compensation=True, stop_after_s=1.0, wall_timeout_s=10)
    assert [row["chosen"] for row in left["decisions"]] == [
        row["chosen"] for row in right["decisions"]]
    assert left["compensation_telemetry"]
    assert all(row["eligible"][row["selected_candidate_index"]]
               for row in left["decisions"] if row["selected_candidate_index"] is not None)
    assert all(len(row["candidate_labels"]) == len(row["eligible"])
               for row in left["decisions"])
    assert all(row["selection_reason"] in {
        "scale_free_network_continuity_rank", "no_eligible_candidate"}
        for row in left["decisions"])
    assert left["teacher_shadow_rollout_used"] is False


def test_v05_output_is_create_once_and_keeps_v03_overlay(tmp_path, monkeypatch):
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
    monkeypatch.setattr(v05.baseline, "load_model", lambda _path: model)
    output = tmp_path / "v05"
    report = v05.run(chart, checkpoint, output, stop_after_s=.2, wall_timeout_s=10)
    assert tuple(report["policies"]) == v05.POLICY_ORDER
    html = (output / "replay-v05.html").read_text(encoding="utf-8")
    assert "16 Contact KeyDown Counter" in html
    assert "Continuity + Compensation v0.5" in html
    assert "Motor compensation ON" in html
    with pytest.raises(FileExistsError):
        v05.run(chart, checkpoint, output, stop_after_s=.2, wall_timeout_s=10)


def test_existing_v04_default_remains_free():
    model = network.FingeringRanker()
    model.eval()
    result = v04.run_research_arm(
        _segment(), model, policy_name="latency_only", planning=False,
        compensation=True, stop_after_s=1.0, wall_timeout_s=10)
    assert all(row["selection_reason"] == "free_network_argmin"
               for row in result["decisions"])
