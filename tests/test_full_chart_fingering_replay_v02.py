from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
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


def test_wall_timeout_is_enforced():
    model = network.FingeringRanker()
    model.eval()
    try:
        full.run_network(_segment(), model, wall_timeout_s=-1)
    except TimeoutError as error:
        assert "wall timeout" in str(error)
    else:
        raise AssertionError("wall timeout was ignored")


def test_run_accepts_two_distinct_chart_files_and_records_actual_hashes(tmp_path, monkeypatch):
    """DMDOGO accepts ordinary .adofai inputs rather than one fixed chart SHA."""
    model = network.FingeringRanker()
    model.eval()
    monkeypatch.setattr(full, "load_model", lambda _path: model)
    checkpoint = tmp_path / "unchanged-checkpoint.pt"
    checkpoint.write_bytes(b"stub checksum; model load is mocked")
    checksums = []

    for index, (angles, bpm) in enumerate((
        ([0, 0, 0, 0], 120),
        ([0, 90, 0, 0], 180),
    )):
        chart_path = tmp_path / f"custom-{index}.adofai"
        chart_path.write_text(json.dumps({
            "angleData": angles,
            "settings": {"bpm": bpm, "pitch": 100, "countdownTicks": 0},
            "actions": [],
        }), encoding="utf-8")
        output_dir = tmp_path / f"replay-{index}"
        summary = full.run(chart_path, "unrated", checkpoint, output_dir,
                           stop_after_s=.08, wall_timeout_s=10)
        payload = json.loads((output_dir / "full-chart-replay.json").read_text(encoding="utf-8"))
        disk_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

        assert summary == disk_summary
        assert summary["chart"] == payload["chart"] == chart_path.name
        assert summary["chart_sha256"] == payload["chart_sha256"] == full.viewer.file_sha256(chart_path)
        assert summary["difficulty"] == payload["difficulty"] == "unrated"
        assert summary["timing_source"] == "AutoPlay Oracle exact event times"
        assert summary["training_performed"] is False
        assert summary["teacher_shadow_rollout_used"] is False
        assert (output_dir / "full-chart-replay.html").is_file()
        assert all("has no Pause" not in item for item in payload["rendering_limitations"])
        checksums.append(summary["chart_sha256"])

    assert checksums[0] != checksums[1]


def test_run_rejects_non_adofai_before_creating_output(tmp_path):
    wrong_chart = tmp_path / "input.txt"
    wrong_chart.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "should-not-exist"
    with pytest.raises(ValueError, match=r"\.adofai"):
        full.run(wrong_chart, "unrated", tmp_path / "unused.pt", output_dir)
    assert not output_dir.exists()
