from __future__ import annotations

import pytest

from dmdod.evaluation.motor_latency_v04 import analyze_latency_rows, pair_fifo_events


def _event(index: int, due: float, decision: float, axis: str) -> dict:
    return {
        "event_index": index,
        "episode_s": due,
        "floor_indices": [index + 1],
        "chord_size": 1,
        "decisions": {"free": {"decision_s": decision, "axes": [axis]}},
    }


def test_fifo_complete_hit_latency_is_decomposed_without_nearest_matching():
    events = [_event(0, 1.0, .96, "left_1"),
              _event(1, 1.03, .99, "right_1")]
    keydowns = [{"episode_s": 1.005, "axis": "left_1"},
                {"episode_s": 1.04, "axis": "right_1"}]
    rows = pair_fifo_events(events, keydowns, policy="free", stats={
        "hits": 2, "misses": 0, "too_early": 0,
        "overloaded": False, "physical_keydowns": 2,
    })
    assert [row["command_to_keydown_ms"] for row in rows] == pytest.approx([45, 50])
    assert [row["scheduled_lead_ms"] for row in rows] == pytest.approx([40, 40])
    assert [row["timing_error_ms"] for row in rows] == pytest.approx([5, 10])


def test_fifo_pairing_refuses_incomplete_or_ambiguous_trace():
    with pytest.raises(ValueError, match="complete all-Hit"):
        pair_fifo_events([_event(0, 1.0, .96, "left_1")], [], policy="free",
                         stats={"hits": 0, "misses": 1, "too_early": 0,
                                "overloaded": False, "physical_keydowns": 0})


def test_latency_analysis_reports_density_reuse_and_quantiles():
    rows = [
        {"axis": "left_1", "bpm": 480.0, "command_to_keydown_ms": 40.0,
         "timing_error_ms": 0.0, "scheduled_lead_ms": 40.0,
         "axis_matches": True, "axis_reuse_ms": None},
        {"axis": "left_1", "bpm": 1920.0, "command_to_keydown_ms": 60.0,
         "timing_error_ms": 20.0, "scheduled_lead_ms": 40.0,
         "axis_matches": True, "axis_reuse_ms": 100.0},
        {"axis": "right_1", "bpm": 1920.0, "command_to_keydown_ms": 70.0,
         "timing_error_ms": 30.0, "scheduled_lead_ms": 40.0,
         "axis_matches": True, "axis_reuse_ms": 200.0},
    ]
    result = analyze_latency_rows(rows)
    assert result["overall"]["count"] == 3
    assert result["overall"]["latency_ms"]["median"] == 60.0
    assert result["by_bpm"]["1920"]["latency_ms"]["mean"] == 65.0
    assert result["reuse_le_125ms"]["count"] == 1
