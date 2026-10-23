from __future__ import annotations

from dmdod.evaluation.fingering_continuity_v03 import (
    associate_event_keydowns,
    common_fast_metrics,
    summarize_error_rows,
)


def _event(index: int, time_s: float, axis: str = "left_1") -> dict:
    return {
        "event_index": index,
        "episode_s": time_s,
        "chord_size": 1,
        "decisions": {"free": {"axes": [axis], "timing_error_ms": 0.0}},
    }


def test_complete_fifo_trace_has_unique_ordered_correspondence():
    events = [_event(0, 1.0), _event(1, 1.02)]
    keydowns = [
        {"episode_s": 1.01, "axis": "left_1"},
        {"episode_s": 1.03, "axis": "left_1"},
    ]
    result = associate_event_keydowns(
        events, keydowns, policy="free",
        stats={"hits": 2, "misses": 0, "too_early": 0,
               "physical_keydowns": 2, "overloaded": False},
    )
    assert result["method"] == "repository_fifo_complete_hit_order"
    assert result["unique_count"] == 2
    assert [row["keydown_index"] for row in result["rows"]] == [0, 1]


def test_complete_fifo_order_remains_unique_when_intended_axis_differs():
    result = associate_event_keydowns(
        [_event(0, 1.0, "left_1")],
        [{"episode_s": 1.01, "axis": "right_1"}],
        policy="free",
        stats={"hits": 1, "misses": 0, "too_early": 0,
               "physical_keydowns": 1, "overloaded": False},
    )
    assert result["unique_count"] == 1
    assert result["axis_mismatch_count"] == 1
    assert result["rows"][0]["axis_matches"] is False


def test_nearest_window_fallback_preserves_ambiguity():
    events = [_event(0, 1.0)]
    keydowns = [
        {"episode_s": .99, "axis": "left_1"},
        {"episode_s": 1.01, "axis": "left_1"},
    ]
    result = associate_event_keydowns(events, keydowns, policy="free", stats=None)
    assert result["method"] == "axis_time_window_candidates"
    assert result["ambiguous_count"] == 1
    assert result["rows"][0]["status"] == "ambiguous"
    assert result["rows"][0]["candidate_keydown_indices"] == [0, 1]


def test_error_summary_keeps_signed_and_absolute_error_separate():
    summary = summarize_error_rows([
        {"error_ms": -10.0, "bpm": 480.0},
        {"error_ms": 20.0, "bpm": 480.0},
    ], pitch_ratio=1.0)
    assert summary["count"] == 2
    assert summary["signed_error_ms"] == 5.0
    assert summary["timing_mae_ms"] == 15.0
    assert 0.0 <= summary["x_accuracy_percent"] <= 100.0


def test_common_fast_metrics_use_the_same_event_population_for_all_policies():
    event_times = {0: 1.0, 1: 1.02, 2: 1.04, 3: 1.06}
    free = [
        {"event_index": i, "due_ms": t * 1000, "axes": [f"left_{i % 4 + 1}"]}
        for i, t in event_times.items()
    ]
    continuity = [
        {"event_index": i, "due_ms": t * 1000,
         "axes": [f"{'left' if i % 2 == 0 else 'right'}_{i % 4 + 1}"]}
        for i, t in event_times.items()
    ]
    result = common_fast_metrics(
        {"free": free, "continuity": continuity}, event_times, threshold_ms=35.0)
    assert result["event_indices"] == [0, 1, 2, 3]
    assert result["policies"]["free"]["decision_count"] == 4
    assert result["policies"]["continuity"]["decision_count"] == 4
    assert result["policies"]["free"]["fixed_window_comparisons"] == 1
    assert result["policies"]["continuity"]["fixed_window_comparisons"] == 1
    assert result["policies"]["free"]["fixed_window_stability"] == 1.0
    assert result["policies"]["continuity"]["fixed_window_stability"] == 0.0


def test_common_fast_metrics_intersect_prefixes_instead_of_reading_future_rows():
    event_times = {0: 1.0, 1: 1.02, 2: 1.04, 3: 1.06, 4: 1.08}
    free = [{"event_index": i, "due_ms": event_times[i] * 1000,
             "axes": [f"left_{i % 4 + 1}"]} for i in range(4)]
    shorter = free[:3]
    result = common_fast_metrics({"free": free, "short": shorter}, event_times)
    assert result["event_indices"] == [0, 1, 2]
    assert result["fixed_windows"] == []
