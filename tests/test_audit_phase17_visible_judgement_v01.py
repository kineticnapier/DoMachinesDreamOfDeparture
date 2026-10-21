from __future__ import annotations

import pytest

from scripts.audit_phase17_visible_judgement_v01 import (
    next_capture_s,
    observe_frames,
    render_counters,
)


def test_renderer_discards_target_identity_before_observer_boundary():
    trace = [
        {"time_s": 3.261, "target_index": 37, "margin": "very_late", "key": "right_2"},
        {"time_s": 3.264, "target_index": 999, "margin": "perfect", "key": "left_1"},
    ]
    labels = ("perfect", "very_late")
    frames = render_counters(trace, labels, [3.260, 3.270])
    rows = observe_frames(frames, labels)

    assert frames[-1].counters == (("perfect", 1), ("very_late", 1))
    assert set(frames[-1].__dataclass_fields__) == {
        "observed_at_s", "displayed_at_s", "counters",
    }
    assert rows[-1]["deltas"] == {"perfect": 1, "very_late": 1}
    assert rows[-1]["within_interval_order_known"] is False
    assert "target_index" not in rows[-1]


def test_ideal_capture_is_first_grid_point_not_before_visible_update():
    assert next_capture_s(3.261, 0.010) == pytest.approx(3.27)
    assert next_capture_s(3.260, 0.010) == pytest.approx(3.26)


def test_same_delta_vector_cannot_identify_event_correspondence():
    labels = ("perfect", "very_late")
    backlog = render_counters(
        [{"time_s": 1.001, "target_index": 37, "margin": "very_late"}],
        labels,
        [1.000, 1.010],
    )
    correctly_matched_but_late = render_counters(
        [{"time_s": 1.001, "target_index": 38, "margin": "very_late"}],
        labels,
        [1.000, 1.010],
    )

    assert backlog == correctly_matched_but_late
    assert observe_frames(backlog, labels) == observe_frames(correctly_matched_but_late, labels)
