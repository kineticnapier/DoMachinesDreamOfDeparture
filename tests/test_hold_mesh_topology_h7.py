from __future__ import annotations

from math import pi
import pytest

from dmdod.evaluation.hold_mesh_topology_h7 import (
    STOCK_HOLD_RENDERER_SHA256,
    reconstruct_stock_mesh_topology,
    mesh_counters_are_policy_visible,
)


def _mesh(duration: int, **overrides):
    args = dict(entry_angle_rad=0.0, exit_angle_rad=pi / 2,
                is_ccw=False, hold_length=duration)
    args.update(overrides)
    return reconstruct_stock_mesh_topology(**args)


def test_renderer_snapshot_fingerprint_is_frozen():
    assert STOCK_HOLD_RENDERER_SHA256 == (
        "b6c19c740a21aa04a8191de952990345b0034042a9f9f413284f2ce5ced8ad80"
    )


@pytest.mark.parametrize("duration", (0, 1, 2, 3))
def test_quarter_turn_mesh_adds_100_segments_and_200_vertices_per_loop(duration):
    result = _mesh(duration)
    assert result is not None
    assert result.segment_count == 25 + 100 * duration
    assert result.vertex_count == 52 + 200 * duration
    assert result.triangle_count == 50 + 200 * duration
    assert result.uv0_last_y == pytest.approx(0.25 + duration)
    assert result.uv1_last_x == 51 + 200 * duration


def test_mesh_differs_for_same_long_icon_durations_1_and_3():
    one, three = _mesh(1), _mesh(3)
    assert three.segment_count - one.segment_count == 200
    assert three.vertex_count - one.vertex_count == 400
    assert three.triangle_count - one.triangle_count == 400
    assert not mesh_counters_are_policy_visible()


def test_zero_or_full_turn_counts_as_one_turn_in_stock_mesh():
    zero = reconstruct_stock_mesh_topology(entry_angle_rad=0.0, exit_angle_rad=0.0,
                                           is_ccw=False, hold_length=0)
    assert zero.segment_count == 100


def test_midspin_does_not_create_mesh_and_special_extra_loop_is_explicit():
    assert _mesh(2, midspin=True) is None
    a, b = _mesh(1), _mesh(1, added_loops=1)
    assert b.segment_count - a.segment_count == 100


@pytest.mark.parametrize("value", (-1, 1.5, True))
def test_invalid_duration_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        _mesh(value)


def test_no_projection_or_release_time_is_returned():
    result = _mesh(1)
    assert not hasattr(result, "pixels")
    assert not hasattr(result, "release_time_s")
    assert not hasattr(result, "visible_loops")
    assert not hasattr(result, "duration_observed_from_pixels")
