from __future__ import annotations

from dataclasses import fields
import pytest

from dmdod.features.hold_mesh_observability_v01 import (
    DiagnosticCurveModel,
    DiagnosticHoldMeshPixelRenderer,
    HoldDurationObservation,
    HoldMeshPixelDurationSensor,
    PI_STOCK,
    RenderedHoldMeshFrame,
    StockHoldMeshBuilder,
    VisibleMeshScene,
    mesh_buffer_duration_candidates,
)


def _metadata(duration: int, **kwargs):
    return StockHoldMeshBuilder.build(
        entry_angle_rad=0.0,
        exit_angle_rad=PI_STOCK,
        is_ccw=False,
        num_planets=2,
        midspin=False,
        duration=duration,
        **kwargs,
    )


def _templates(model, scene=VisibleMeshScene(), *, occlusion=0.0):
    return tuple((duration, DiagnosticHoldMeshPixelRenderer.render(
        rendered_at_s=1.0,
        metadata=_metadata(duration),
        scene=scene,
        model=model,
        occlude_right_fraction=occlusion,
    )) for duration in range(4))


def test_stock_cpu_mesh_adds_exactly_100_segments_per_duration_loop() -> None:
    values = [_metadata(duration) for duration in range(4)]
    assert [item.longitudinal_segments for item in values] == [50, 150, 250, 350]
    assert [item.vertex_count for item in values] == [102, 302, 502, 702]
    assert [item.triangle_index_count for item in values] == [300, 900, 1500, 2100]


def test_mesh_buffer_can_identify_duration_but_is_not_student_pixels() -> None:
    metadata = tuple((duration, _metadata(duration)) for duration in range(4))
    assert mesh_buffer_duration_candidates(
        observed_vertex_count=metadata[2][1].vertex_count,
        base_metadata_by_duration=metadata,
    ) == (2,)


def test_midspin_has_no_stock_hold_mesh_and_multiplanet_loop_is_explicit() -> None:
    assert StockHoldMeshBuilder.build(
        entry_angle_rad=0.0, exit_angle_rad=0.0, is_ccw=False,
        num_planets=2, midspin=True, duration=3,
    ) is None
    metadata = StockHoldMeshBuilder.build(
        entry_angle_rad=0.0, exit_angle_rad=PI_STOCK / 3.0,
        is_ccw=False, num_planets=3, midspin=False, duration=0,
    )
    assert metadata.added_multiplanet_loops in (0, 1)


def test_opaque_overlapping_pixel_curve_cannot_identify_positive_duration() -> None:
    templates = _templates(DiagnosticCurveModel.OPAQUE_CONSTANT_RADIUS)
    frame = dict(templates)[3]
    observation = HoldMeshPixelDurationSensor().observe(
        observed_at_s=1.0, frame=frame, candidate_templates=templates
    )
    assert observation.valid
    assert 3 in observation.candidates
    assert len(observation.candidates) > 1
    assert observation.confidence is None


def test_explicit_radial_drift_proxy_identifies_duration_from_pixels_only() -> None:
    scene = VisibleMeshScene(zoom=1.4, camera_rotation_rad=0.73)
    templates = _templates(DiagnosticCurveModel.PROXY_RADIAL_DRIFT, scene)
    frame = dict(templates)[2]
    observation = HoldMeshPixelDurationSensor().observe(
        observed_at_s=1.01, frame=frame, candidate_templates=templates
    )
    assert observation.valid
    assert observation.candidates == (2,)
    assert observation.confidence == 1.0


def test_occlusion_missing_and_stale_frames_abstain() -> None:
    clean = _templates(DiagnosticCurveModel.PROXY_RADIAL_DRIFT)
    occluded = dict(_templates(
        DiagnosticCurveModel.PROXY_RADIAL_DRIFT, occlusion=0.35
    ))[2]
    sensor = HoldMeshPixelDurationSensor(max_frame_age_s=0.02)
    mismatch = sensor.observe(
        observed_at_s=1.0, frame=occluded, candidate_templates=clean
    )
    assert not mismatch.valid
    assert mismatch.reason == "no_exact_pixel_template"
    stale = sensor.observe(
        observed_at_s=1.1, frame=dict(clean)[1], candidate_templates=clean
    )
    assert not stale.valid
    assert stale.reason == "missing_or_stale_frame"
    missing = RenderedHoldMeshFrame(2.0, 32, 32, frozenset(), False)
    missing_result = sensor.observe(
        observed_at_s=2.0, frame=missing, candidate_templates=()
    )
    assert not missing_result.valid


def test_student_duration_observation_has_no_hidden_mesh_truth() -> None:
    public = {item.name for item in fields(HoldDurationObservation)}
    assert public.isdisjoint({
        "hold_length", "holdLength", "duration", "angle_length",
        "vertex_count", "target_time_s", "release_time_s",
    })


def test_future_frame_and_invalid_geometry_are_rejected() -> None:
    frame = dict(_templates(DiagnosticCurveModel.PROXY_RADIAL_DRIFT))[0]
    with pytest.raises(ValueError, match="future"):
        HoldMeshPixelDurationSensor().observe(
            observed_at_s=0.9, frame=frame, candidate_templates=()
        )
    with pytest.raises(ValueError):
        VisibleMeshScene(zoom=0.0)
