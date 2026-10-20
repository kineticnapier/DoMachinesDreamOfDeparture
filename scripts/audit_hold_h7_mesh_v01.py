from __future__ import annotations

"""Create the H7 Hold mesh counterfactual observability report."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from dmdod.features.hold_mesh_observability_v01 import (
    DiagnosticCurveModel,
    DiagnosticHoldMeshPixelRenderer,
    HoldMeshPixelDurationSensor,
    PI_STOCK,
    StockHoldMeshBuilder,
    VisibleMeshScene,
    mesh_buffer_duration_candidates,
)


FORMAT = "dmdod-hold-h7-mesh-observability-v0.1"


def _metadata(duration: int):
    return StockHoldMeshBuilder.build(
        entry_angle_rad=0.0,
        exit_angle_rad=PI_STOCK,
        is_ccw=False,
        num_planets=2,
        midspin=False,
        duration=duration,
    )


def _templates(model: DiagnosticCurveModel, scene: VisibleMeshScene):
    return tuple((duration, DiagnosticHoldMeshPixelRenderer.render(
        rendered_at_s=1.0,
        metadata=_metadata(duration),
        scene=scene,
        model=model,
    )) for duration in range(4))


def run() -> dict:
    metadata = tuple((duration, _metadata(duration)) for duration in range(4))
    scenes = (
        VisibleMeshScene(),
        VisibleMeshScene(zoom=0.7, camera_rotation_rad=0.4),
        VisibleMeshScene(zoom=1.4, camera_rotation_rad=-0.8),
    )
    sensor = HoldMeshPixelDurationSensor()
    model_rows = []
    for model in DiagnosticCurveModel:
        scene_rows = []
        for scene in scenes:
            templates = _templates(model, scene)
            observations = []
            for duration, frame in templates:
                result = sensor.observe(
                    observed_at_s=1.0,
                    frame=frame,
                    candidate_templates=templates,
                )
                observations.append({
                    "true_duration_evaluator_only": duration,
                    "candidates": result.candidates,
                    "valid": result.valid,
                    "reason": result.reason,
                })
            scene_rows.append({
                "zoom": scene.zoom,
                "rotation_rad": scene.camera_rotation_rad,
                "observations": observations,
            })
        model_rows.append({"model": model.value, "scenes": scene_rows})
    buffer_rows = [{
        "true_duration_evaluator_only": duration,
        "segments": item.longitudinal_segments,
        "vertices": item.vertex_count,
        "candidates_from_buffer": mesh_buffer_duration_candidates(
            observed_vertex_count=item.vertex_count,
            base_metadata_by_duration=metadata,
        ),
    } for duration, item in metadata]
    checks = {
        "buffer_identifies_0_to_3": all(
            row["candidates_from_buffer"] == (row["true_duration_evaluator_only"],)
            for row in buffer_rows
        ),
        "opaque_positive_duration_ambiguous": all(
            len(observation["candidates"]) > 1
            for model in model_rows if model["model"] == "opaque_constant_radius"
            for scene in model["scenes"]
            for observation in scene["observations"]
            if observation["true_duration_evaluator_only"] > 0
        ),
        "proxy_unique_all_scenes": all(
            observation["candidates"]
            == (observation["true_duration_evaluator_only"],)
            for model in model_rows if model["model"] == "proxy_radial_drift"
            for scene in model["scenes"]
            for observation in scene["observations"]
        ),
    }
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "HOLD_LIVE_VALIDATION_REQUIRED",
        "checks": checks,
        "mesh_buffer_diagnostic": buffer_rows,
        "pixel_counterfactuals": model_rows,
        "causality": {
            "student_pixel_fields": "frame dimensions, lit pixels, render/observe timestamps",
            "student_receives_hold_length": False,
            "student_receives_vertex_count": False,
            "student_receives_target_or_release_truth": False,
            "truth_used_for_posthoc_rows_only": True,
        },
        "unverified": {
            "stock_vertex_shader_formula": True,
            "live_pixel_capture": True,
            "first_visible_time": True,
            "capture_latency": True,
            "unique_duration_before_44ms_deadline": True,
        },
    }


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    _write_new(args.output, result)
    print("verdict:", result["verdict"])
    print("checks:", result["checks"])


if __name__ == "__main__":
    main()
