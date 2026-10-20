from __future__ import annotations

"""Stock CPU Hold-mesh metadata and pixel-only observability probes.

The exact audited ``scrHoldRenderer.CreateMesh`` CPU calculations are kept
separate from rendered pixels.  A Student may consume ``RenderedHoldMeshFrame``
only; it never receives the mesh segment count, angle length, or Hold duration.

The stock vertex shader was not present in the audited source snapshot.  The
pixel renderer below therefore provides explicit diagnostic models, not a
claim of Unity pixel parity.
"""

from dataclasses import dataclass
from enum import Enum
from math import atan2, cos, floor, isfinite, pi, sin


PI_STOCK = 3.1415927410125732
TWO_PI_STOCK = 6.2831854820251465
MULTIPLANET_PI = 3.1415926


def _mod(value: float, modulus: float) -> float:
    result = value % modulus
    return result + modulus if result < 0.0 else result


def _angle_moved(entry: float, exit: float, clockwise: bool) -> float:
    sign = 1.0 if clockwise else -1.0
    return _mod((exit - entry) * sign, TWO_PI_STOCK)


@dataclass(frozen=True, slots=True)
class StockHoldMeshMetadata:
    """Privileged renderer artifact. Never a Student observation."""

    shader_angle_length_rad: float
    timing_angle_length_rad: float
    longitudinal_segments: int
    vertex_count: int
    triangle_index_count: int
    added_multiplanet_loops: int


class StockHoldMeshBuilder:
    @staticmethod
    def build(
        *,
        entry_angle_rad: float,
        exit_angle_rad: float,
        is_ccw: bool,
        num_planets: int,
        midspin: bool,
        duration: int,
        fool_swirl: bool = False,
    ) -> StockHoldMeshMetadata | None:
        """Reproduce audited CreateMesh CPU parameters before the shader."""
        if any(not isfinite(x) for x in (entry_angle_rad, exit_angle_rad)):
            raise ValueError("mesh angles must be finite")
        if type(num_planets) is not int or num_planets < 2:
            raise ValueError("num_planets must be an integer >= 2")
        if type(midspin) is not bool or type(fool_swirl) is not bool:
            raise TypeError("mesh flags must be bool")
        if type(duration) is not int or duration < 0:
            raise ValueError("duration must be a nonnegative integer")
        # scrHoldRenderer returns before allocating a mesh for Midspin.
        if midspin:
            return None

        direction = is_ccw if fool_swirl else not is_ccw
        raw_moved = _angle_moved(entry_angle_rad, exit_angle_rad, direction)
        inverse = MULTIPLANET_PI * (num_planets - 2.0) / num_planets
        timing_angle = raw_moved - inverse
        added_loops = 0
        if timing_angle < 0.001 and num_planets > 2:
            timing_angle += PI_STOCK * 2.0
            added_loops = 1
        timing_angle += PI_STOCK * float(duration * 2)

        shader_angle = raw_moved
        magnitude = abs(shader_angle)
        if magnitude <= 1e-6 or magnitude >= PI_STOCK * 2.0 - 1e-6:
            shader_angle = PI_STOCK * 2.0
        shader_angle += float(duration + added_loops) * PI_STOCK * 2.0
        segments = int(floor(100.0 * (shader_angle / (PI_STOCK * 2.0))))
        return StockHoldMeshMetadata(
            shader_angle_length_rad=shader_angle,
            timing_angle_length_rad=timing_angle,
            longitudinal_segments=segments,
            vertex_count=2 + segments * 2,
            triangle_index_count=segments * 2 * 3,
            added_multiplanet_loops=added_loops,
        )


class DiagnosticCurveModel(str, Enum):
    OPAQUE_CONSTANT_RADIUS = "opaque_constant_radius"
    PROXY_RADIAL_DRIFT = "proxy_radial_drift"


@dataclass(frozen=True, slots=True)
class VisibleMeshScene:
    """Camera/floor quantities measurable from the rendered scene."""

    width_px: int = 192
    height_px: int = 192
    center_x_px: float = 96.0
    center_y_px: float = 96.0
    radius_px: float = 48.0
    zoom: float = 1.0
    camera_rotation_rad: float = 0.0

    def __post_init__(self) -> None:
        if type(self.width_px) is not int or type(self.height_px) is not int:
            raise TypeError("pixel dimensions must be integers")
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("pixel dimensions must be positive")
        if any(not isfinite(x) for x in (
            self.center_x_px, self.center_y_px, self.radius_px,
            self.zoom, self.camera_rotation_rad,
        )):
            raise ValueError("visible mesh scene must be finite")
        if self.radius_px <= 0.0 or self.zoom <= 0.0:
            raise ValueError("radius and zoom must be positive")


@dataclass(frozen=True, slots=True)
class RenderedHoldMeshFrame:
    """Pixel-only renderer/sensor boundary."""

    rendered_at_s: float
    width_px: int
    height_px: int
    lit_pixels: frozenset[tuple[int, int]]
    frame_available: bool = True

    def __post_init__(self) -> None:
        if not isfinite(self.rendered_at_s):
            raise ValueError("rendered_at_s must be finite")
        if type(self.width_px) is not int or type(self.height_px) is not int:
            raise TypeError("pixel dimensions must be integers")
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("pixel dimensions must be positive")
        if type(self.frame_available) is not bool:
            raise TypeError("frame_available must be bool")
        if not self.frame_available and self.lit_pixels:
            raise ValueError("unavailable frame cannot contain pixels")
        if any(
            type(x) is not int or type(y) is not int
            or not 0 <= x < self.width_px or not 0 <= y < self.height_px
            for x, y in self.lit_pixels
        ):
            raise ValueError("lit pixel is outside the frame")


class DiagnosticHoldMeshPixelRenderer:
    """Rasterize explicit shader hypotheses for counterfactual tests only."""

    @staticmethod
    def render(
        *,
        rendered_at_s: float,
        metadata: StockHoldMeshMetadata,
        scene: VisibleMeshScene,
        model: DiagnosticCurveModel,
        occlude_right_fraction: float = 0.0,
    ) -> RenderedHoldMeshFrame:
        if not isinstance(metadata, StockHoldMeshMetadata):
            raise TypeError("metadata must be StockHoldMeshMetadata")
        if not isinstance(scene, VisibleMeshScene):
            raise TypeError("scene must be VisibleMeshScene")
        if not isinstance(model, DiagnosticCurveModel):
            raise TypeError("model must be DiagnosticCurveModel")
        if not 0.0 <= occlude_right_fraction < 1.0:
            raise ValueError("occlusion fraction must lie in [0,1)")

        angle_length = metadata.shader_angle_length_rad
        sample_count = max(2, metadata.longitudinal_segments + 1)
        pixels: set[tuple[int, int]] = set()
        if (
            model is DiagnosticCurveModel.OPAQUE_CONSTANT_RADIUS
            and angle_length >= TWO_PI_STOCK
        ):
            # An ideal opaque continuous circle has no traversal count. Use a
            # canonical full-circle raster so floating-point resampling cannot
            # accidentally leak the number of repeated revolutions.
            samples = tuple(
                (index / 720.0, TWO_PI_STOCK * index / 720.0)
                for index in range(721)
            )
        else:
            samples = tuple(
                (index / (sample_count - 1), angle_length * index / (sample_count - 1))
                for index in range(sample_count)
            )
        for progress, curve_angle in samples:
            angle = scene.camera_rotation_rad + curve_angle
            radius = scene.radius_px * scene.zoom
            if model is DiagnosticCurveModel.PROXY_RADIAL_DRIFT:
                # Explicit proxy for a shader that separates overlapping turns.
                # It is not asserted to be the stock Unity shader.
                radius *= 0.72 + 0.28 * progress
            x = int(round(scene.center_x_px + radius * cos(angle)))
            y = int(round(scene.center_y_px + radius * sin(angle)))
            if not 0 <= x < scene.width_px or not 0 <= y < scene.height_px:
                continue
            cutoff = scene.width_px * (1.0 - occlude_right_fraction)
            if x >= cutoff:
                continue
            pixels.add((x, y))
        return RenderedHoldMeshFrame(
            float(rendered_at_s), scene.width_px, scene.height_px,
            frozenset(pixels),
        )


@dataclass(frozen=True, slots=True)
class HoldDurationObservation:
    observed_at_s: float
    rendered_at_s: float
    valid: bool
    candidates: tuple[int, ...]
    confidence: float | None
    reason: str

    def __post_init__(self) -> None:
        if not isfinite(self.observed_at_s) or not isfinite(self.rendered_at_s):
            raise ValueError("observation times must be finite")
        if self.rendered_at_s > self.observed_at_s + 1e-12:
            raise ValueError("future frames are not causal observations")
        if tuple(sorted(set(self.candidates))) != self.candidates:
            raise ValueError("duration candidates must be sorted and unique")
        if any(type(value) is not int or value < 0 for value in self.candidates):
            raise ValueError("duration candidates must be nonnegative integers")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie in [0,1]")
        if not self.reason:
            raise ValueError("observation reason is required")


class HoldMeshPixelDurationSensor:
    """Template matcher over pixels only; ambiguous matches remain a set."""

    def __init__(self, *, max_frame_age_s: float = 0.050) -> None:
        if not isfinite(max_frame_age_s) or max_frame_age_s < 0.0:
            raise ValueError("max_frame_age_s must be finite and nonnegative")
        self.max_frame_age_s = float(max_frame_age_s)

    def observe(
        self,
        *,
        observed_at_s: float,
        frame: RenderedHoldMeshFrame,
        candidate_templates: tuple[tuple[int, RenderedHoldMeshFrame], ...],
    ) -> HoldDurationObservation:
        if not isfinite(observed_at_s):
            raise ValueError("observed_at_s must be finite")
        if frame.rendered_at_s > observed_at_s + 1e-12:
            raise ValueError("sensor cannot consume a future frame")
        stale = observed_at_s - frame.rendered_at_s > self.max_frame_age_s + 1e-12
        if not frame.frame_available or stale:
            return HoldDurationObservation(
                observed_at_s, frame.rendered_at_s, False, (), None,
                "missing_or_stale_frame",
            )
        matches = []
        for duration, template in candidate_templates:
            if (template.width_px, template.height_px) != (
                frame.width_px, frame.height_px
            ):
                raise ValueError("candidate template dimensions differ")
            if template.lit_pixels == frame.lit_pixels:
                matches.append(duration)
        candidates = tuple(sorted(set(matches)))
        if len(candidates) == 1:
            return HoldDurationObservation(
                observed_at_s, frame.rendered_at_s, True, candidates, 1.0,
                "unique_pixel_template",
            )
        if candidates:
            return HoldDurationObservation(
                observed_at_s, frame.rendered_at_s, True, candidates, None,
                "ambiguous_pixel_templates",
            )
        return HoldDurationObservation(
            observed_at_s, frame.rendered_at_s, False, (), None,
            "no_exact_pixel_template",
        )


def mesh_buffer_duration_candidates(
    *,
    observed_vertex_count: int,
    base_metadata_by_duration: tuple[tuple[int, StockHoldMeshMetadata], ...],
) -> tuple[int, ...]:
    """Diagnostic mesh-buffer classifier, explicitly not a pixel sensor."""
    if type(observed_vertex_count) is not int or observed_vertex_count < 0:
        raise ValueError("observed vertex count must be nonnegative integer")
    return tuple(
        duration for duration, metadata in base_metadata_by_duration
        if metadata.vertex_count == observed_vertex_count
    )
