from __future__ import annotations

"""Evaluator-only topology reconstruction of one audited Stock Hold mesh.

This is NOT a visual sensor. The decompiled scrHoldRenderer.CreateMesh uses
privileged startFloor.holdLength to build GPU input data, while the rendered
pixels depend on a shader absent from the C# source snapshot. Never import this
module into a Student, TimingSource, or Hold perception pipeline.
"""

from dataclasses import dataclass
from math import floor, isfinite, pi
from struct import pack, unpack


STOCK_HOLD_RENDERER_SHA256 = "b6c19c740a21aa04a8191de952990345b0034042a9f9f413284f2ce5ced8ad80"
STOCK_PI_F32 = unpack("<f", pack("<f", pi))[0]
STOCK_TWO_PI_F32 = unpack("<f", pack("<f", STOCK_PI_F32 * 2.0))[0]
SEGMENTS_PER_LOOP = 100


def _f32(value: float) -> float:
    return unpack("<f", pack("<f", value))[0]


@dataclass(frozen=True, slots=True)
class EvaluatorOnlyHoldMeshTopology:
    """Numbers used to construct a mesh; NOT evidence of visible loops."""

    adjusted_angle_rad: float
    segment_count: int
    vertex_count: int
    triangle_count: int
    uv0_last_y: float
    uv1_last_x: int
    hold_length_privileged: int


def reconstruct_stock_mesh_topology(
    *, entry_angle_rad: float, exit_angle_rad: float,
    is_ccw: bool, hold_length: int, added_loops: int = 0,
    midspin: bool = False, fool_swirl: bool = False,
) -> EvaluatorOnlyHoldMeshTopology | None:
    """Reproduce CreateMesh's num2/num4/num5/num6; no shader projection.

    added_loops is an explicit renderer-local diagnostic value. The audited
    source increments it in a special MultiPlanet branch; no policy can read it.
    """
    if type(midspin) is not bool or type(is_ccw) is not bool or type(fool_swirl) is not bool:
        raise TypeError("geometry flags must be bool")
    if type(hold_length) is not int or hold_length < 0:
        raise ValueError("hold_length must be a nonnegative integer")
    if type(added_loops) is not int or added_loops < 0:
        raise ValueError("added_loops must be a nonnegative integer")
    if hold_length + added_loops > 10000:
        raise ValueError("diagnostic loop count exceeds safe bound")
    if not isfinite(entry_angle_rad) or not isfinite(exit_angle_rad):
        raise ValueError("angles must be finite")
    if midspin:
        return None  # scrHoldRenderer.CreateMesh immediately returns.

    # scrMisc.GetAngleMoved(entry, exit, isCW), where C# uses double modulo
    # and 6.2831854820251465 (float32 2pi). FOOL_SWIRL reverses isCW.
    is_cw = is_ccw if fool_swirl else not is_ccw
    sign = 1.0 if is_cw else -1.0
    modulus = STOCK_TWO_PI_F32
    delta = (exit_angle_rad - entry_angle_rad) * sign
    raw_arc = ((delta % modulus) + modulus) % modulus
    if abs(raw_arc) <= 1e-6 or abs(raw_arc) >= modulus - 1e-6:
        raw_arc = modulus

    # C#: num2 += (double)((float)num * MathF.PI * 2f)
    loop_increment = _f32(_f32(_f32(float(hold_length + added_loops)) * STOCK_PI_F32) * 2.0)
    adjusted = raw_arc + loop_increment
    # C#: Mathf.Floor(100f * ((float)num2 / (MathF.PI * 2f)))
    segment_count = floor(_f32(_f32(_f32(adjusted) / modulus) * SEGMENTS_PER_LOOP))
    if segment_count < 0:
        raise ValueError("mesh segment count invalid")
    # num5 = 2 + num4 * 2, num6 = num4 * 2; triangles = 2*num4.
    return EvaluatorOnlyHoldMeshTopology(
        adjusted_angle_rad=adjusted,
        segment_count=segment_count,
        vertex_count=2 + 2 * segment_count,
        triangle_count=2 * segment_count,
        uv0_last_y=segment_count / SEGMENTS_PER_LOOP,
        uv1_last_x=1 + 2 * segment_count,
        hold_length_privileged=hold_length,
    )


def mesh_counters_are_policy_visible() -> bool:
    """Fail-closed contract: Stock mesh counters are NOT observed pixels."""
    return False
