from __future__ import annotations

"""Relocate empty sampled windows onto the nearest playable target.

The original multi-chart trainer assumes every requested anchor/validation/round
window contains at least one playable floor.  Larger public datasets can contain
long intros, breaks, or outros, so a perfectly valid chart may violate that
assumption.  Keep the requested window length, but only when a window is empty,
move it to the nearest playable target instead of dropping the chart/guard.
"""

from typing import Iterable

import train_real_chart_v080 as v080


_ORIGINAL_NAMED_SEGMENT = None


def _relocate_window(
    *,
    duration_s: float,
    start_s: float,
    end_s: float,
    target_times_s: Iterable[float],
) -> tuple[float, float]:
    duration = float(duration_s)
    start = float(start_s)
    end = float(end_s)
    if duration <= 0.0:
        raise ValueError("chart duration must be positive")
    if end <= start:
        raise ValueError("segment end must be greater than start")

    targets = tuple(float(value) for value in target_times_s)
    if not targets:
        raise ValueError("chart has no playable targets")

    length = min(end - start, duration)
    midpoint = (start + end) * 0.5
    nearest = min(targets, key=lambda value: (abs(value - midpoint), value))

    max_start = max(0.0, duration - length)
    relocated_start = min(max(0.0, nearest - length * 0.5), max_start)
    relocated_end = relocated_start + length
    return relocated_start, relocated_end


def _playable_target_times(runtime) -> tuple[float, ...]:
    return tuple(
        float(floor.target_time_s)
        for floor in runtime.compiled.floors[1:]
        if not bool(floor.midspin)
    )


def _named_segment_nonempty(runtime, role: str, start_s: float, end_s: float):
    if _ORIGINAL_NAMED_SEGMENT is None:
        raise RuntimeError("nonempty segment relocation is not installed")

    try:
        return _ORIGINAL_NAMED_SEGMENT(runtime, role, start_s, end_s)
    except ValueError as exc:
        if "segment contains no playable targets:" not in str(exc):
            raise

    relocated_start, relocated_end = _relocate_window(
        duration_s=runtime.duration_s,
        start_s=start_s,
        end_s=end_s,
        target_times_s=_playable_target_times(runtime),
    )
    print(
        f"segment-relocate: role={role} chart={runtime.spec.name} "
        f"empty={start_s:.3f}..{end_s:.3f}s -> "
        f"{relocated_start:.3f}..{relocated_end:.3f}s"
    )
    return _ORIGINAL_NAMED_SEGMENT(runtime, role, relocated_start, relocated_end)


def install_nonempty_segment_relocation() -> None:
    global _ORIGINAL_NAMED_SEGMENT
    if _ORIGINAL_NAMED_SEGMENT is not None:
        return
    _ORIGINAL_NAMED_SEGMENT = v080._named_segment
    v080._named_segment = _named_segment_nonempty
