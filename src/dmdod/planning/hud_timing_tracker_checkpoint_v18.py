from __future__ import annotations

"""Opt-in HUD-only recovery for arrival at one visible Stock Checkpoint floor.

Checkpoint is a hit-triggered save/revival effect, not a modification of the
arrival-time geometry. Removing its visible marker in a *calculation copy*
allows the frozen v1.4 incoming-edge computation; stop_after_checkpoint then
truncates the output at the arrival. The future SetSpeed/Pause gate remains.
No compiled charts, target times or event action numeric fields are read.
"""

from dataclasses import dataclass

from dmdod.planning.hud_timing_tracker_transition_v14 import (
    CURRENT_SLOT,
    FLOOR_DIM,
    FLOORS,
    HUD_DIM,
    SLOT_START,
    WindowForecast,
    visible_slot,
)


@dataclass(frozen=True, slots=True)
class CheckpointNormalization:
    hud: tuple[float, ...]
    recovered_floor_slots: tuple[int, ...]


def normalize_visible_checkpoint_transition(
    hud: tuple[float, ...],
) -> CheckpointNormalization:
    """Allow only one future ordinary 2-planet Checkpoint, +/- SetSpeed.

    Other markers (Twirl, MultiPlanet, Pause, SetFloorIcon) remain blocked;
    in particular, never infer unknown future speed or pause duration.
    """
    if len(hud) != HUD_DIM:
        return CheckpointNormalization(hud, ())
    for slot in range(
        CURRENT_SLOT + 1, CURRENT_SLOT + FLOORS.ahead_floors + 1
    ):
        floor = visible_slot(hud, slot)
        if not (floor[-1] > .5 and floor[12] > .5):
            continue
        if (
            floor[6] > .5
            or floor[7] > .5
            or any(floor[index] > .5 for index in (8, 10, 11, 13))
        ):
            return CheckpointNormalization(hud, ())
        normalized = list(hud)
        normalized[SLOT_START + slot * FLOOR_DIM + 12] = 0.0
        return CheckpointNormalization(tuple(normalized), (slot,))
    return CheckpointNormalization(hud, ())


def stop_after_checkpoint(
    forecast: WindowForecast,
    recovered_slots: tuple[int, ...],
) -> WindowForecast:
    """A licensed Checkpoint arrival is the last emitted event of the window."""
    if not forecast.eligible or not recovered_slots:
        return forecast
    offset = recovered_slots[0] - CURRENT_SLOT
    if offset not in forecast.floor_offsets:
        return forecast
    index = forecast.floor_offsets.index(offset)
    due = forecast.times_s[index]
    return WindowForecast(
        forecast.floor_offsets[:index + 1],
        forecast.times_s[:index + 1],
        "eligible_partial",
        due,
    )
