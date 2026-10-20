from __future__ import annotations

import pytest

from dmdod.planning.hud_timing_tracker_checkpoint_v18 import (
    normalize_visible_checkpoint_transition,
    stop_after_checkpoint,
)
from dmdod.planning.hud_timing_tracker_transition_v14 import (
    CURRENT_SLOT, FLOOR_DIM, HUD_DIM, SLOT_START, WindowForecast, visible_slot,
)
from dmdod.planning.hud_v3_composite_forecast_source_v30 import (
    HUDV3CompositeForecastSource,
)


def _hud(*, other_markers=(), relative=1):
    result = [0.0] * HUD_DIM
    for slot in (CURRENT_SLOT, CURRENT_SLOT + 1, CURRENT_SLOT + 2):
        result[SLOT_START + slot * FLOOR_DIM + FLOOR_DIM - 1] = 1.0
    for marker in (12, *other_markers):
        result[SLOT_START + (CURRENT_SLOT + relative) * FLOOR_DIM + marker] = 1.0
    return tuple(result)


def test_checkpoint_flag_is_removed_in_a_copy_only():
    hud = _hud()
    normalized = normalize_visible_checkpoint_transition(hud)
    assert normalized.recovered_floor_slots == (CURRENT_SLOT + 1,)
    assert visible_slot(hud, CURRENT_SLOT + 1)[12] == 1.0
    assert visible_slot(normalized.hud, CURRENT_SLOT + 1)[12] == 0.0
    assert visible_slot(hud, CURRENT_SLOT) == visible_slot(normalized.hud, CURRENT_SLOT)


def test_checkpoint_and_setspeed_can_recover_only_the_incoming_event():
    normalized = normalize_visible_checkpoint_transition(_hud(other_markers=(9,)))
    assert normalized.recovered_floor_slots == (CURRENT_SLOT + 1,)
    assert visible_slot(normalized.hud, CURRENT_SLOT + 1)[9] == 1.0


@pytest.mark.parametrize("marker", (6, 7, 8, 10, 11, 13))
def test_combined_unsafe_markers_remain_blocked(marker):
    hud = _hud(other_markers=(marker,))
    normalized = normalize_visible_checkpoint_transition(hud)
    assert normalized.hud == hud
    assert normalized.recovered_floor_slots == ()


def test_forecast_is_truncated_at_checkpoint_even_without_setspeed():
    forecast = WindowForecast((1, 2, 3), (.010, .020, .030), "eligible")
    truncated = stop_after_checkpoint(forecast, (CURRENT_SLOT + 2,))
    assert truncated.floor_offsets == (1, 2)
    assert truncated.times_s == (.010, .020)
    assert truncated.status == "eligible_partial"
    assert truncated.covered_until_s == .020


def test_no_checkpoint_in_window_does_not_change_forecast():
    forecast = WindowForecast((1,), (.015,), "eligible")
    assert stop_after_checkpoint(forecast, (CURRENT_SLOT + 3,)) is forecast


def test_opt_in_requires_v15_and_default_is_false():
    baseline = HUDV3CompositeForecastSource(lifecycle_version="v1.5")
    candidate = HUDV3CompositeForecastSource(
        lifecycle_version="v1.5", midspin_recovery=True,
        twirl_recovery=True, checkpoint_recovery=True,
    )
    assert baseline.checkpoint_recovery is False
    assert candidate.checkpoint_recovery is True
    candidate.reset()
    with pytest.raises(ValueError, match="requires lifecycle v1.5"):
        HUDV3CompositeForecastSource(
            lifecycle_version="v1.4", checkpoint_recovery=True
        )
    with pytest.raises(TypeError, match="checkpoint_recovery must be bool"):
        HUDV3CompositeForecastSource(
            lifecycle_version="v1.5", checkpoint_recovery=1
        )


def test_checkpoint_recovery_does_not_enable_twirl_implicitly():
    source = HUDV3CompositeForecastSource(
        lifecycle_version="v1.5", checkpoint_recovery=True
    )
    assert not source.twirl_recovery
    assert not source.midspin_recovery
    assert source.checkpoint_recovery is True
