from __future__ import annotations

import pytest

from dmdod.planning.motor_lead_feasibility_v01 import (
    EvidenceState,
    assess_lead,
    calibrated_nominal_lead_s,
    next_control_tick,
)


DUE = 3.2205882724022707
WINDOW = (3.1555882724022708, 3.2855882724022707)


def test_44ms_is_39ms_cold_measurement_plus_half_control_tick() -> None:
    assert calibrated_nominal_lead_s(
        cold_press_latency_s=0.039, control_dt_s=0.010
    ) == pytest.approx(0.044)


def test_control_tick_quantisation_does_not_move_request_backwards() -> None:
    assert next_control_tick(3.190, control_dt_s=0.010) == pytest.approx(3.190)
    assert next_control_tick(3.190588, control_dt_s=0.010) == pytest.approx(3.200)


@pytest.mark.parametrize(
    ("lead_s", "temporal", "by_due", "in_window"),
    [
        (0.044, False, False, True),
        (0.030, True, False, True),
        (0.020, True, False, True),
        (0.010, True, False, True),
        (0.000, True, False, True),
    ],
)
def test_event37_existing_forecast_separates_temporal_motor_and_stock(
    lead_s: float, temporal: bool, by_due: bool, in_window: bool
) -> None:
    result = assess_lead(
        observed_at_s=3.190,
        due_s_posthoc=DUE,
        lead_s=lead_s,
        due_observability=EvidenceState.YES,
        stock_window_s=WINDOW,
    )
    assert result.nominal_lead_temporally_available is temporal
    assert result.cold_keydown_by_due is by_due
    assert result.cold_keydown_in_stock_window is in_window
    assert not result.all_counted_hit_conditions  # actual free-axis proof is absent


def test_oracle_exact_due_at_3180_is_not_relabelled_as_observation() -> None:
    result = assess_lead(
        observed_at_s=3.180,
        due_s_posthoc=DUE,
        lead_s=0.030,
        due_observability=EvidenceState.UNVERIFIED,
        stock_window_s=WINDOW,
        reservation_conflict_clear=EvidenceState.YES,
    )
    assert result.nominal_lead_temporally_available
    assert result.due_observability is EvidenceState.UNVERIFIED
    assert not result.all_due_aligned_conditions
    assert not result.all_counted_hit_conditions


def test_bad_windows_and_negative_leads_are_rejected() -> None:
    with pytest.raises(ValueError):
        assess_lead(
            observed_at_s=0.0,
            due_s_posthoc=1.0,
            lead_s=-0.001,
            due_observability=EvidenceState.YES,
            stock_window_s=(0.9, 1.1),
        )
    with pytest.raises(ValueError):
        assess_lead(
            observed_at_s=0.0,
            due_s_posthoc=1.0,
            lead_s=0.0,
            due_observability=EvidenceState.YES,
            stock_window_s=(1.1, 0.9),
        )
