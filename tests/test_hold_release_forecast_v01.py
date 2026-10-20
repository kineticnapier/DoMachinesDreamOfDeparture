from __future__ import annotations

from dataclasses import fields
from math import pi

import pytest

from dmdod.evaluation.hold_judgement_v01 import TimeWindow
from dmdod.features.hold_mesh_observability_v01 import HoldDurationObservation
from dmdod.planning.hold_release_forecast_v01 import (
    HoldReleaseForecastRequest,
    HoldReleaseForecastStatus,
    HoldVisualEvidence,
    MotorReleaseLatency,
    VisibleHoldKinematics,
    forecast_hold_release,
)


def _duration(candidates=(2,), *, valid=True, observed_at=1.0):
    return HoldDurationObservation(
        observed_at, observed_at - 0.01, valid, candidates,
        1.0 if valid and len(candidates) == 1 else None,
        "fixture_visible_pixels",
    )


def _request(
    *,
    duration=None,
    evidence=HoldVisualEvidence.STOCK_PIXEL_VERIFIED,
    future_transition_free=True,
    bpm=TimeWindow(120.0, 120.0),
    angle=TimeWindow(pi, pi),
    start=TimeWindow(1.0, 1.0),
    latency=TimeWindow(0.020, 0.020),
    observed_at=1.0,
):
    return HoldReleaseForecastRequest(
        hold_start_local_id=7,
        start_due_s=start,
        duration=duration or _duration(observed_at=observed_at),
        evidence=evidence,
        kinematics=VisibleHoldKinematics(
            observed_at, angle, bpm, True, future_transition_free,
            "visible_floor_geometry_and_real_bpm",
        ),
        release_latency=MotorReleaseLatency(
            observed_at, latency, True
        ),
    )


def test_unique_verified_pixels_produce_release_and_command_intervals() -> None:
    result = forecast_hold_release(_request())
    # Base half-turn=.5s; two added full turns=2s at 120 BPM.
    assert result.release_due_s == TimeWindow(3.5, 3.5)
    assert result.release_command_s == TimeWindow(3.48, 3.48)
    assert result.forecast_lead_lower_s == pytest.approx(2.48)
    assert result.status is HoldReleaseForecastStatus.MOTOR_READY
    assert result.motor_ready


def test_duration_candidate_set_must_be_unique() -> None:
    result = forecast_hold_release(_request(duration=_duration((1, 2, 3))))
    assert result.status is HoldReleaseForecastStatus.ABSTAIN_DURATION
    assert result.duration_candidates == (1, 2, 3)
    assert result.release_due_s is None


def test_proxy_evidence_can_exercise_math_but_never_drive_motor() -> None:
    result = forecast_hold_release(_request(
        evidence=HoldVisualEvidence.DIAGNOSTIC_PROXY
    ))
    assert result.status is HoldReleaseForecastStatus.DIAGNOSTIC_ONLY
    assert result.release_due_s is not None
    assert not result.motor_ready


def test_unknown_future_setspeed_or_pause_keeps_safety_gate_closed() -> None:
    result = forecast_hold_release(_request(future_transition_free=False))
    assert result.status is HoldReleaseForecastStatus.ABSTAIN_FUTURE_TRANSITION
    assert result.reason == "DO_NOT_RELAX_SETSPEED_GATE"
    assert result.release_due_s is None


def test_bpm_start_and_latency_uncertainty_are_preserved() -> None:
    result = forecast_hold_release(_request(
        bpm=TimeWindow(119.0, 121.0),
        angle=TimeWindow(pi - 0.01, pi + 0.01),
        start=TimeWindow(0.995, 1.005),
        latency=TimeWindow(0.015, 0.025),
    ))
    assert result.release_due_s.lower_s < result.release_due_s.upper_s
    assert result.release_command_s.lower_s < result.release_command_s.upper_s
    assert result.status is HoldReleaseForecastStatus.ABSTAIN_UNCERTAINTY
    assert not result.motor_ready


def test_44ms_lead_uses_release_command_lower_bound() -> None:
    result = forecast_hold_release(_request(
        duration=_duration((0,), observed_at=1.44),
        start=TimeWindow(1.0, 1.0),
        angle=TimeWindow(pi, pi),
        latency=TimeWindow(0.020, 0.020),
        observed_at=1.44,
    ))
    assert result.release_command_s == TimeWindow(1.48, 1.48)
    assert result.forecast_lead_lower_s == pytest.approx(0.04)
    assert result.status is HoldReleaseForecastStatus.ABSTAIN_INSUFFICIENT_LEAD


def test_invalid_latency_and_kinematics_abstain_without_imputation() -> None:
    request = _request()
    invalid_latency = HoldReleaseForecastRequest(
        request.hold_start_local_id, request.start_due_s, request.duration,
        request.evidence, request.kinematics,
        MotorReleaseLatency(1.0, TimeWindow(0.0, 0.0), False),
    )
    assert forecast_hold_release(invalid_latency).status is (
        HoldReleaseForecastStatus.ABSTAIN_RELEASE_LATENCY
    )
    invalid_kinematics = HoldReleaseForecastRequest(
        request.hold_start_local_id, request.start_due_s, request.duration,
        request.evidence,
        VisibleHoldKinematics(
            1.0, TimeWindow(pi, pi), TimeWindow(120, 120), False, True, "missing"
        ),
        request.release_latency,
    )
    assert forecast_hold_release(invalid_kinematics).status is (
        HoldReleaseForecastStatus.ABSTAIN_KINEMATICS
    )


def test_forecast_contract_has_no_chart_truth_or_hidden_hold_length() -> None:
    public = {
        item.name
        for cls in (
            HoldReleaseForecastRequest, VisibleHoldKinematics,
            MotorReleaseLatency,
        )
        for item in fields(cls)
    }
    assert public.isdisjoint({
        "hold_length", "holdLength", "chart", "floor_index",
        "target_time_s", "true_release_s", "future_bpm",
    })
