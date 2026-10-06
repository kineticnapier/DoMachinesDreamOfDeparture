from __future__ import annotations

import json

from dmdod.training.action_trust import ActionTrustMetrics
from dmdod.training.trajectory_trust import (
    BoundaryDivergence,
    DivergencePoint,
    TrajectoryProbeResult,
    _SquaredAccumulator,
    format_probe_result,
    save_probe_report,
)


def test_squared_accumulator_tracks_rms_and_max() -> None:
    acc = _SquaredAccumulator()
    acc.add((0.0, 2.0), (1.0, 0.0))

    assert abs(acc.rms - (2.5 ** 0.5)) < 1e-12
    assert acc.maximum == 2.0


def test_probe_report_serializes_divergence_points(tmp_path) -> None:
    point = DivergencePoint(
        step=12,
        time_s=0.12,
        target_ordinal=7,
        baseline=(1, 2),
        candidate=(1, 3),
    )
    boundary = BoundaryDivergence(
        step=12,
        time_s=0.121,
        target_ordinal=7,
        key="right_1",
        event="down",
        threshold_m=0.002,
        baseline_event=False,
        candidate_event=True,
        baseline_position_m=0.00199994,
        candidate_position_m=0.00200002,
        position_delta_m=0.00000008,
        baseline_margin_m=-0.00000006,
        candidate_margin_m=0.00000002,
        baseline_action=0.123456,
        candidate_action=0.123500,
    )
    result = TrajectoryProbeResult(
        anchor_index=2,
        chart_name="probe",
        steps=20,
        action_rms=2e-5,
        action_max=2e-4,
        position_rms_m=1e-6,
        position_max_m=2e-6,
        velocity_rms_m_s=3e-5,
        activation_rms=4e-5,
        fatigue_rms=5e-6,
        hand_fatigue_rms=6e-6,
        coordination_rms=7e-6,
        pressed_mismatch_frames=3,
        first_pressed_divergence=point,
        first_keydown_divergence=point,
        first_score_divergence=point,
        first_overload_divergence=None,
        baseline_hits=10,
        candidate_hits=9,
        baseline_misses=1,
        candidate_misses=2,
        baseline_too_early=0,
        candidate_too_early=1,
        baseline_overloaded=False,
        candidate_overloaded=True,
        baseline_keydowns=11,
        candidate_keydowns=12,
        stopped_side="candidate",
        first_boundary_divergence=boundary,
    )
    metrics = ActionTrustMetrics(
        objective=1.0,
        teacher_loss=0.9,
        stay_loss=0.005,
        grad_norm=2.0,
        action_rms=2e-5,
        action_max=2e-4,
        accepted_inner_steps=8,
        final_lr=1e-8,
    )
    path = tmp_path / "probe.json"

    save_probe_report(
        path,
        candidate_metrics=metrics,
        results=[result],
        source_checkpoint="source.pt",
        configured_action_rms=2.5e-5,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["kind"] == "trajectory-divergence-probe"
    assert payload["configured_action_rms"] == 2.5e-5
    assert payload["anchors"][0]["first_score_divergence"]["target_ordinal"] == 7
    assert payload["anchors"][0]["first_boundary_divergence"]["key"] == "right_1"
    assert payload["anchors"][0]["first_boundary_divergence"]["candidate_event"] is True
    assert payload["anchors"][0]["candidate_overloaded"] is True


def test_probe_summary_contains_physical_divergence() -> None:
    point = DivergencePoint(12, 0.12, 7, 10, 11)
    boundary = BoundaryDivergence(
        step=12,
        time_s=0.121,
        target_ordinal=7,
        key="right_1",
        event="down",
        threshold_m=0.002,
        baseline_event=False,
        candidate_event=True,
        baseline_position_m=0.00199994,
        candidate_position_m=0.00200002,
        position_delta_m=0.00000008,
        baseline_margin_m=-0.00000006,
        candidate_margin_m=0.00000002,
        baseline_action=0.123456,
        candidate_action=0.123500,
    )
    result = TrajectoryProbeResult(
        anchor_index=2,
        chart_name="probe",
        steps=20,
        action_rms=2e-5,
        action_max=2e-4,
        position_rms_m=1e-6,
        position_max_m=2e-6,
        velocity_rms_m_s=3e-5,
        activation_rms=4e-5,
        fatigue_rms=5e-6,
        hand_fatigue_rms=6e-6,
        coordination_rms=7e-6,
        pressed_mismatch_frames=3,
        first_pressed_divergence=point,
        first_keydown_divergence=point,
        first_score_divergence=point,
        first_overload_divergence=None,
        baseline_hits=10,
        candidate_hits=9,
        baseline_misses=1,
        candidate_misses=2,
        baseline_too_early=0,
        candidate_too_early=1,
        baseline_overloaded=False,
        candidate_overloaded=True,
        baseline_keydowns=11,
        candidate_keydowns=12,
        stopped_side="candidate",
        first_boundary_divergence=boundary,
    )

    text = format_probe_result(result)

    assert "anchor 02" in text
    assert "boundary=[0.121s right_1 DOWN" in text
    assert "thr=2.000000mm" in text
    assert "d=+0.080um" in text
    assert "margin=-0.060/+0.020um" in text
    assert "event=0/1" in text
    assert "key-div=0.120s/t7" in text
    assert "H=10->9" in text
    assert "over=False->True" in text
