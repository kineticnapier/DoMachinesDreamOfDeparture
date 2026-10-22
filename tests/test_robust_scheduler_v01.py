from dmdod.research.robust_scheduler_v01 import (
    DueHypothesis,
    FeasibilityStatus,
    FloorStateHypothesis,
    Reservation,
    TimeInterval,
    assess_reservation,
)


def due(name, window, causal=True):
    return DueHypothesis(name, TimeInterval(*window), TimeInterval(*window), causal)


def floor(name, target="e37", mode="no_fail", supported=True):
    return FloorStateHypothesis(name, "e37", target, mode, supported)


def test_safe_requires_all_hypotheses_to_agree() -> None:
    result = assess_reservation(
        intended_event_id="e37", axis="left_1",
        motor_arrival=TimeInterval(3.22, 3.23), motor_envelope_valid=True,
        due_hypotheses=(due("a", (3.20, 3.24)), due("b", (3.21, 3.25))),
        floor_hypotheses=(floor("f1"), floor("f2")), declared_run_mode="no_fail",
    )
    assert result.status is FeasibilityStatus.SAFE


def test_counterfactual_disjoint_windows_force_abstention() -> None:
    result = assess_reservation(
        intended_event_id="e37", axis="left_1",
        motor_arrival=TimeInterval(3.22, 3.23), motor_envelope_valid=True,
        due_hypotheses=(due("original", (3.15, 3.28)), due("alternate", (3.41, 3.65))),
        floor_hypotheses=(floor("f"),), declared_run_mode="no_fail",
    )
    assert result.status is FeasibilityStatus.ABSTAIN
    assert any("alternate" in reason for reason in result.reasons)


def test_floor_uncertainty_or_mode_mismatch_never_claims_safe() -> None:
    result = assess_reservation(
        intended_event_id="e37", axis="left_1",
        motor_arrival=TimeInterval(3.22, 3.23), motor_envelope_valid=True,
        due_hypotheses=(due("d", (3.2, 3.3)),),
        floor_hypotheses=(floor("old", target="e36"), floor("mode", mode="fail")),
        declared_run_mode="no_fail",
    )
    assert result.status is FeasibilityStatus.ABSTAIN
    assert any("floor_attribution" in reason for reason in result.reasons)
    assert any("run_mode" in reason for reason in result.reasons)


def test_unknown_motor_or_noncausal_due_remains_unknown() -> None:
    motor = assess_reservation(
        intended_event_id="e37", axis="left_1", motor_arrival=None,
        motor_envelope_valid=False, due_hypotheses=(due("d", (3.2, 3.3)),),
        floor_hypotheses=(floor("f"),), declared_run_mode="no_fail",
    )
    leaked = assess_reservation(
        intended_event_id="e37", axis="left_1",
        motor_arrival=TimeInterval(3.22, 3.23), motor_envelope_valid=True,
        due_hypotheses=(due("oracle", (3.2, 3.3), causal=False),),
        floor_hypotheses=(floor("f"),), declared_run_mode="no_fail",
    )
    assert motor.status is FeasibilityStatus.UNKNOWN
    assert leaked.status is FeasibilityStatus.UNKNOWN


def test_reservation_conflict_preserves_baseline_hit() -> None:
    result = assess_reservation(
        intended_event_id="e37", axis="left_1",
        motor_arrival=TimeInterval(3.22, 3.23), motor_envelope_valid=True,
        due_hypotheses=(due("d", (3.2, 3.3)),), floor_hypotheses=(floor("f"),),
        reservations=(Reservation("baseline:e35", "left_1", TimeInterval(3.21, 3.24)),),
        declared_run_mode="no_fail",
    )
    assert result.status is FeasibilityStatus.ABSTAIN
    assert "baseline_hit_at_risk:baseline:e35" in result.reasons
