from __future__ import annotations

from dmdod.features.judgement_observer_v01 import (
    JudgementDelta,
    JudgementDeltaStatus,
)
from dmdod.planning.judgement_state_estimator_v01 import (
    EstimateStatus,
    JudgementStateEstimator,
    MotorKeyDownEvidence,
    RepositoryWindowModel,
    TimingEventEvidence,
    TimingEventStatus,
)


def event(local_id: int, due: float, *, status=TimingEventStatus.CONFIRMED):
    return TimingEventEvidence(
        local_id=str(local_id),
        observed_at_s=due - 0.08,
        due_lower_s=due,
        due_upper_s=due,
        status=status,
        reason="fixture",
    )


def keydown(name: str, intended: int, at: float):
    return MotorKeyDownEvidence(
        input_id=name,
        intended_local_id=str(intended),
        occurred_at_s=at,
        observed_at_s=at + 0.001,
    )


def delta(at: float, **values: int):
    return JudgementDelta(
        status=JudgementDeltaStatus.DELTA,
        observed_at_s=at,
        displayed_at_s=at,
        deltas=tuple(values.items()),
        event_interval_s=(at - 0.010, at),
        delta_available=True,
        within_interval_order_known=False,
        reason="fixture",
    )


def estimator():
    return JudgementStateEstimator(
        RepositoryWindowModel(perfect_s=0.025, early_late_perfect_s=0.030, pass_s=0.065),
        max_predecessors=3,
    )


def test_id38_visible_very_late_identifies_predecessor_only_under_explicit_model():
    est = estimator()
    est.observe_timing(event(36, 3.183823566, status=TimingEventStatus.ABSTAINED))
    est.observe_timing(event(37, 3.220588273, status=TimingEventStatus.ABSTAINED))
    est.observe_timing(event(38, 3.264705920))
    est.observe_keydown(keydown("down-38", 38, 3.261))

    result = est.observe_judgement(delta(3.270, very_late=1))

    assert result.status is EstimateStatus.MODEL_IDENTIFIED
    assert {h.attributions for h in result.hypotheses} == {
        (("down-38", "38", "37", "very_late"),)
    }
    assert result.aligned_state_possible is False
    assert result.possible_unresolved_local_ids == ("38",)
    assert result.stock_identity_verified is False
    assert "simulator" in result.reason.lower()


def test_normal_very_late_remains_aligned_when_timing_predicts_very_late():
    est = estimator()
    est.observe_timing(event(10, 1.000))
    est.observe_keydown(keydown("late-but-aligned", 10, 1.040))

    result = est.observe_judgement(delta(1.050, very_late=1))

    assert result.status is EstimateStatus.MODEL_IDENTIFIED
    assert result.aligned_state_possible is True
    assert result.possible_unresolved_local_ids == ()


def test_overlapping_due_uncertainty_retains_multiple_state_hypotheses():
    est = estimator()
    est.observe_timing(event(1, 0.960))
    est.observe_timing(TimingEventEvidence(
        local_id="2", observed_at_s=0.9, due_lower_s=0.960, due_upper_s=1.000,
        status=TimingEventStatus.CONFIRMED, reason="wide HUD interval",
    ))
    est.observe_keydown(keydown("ambiguous", 2, 1.000))

    result = est.observe_judgement(delta(1.010, very_late=1))

    assert result.status is EstimateStatus.AMBIGUOUS
    assert {h.attributions[0][2] for h in result.hypotheses} == {"1", "2"}
    assert result.aligned_state_possible is True
    assert result.possible_unresolved_local_ids == ("2",)


def test_multiple_inputs_and_categories_preserve_order_ambiguity():
    est = estimator()
    est.observe_timing(TimingEventEvidence(
        local_id="1", observed_at_s=0.9, due_lower_s=0.971, due_upper_s=1.001,
        status=TimingEventStatus.CONFIRMED, reason="wide HUD interval",
    ))
    est.observe_timing(TimingEventEvidence(
        local_id="2", observed_at_s=0.9, due_lower_s=0.979, due_upper_s=1.009,
        status=TimingEventStatus.CONFIRMED, reason="wide HUD interval",
    ))
    est.observe_keydown(keydown("a", 1, 1.001))
    est.observe_keydown(keydown("b", 2, 1.009))

    result = est.observe_judgement(delta(1.010, perfect=1, late_perfect=1))

    assert result.status is EstimateStatus.AMBIGUOUS
    assert len(result.hypotheses) >= 2
    assert all(h.within_interval_order_known is False for h in result.hypotheses)


def test_reset_or_invalid_feedback_creates_gap_and_does_not_attribute():
    est = estimator()
    est.observe_timing(event(1, 1.000))
    est.observe_keydown(keydown("a", 1, 1.001))
    feedback = JudgementDelta(
        status=JudgementDeltaStatus.RESET,
        observed_at_s=1.01,
        displayed_at_s=1.01,
        reason="visible counters decreased",
    )

    result = est.observe_judgement(feedback)

    assert result.status is EstimateStatus.OBSERVATION_GAP
    assert result.hypotheses == ()
    assert result.identifiable is False


def test_id90_perfect_supports_resynchronization_in_simulator_model():
    est = estimator()
    est.observe_timing(event(88, 5.205882397))
    est.observe_timing(event(89, 5.250000044))
    est.observe_timing(event(90, 5.294117691))
    est.observe_keydown(keydown("down-89", 89, 5.253))
    first = est.observe_judgement(delta(5.260, very_late=1))
    assert first.possible_unresolved_local_ids == ("89",)

    est.observe_keydown(keydown("down-90", 90, 5.319))
    resync = est.observe_judgement(delta(5.320, perfect=1))

    assert resync.status is EstimateStatus.MODEL_IDENTIFIED
    assert resync.aligned_state_possible is True
    assert resync.possible_unresolved_local_ids == ()


def test_public_evidence_contract_contains_no_scorer_truth_fields():
    timing_fields = set(TimingEventEvidence.__dataclass_fields__)
    motor_fields = set(MotorKeyDownEvidence.__dataclass_fields__)

    forbidden = {"target_index", "_used", "_missed", "truth_time_s", "scorer_cursor"}
    assert timing_fields.isdisjoint(forbidden)
    assert motor_fields.isdisjoint(forbidden)


def test_hypothesis_cap_cannot_claim_unique_attribution():
    est = JudgementStateEstimator(
        RepositoryWindowModel(0.025, 0.030, 0.065),
        max_hypotheses=1,
    )
    est.observe_timing(event(1, 1.0))
    est.observe_timing(event(2, 1.0))
    est.observe_keydown(keydown("one", 2, 1.0))
    result = est.observe_judgement(delta(1.01, perfect=1))
    assert result.status is EstimateStatus.AMBIGUOUS
    assert not result.identifiable
    assert result.aligned_state_possible
    assert "2" in result.possible_unresolved_local_ids


def test_huge_counter_delta_abstains_without_expanding_labels():
    est = estimator()
    result = est.observe_judgement(delta(1.01, perfect=100_000))
    assert result.status is EstimateStatus.AMBIGUOUS
    assert not result.identifiable
    assert "truncated" in result.reason


def test_distinct_label_orders_exceeding_budget_abstain():
    est = estimator()
    result = est.observe_judgement(delta(
        1.01, **{f"label_{index}": 1 for index in range(7)}
    ))
    assert result.status is EstimateStatus.AMBIGUOUS
    assert not result.identifiable


def test_large_attribution_product_abstains_without_enumeration():
    est = JudgementStateEstimator(
        RepositoryWindowModel(0.025, 0.030, 0.065),
        max_predecessors=6,
    )
    for index in range(7):
        est.observe_timing(event(index, 1.0))
        est.observe_keydown(keydown(f"k{index}", index, 1.0))
    # 1 * 2 * ... * 7 = 5040 candidate combinations (>4096).
    result = est.observe_judgement(delta(1.01, perfect=7))
    assert result.status is EstimateStatus.AMBIGUOUS
    assert not result.identifiable
    assert result.hypotheses == ()
    assert set(result.possible_unresolved_local_ids) == {
        str(index) for index in range(7)
    }
    assert "truncated" in result.reason
