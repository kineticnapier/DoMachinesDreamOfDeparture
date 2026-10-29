from __future__ import annotations

from dmdod.features.judgement_observer_v01 import (
    JudgementDelta,
    JudgementDeltaStatus,
)
from dmdod.planning.judgement_state_estimator_v01 import (
    MotorKeyDownEvidence,
    RepositoryWindowModel,
    TimingEventEvidence,
    TimingEventStatus,
)
from dmdod.planning.persistent_judgement_state_estimator_v02 import (
    AllocationRule,
    MotorReservationEvidence,
    PersistentEstimateStatus,
    PersistentJudgementStateEstimator,
    ReservationStatus,
)


def event(local_id: int, due: float, observed: float = 0.0) -> TimingEventEvidence:
    return TimingEventEvidence(
        local_id=str(local_id),
        observed_at_s=observed,
        due_lower_s=due,
        due_upper_s=due,
        status=TimingEventStatus.CONFIRMED,
    )


def key(input_id: str, intended: int, at: float) -> MotorKeyDownEvidence:
    return MotorKeyDownEvidence(
        input_id=input_id,
        intended_local_id=str(intended),
        occurred_at_s=at,
        observed_at_s=at,
    )


def delta(at: float, interval: tuple[float, float], **counts: int) -> JudgementDelta:
    return JudgementDelta(
        status=JudgementDeltaStatus.DELTA,
        observed_at_s=at,
        displayed_at_s=at,
        deltas=tuple(counts.items()),
        event_interval_s=interval,
        delta_available=True,
        within_interval_order_known=False,
    )


def unchanged(at: float) -> JudgementDelta:
    return JudgementDelta(
        status=JudgementDeltaStatus.UNCHANGED,
        observed_at_s=at,
        displayed_at_s=at,
        deltas=(("perfect", 0),),
        event_interval_s=(at - 0.01, at),
        delta_available=True,
    )


def estimator(**kwargs) -> PersistentJudgementStateEstimator:
    return PersistentJudgementStateEstimator(
        RepositoryWindowModel(
            perfect_s=0.015,
            early_late_perfect_s=0.030,
            pass_s=0.050,
        ),
        allocation_rule=AllocationRule.COMPATIBLE_ANY,
        **kwargs,
    )


def test_more_than_four_predecessors_retains_unknown_instead_of_false_identity():
    est = estimator(max_predecessors=3)
    for index in range(6):
        est.observe_timing(event(index, 1.0))
    est.observe_keydown(key("k5", 5, 1.0))
    result = est.observe_judgement(delta(1.01, (0.99, 1.01), perfect=1))
    assert result.status is PersistentEstimateStatus.TRUNCATED
    assert not result.search_complete
    assert not result.identifiable
    assert any(row.unknown_search_remainder for row in result.hypotheses)


def test_hypothesis_cap_above_128_is_explicitly_truncated():
    est = estimator(max_predecessors=None, max_hypotheses=128)
    for index in range(140):
        est.observe_timing(event(index, 1.0))
    est.observe_keydown(key("k139", 139, 1.0))
    result = est.observe_judgement(delta(1.01, (0.99, 1.01), perfect=1))
    assert len(result.hypotheses) == 128
    assert result.status is PersistentEstimateStatus.TRUNCATED
    assert not result.identifiable
    assert result.hypotheses[-1].unknown_search_remainder


def test_later_observation_resolves_ambiguity_without_recreating_discarded_history():
    est = estimator(max_predecessors=None)
    est.observe_timing(event(1, 1.00))
    est.observe_timing(event(2, 1.02))
    est.observe_keydown(key("first", 2, 1.01))
    first = est.observe_judgement(delta(1.02, (1.00, 1.02), perfect=1))
    assert len(first.hypotheses) == 2
    assert not first.identifiable

    est.observe_keydown(key("second", 2, 1.04))
    second = est.observe_judgement(delta(1.05, (1.03, 1.05), late_perfect=1))
    assert second.identifiable
    assert second.status is PersistentEstimateStatus.MODEL_IDENTIFIED
    surviving = second.hypotheses[0]
    assert surviving.attributions[0].judged_local_id == "1"
    assert surviving.attributions[1].judged_local_id == "2"

    est.observe_keydown(key("third", 2, 1.06))
    third = est.observe_judgement(delta(1.07, (1.05, 1.07), too_early=1))
    assert all(
        row.attributions[0].judged_local_id == "1"
        for row in third.hypotheses if not row.unknown_search_remainder
    )


def test_same_visible_history_can_retain_multiple_internal_states():
    est = estimator(max_predecessors=None)
    est.observe_timing(event(1, 1.00))
    est.observe_timing(event(2, 1.02))
    est.observe_keydown(key("ambiguous", 2, 1.01))
    result = est.observe_judgement(delta(1.02, (1.00, 1.02), perfect=1))
    assert {row.attributions[0].judged_local_id for row in result.hypotheses} == {"1", "2"}
    assert not result.identifiable


def test_two_keydowns_and_two_judgements_in_one_interval_keep_order_uncertainty():
    est = estimator(max_predecessors=None)
    est.observe_timing(event(1, 1.00))
    est.observe_timing(event(2, 1.02))
    est.observe_keydown(key("a", 1, 1.00))
    est.observe_keydown(key("b", 2, 1.02))
    result = est.observe_judgement(
        delta(1.03, (0.99, 1.03), perfect=1, early_perfect=1)
    )
    assert result.hypotheses
    assert all(
        any(attribution.ambiguous_order for attribution in row.attributions)
        for row in result.hypotheses if not row.unknown_search_remainder
    )


def test_fifo_advances_between_two_keydowns_in_one_visible_interval():
    est = PersistentJudgementStateEstimator(
        RepositoryWindowModel(0.015, 0.030, 0.050),
        allocation_rule=AllocationRule.REPOSITORY_FIFO,
    )
    est.observe_timing(event(1, 1.00))
    est.observe_timing(event(2, 1.02))
    est.observe_keydown(key("a", 1, 1.00))
    est.observe_keydown(key("b", 2, 1.02))
    result = est.observe_judgement(delta(1.03, (0.99, 1.03), perfect=2))
    assert result.identifiable
    assert [
        item.judged_local_id for item in result.hypotheses[0].attributions
    ] == ["1", "2"]


def test_missing_miss_display_and_unchanged_do_not_undo_automatic_resolution():
    est = estimator(max_predecessors=None, automatic_miss_visible=None)
    est.observe_timing(event(1, 1.00))
    advanced = est.advance_time(1.06)
    assert advanced.hypotheses[0].resolutions[0].kind == "automatic_miss"
    result = est.observe_judgement(unchanged(1.07))
    assert result.status is PersistentEstimateStatus.NO_UPDATE
    assert result.hypotheses[0].resolutions[0].kind == "automatic_miss"


def test_counter_reset_preserves_uncertainty_and_never_claims_identification():
    est = estimator(max_predecessors=None)
    est.observe_timing(event(1, 1.0))
    reset = JudgementDelta(
        status=JudgementDeltaStatus.RESET,
        observed_at_s=1.01,
        displayed_at_s=1.01,
        reason="counter decreased",
    )
    result = est.observe_judgement(reset)
    assert result.status is PersistentEstimateStatus.OBSERVATION_GAP
    assert not result.search_complete
    assert not result.identifiable
    assert result.hypotheses[0].unknown_search_remainder


def test_different_window_models_and_non_fifo_rule_remain_separate_analyses():
    narrow = estimator(max_predecessors=None)
    wide = PersistentJudgementStateEstimator(
        RepositoryWindowModel(0.050, 0.060, 0.080),
        allocation_rule=AllocationRule.REPOSITORY_FIFO,
        max_predecessors=None,
    )
    for est in (narrow, wide):
        est.observe_timing(event(1, 1.00))
        est.observe_timing(event(2, 1.04))
        est.observe_keydown(key("k", 2, 1.04))
    non_fifo = narrow.observe_judgement(delta(1.05, (1.03, 1.05), perfect=1))
    fifo = wide.observe_judgement(delta(1.05, (1.03, 1.05), perfect=1))
    assert {row.attributions[0].judged_local_id for row in non_fifo.hypotheses} == {"2"}
    assert {row.attributions[0].judged_local_id for row in fifo.hypotheses} == {"1"}


def test_reservation_cancellation_and_timing_cancellation_are_recorded():
    est = estimator(max_predecessors=None)
    est.observe_timing(event(1, 1.0))
    est.observe_reservation(MotorReservationEvidence(
        reservation_id="r1",
        local_id="1",
        observed_at_s=0.90,
        status=ReservationStatus.RESERVED,
    ))
    est.observe_reservation(MotorReservationEvidence(
        reservation_id="r1",
        local_id="1",
        observed_at_s=0.91,
        status=ReservationStatus.CANCELED,
    ))
    est.observe_timing(TimingEventEvidence(
        local_id="1",
        observed_at_s=0.92,
        due_lower_s=1.0,
        due_upper_s=1.0,
        status=TimingEventStatus.CANCELED,
    ))
    assert est.hypotheses[0].resolutions[0].kind == "cancellation"


def test_eight_tied_keydowns_abstain_before_factorial_search():
    est = estimator()
    for index in range(8):
        est.observe_timing(event(index, 1.0))
    for index in range(8):
        est.observe_keydown(key(f"k{index}", index, 1.0))
    result = est.observe_judgement(delta(1.01, (0.99, 1.01), perfect=8))
    assert result.status is PersistentEstimateStatus.TRUNCATED
    assert not result.identifiable
    assert any(row.unknown_search_remainder for row in result.hypotheses)


def test_huge_visible_delta_is_truncated_before_expansion():
    est = estimator()
    result = est.observe_judgement(delta(1.01, (0.99, 1.01), perfect=100_000))
    assert result.status is PersistentEstimateStatus.TRUNCATED
    assert not result.search_complete
    assert any(row.unknown_search_remainder for row in result.hypotheses)


def test_many_distinct_labels_abstain_before_enumeration():
    est = estimator()
    result = est.observe_judgement(delta(
        1.01, (0.99, 1.01), **{f"label_{index}": 1 for index in range(7)}
    ))
    assert result.status is PersistentEstimateStatus.TRUNCATED
    assert not result.identifiable


def test_untied_compatible_any_product_budget_retains_unknown():
    est = estimator(max_predecessors=None)
    for index in range(7):
        est.observe_timing(event(index, 1.0))
    for index in range(7):
        # Distinct times bypass the tied-input factorial guard; the Cartesian
        # product of candidate predecessor sets is still 7! = 5040.
        est.observe_keydown(key(f"k{index}", index, 1.0 + index * 0.001))
    result = est.observe_judgement(delta(1.02, (0.99, 1.02), perfect=7))
    assert result.status is PersistentEstimateStatus.TRUNCATED
    assert not result.search_complete
    assert not result.identifiable
    assert any(row.unknown_search_remainder for row in result.hypotheses)
    assert any(
        "attribution combination budget exceeded" in gap
        for row in result.hypotheses for gap in row.observation_gaps
    )
