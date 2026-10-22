from __future__ import annotations

from dmdod.features.judgement_observer_v01 import JudgementDelta, JudgementDeltaStatus
from dmdod.planning.judgement_state_estimator_v01 import MotorKeyDownEvidence, TimingEventEvidence, TimingEventStatus
from dmdod.planning.stock_persistent_judgement_state_estimator_v03 import (
    StockMotorReservationEvidence,
    StockPersistentJudgementStateEstimator,
    StockReservationStatus,
)


def _timing(local_id: str, observed: float, due: float, status: TimingEventStatus = TimingEventStatus.CONFIRMED) -> TimingEventEvidence:
    return TimingEventEvidence(local_id, observed, due, due, status)


def _delta(at: float, **counts: int) -> JudgementDelta:
    return JudgementDelta(JudgementDeltaStatus.DELTA, at, at - 0.01, tuple(counts.items()), (at - 0.02, at - 0.01), True, False, "fixture")


def test_visible_automatic_transition_then_keydown_tracks_sequential_floor() -> None:
    estimator = StockPersistentJudgementStateEstimator(automatic_failmiss_visible=True)
    estimator.observe_timing(_timing("36", 3.18, 3.184, TimingEventStatus.ABSTAINED))
    estimator.observe_timing(_timing("37", 3.19, 3.221, TimingEventStatus.ABSTAINED))
    estimator.observe_timing(_timing("38", 3.20, 3.265))
    auto = estimator.observe_judgement(_delta(3.26, fail_miss=1))
    assert auto.possible_current_local_ids == ("37",)
    estimator.observe_keydown(MotorKeyDownEvidence("in38", "38", 3.261, 3.261))
    result = estimator.observe_judgement(_delta(3.28, very_late=1))
    assert result.identifiable
    assert result.possible_current_local_ids == ("38",)
    assert result.hypotheses[0].attributions[-1].judged_local_id == "37"


def test_unknown_failmiss_visibility_prevents_false_unique_floor() -> None:
    estimator = StockPersistentJudgementStateEstimator(automatic_failmiss_visible=None)
    for local_id, observed, due in (("36", 3.18, 3.184), ("37", 3.19, 3.221), ("38", 3.20, 3.265)):
        estimator.observe_timing(_timing(local_id, observed, due))
    estimator.observe_keydown(MotorKeyDownEvidence("in38", "38", 3.261, 3.261))
    result = estimator.observe_judgement(_delta(3.28, very_late=1))
    assert not result.identifiable
    assert {"37", "38"}.issubset(set(result.possible_current_local_ids))
    assert any(row.unknown_search_remainder for row in result.hypotheses)


def test_too_late_does_not_advance_stock_floor() -> None:
    estimator = StockPersistentJudgementStateEstimator(automatic_failmiss_visible=True)
    estimator.observe_timing(_timing("36", 1.0, 1.1))
    estimator.observe_keydown(MotorKeyDownEvidence("bad", "36", 1.3, 1.3))
    result = estimator.observe_judgement(_delta(1.31, too_late=1))
    assert result.possible_current_local_ids == ("36",)
    assert not result.hypotheses[0].attributions[-1].advances_floor


def test_counter_reset_preserves_unknown_instead_of_repairing_state() -> None:
    estimator = StockPersistentJudgementStateEstimator(automatic_failmiss_visible=True)
    estimator.observe_timing(_timing("1", 0.0, 1.0))
    reset = JudgementDelta(JudgementDeltaStatus.RESET, 1.1, 1.1, reason="restart")
    result = estimator.observe_judgement(reset)
    assert not result.identifiable
    assert result.hypotheses[0].unknown_search_remainder


def test_estimator_inputs_and_outputs_contain_no_oracle_target_index() -> None:
    estimator = StockPersistentJudgementStateEstimator(automatic_failmiss_visible=None)
    estimator.observe_timing(_timing("local", 0.0, 1.0))
    result = estimator.advance_time(0.5)
    assert not hasattr(result, "target_index")
    assert not hasattr(result.hypotheses[0], "true_floor")


def test_reservation_history_cannot_rebind_or_resolve_a_floor() -> None:
    estimator = StockPersistentJudgementStateEstimator(automatic_failmiss_visible=None)
    estimator.observe_timing(_timing("1", 0.0, 1.0))
    estimator.observe_timing(_timing("2", 0.1, 2.0))
    estimator.observe_reservation(StockMotorReservationEvidence("r", "1", 0.2, StockReservationStatus.RESERVED))
    estimator.observe_reservation(StockMotorReservationEvidence("r", "1", 0.3, StockReservationStatus.CANCELED))
    assert estimator.advance_time(0.4).possible_current_local_ids == ("1",)
    try:
        estimator.observe_reservation(StockMotorReservationEvidence("r", "2", 0.5, StockReservationStatus.RESERVED))
    except ValueError:
        pass
    else:
        raise AssertionError("reservation rebind must be rejected")
