from __future__ import annotations

import pytest

from dmdod.features.judgement_observer_v01 import JudgementDeltaStatus, JudgementObserver
from dmdod.features.stock_visible_judgement_feedback_v01 import (
    CounterVisibilityEvidence,
    StockJudgementNotification,
    VisibleCounterContract,
    VisibleCounterRenderer,
)


def _contract(*, fail_miss: bool = True) -> VisibleCounterContract:
    mapping = {"perfect": "perfect", "very_late": "very_late"}
    evidence = {
        "perfect": CounterVisibilityEvidence.USER_CONFIRMED_VISIBLE,
        "very_late": CounterVisibilityEvidence.USER_CONFIRMED_VISIBLE,
    }
    labels = ["perfect", "very_late"]
    if fail_miss:
        labels.append("fail_miss")
        mapping["fail_miss"] = "fail_miss"
        evidence["fail_miss"] = CounterVisibilityEvidence.TEST_FIXTURE_ONLY
    return VisibleCounterContract.build(labels=labels, category_to_label=mapping, evidence=evidence)


def test_visible_pipeline_batches_multiple_judgements_without_order_claim() -> None:
    renderer = VisibleCounterRenderer(_contract(), capture_latency_s=0.004)
    observer = JudgementObserver(renderer.contract.labels)
    assert observer.observe(renderer.capture(1.0)).status is JudgementDeltaStatus.BASELINE
    renderer.notify(StockJudgementNotification(1.003, "perfect"))
    renderer.notify(StockJudgementNotification(1.007, "very_late", count=2))
    delta = observer.observe(renderer.capture(1.010))
    assert delta.delta_map == {"perfect": 1, "very_late": 2, "fail_miss": 0}
    assert delta.total_increment == 3
    assert delta.observed_at_s == pytest.approx(1.014)
    assert not delta.within_interval_order_known


def test_display_latency_and_missing_capture_preserve_a_causal_interval() -> None:
    renderer = VisibleCounterRenderer(_contract(), display_latency_s=0.010)
    observer = JudgementObserver(renderer.contract.labels)
    observer.observe(renderer.capture(2.0))
    renderer.notify(StockJudgementNotification(2.005, "perfect"))
    assert observer.observe(renderer.capture(2.010)).status is JudgementDeltaStatus.UNCHANGED
    # The 2.020 capture is deliberately absent.  A later cumulative capture
    # preserves the count while widening, rather than fabricating, event time.
    delta = observer.observe(renderer.capture(2.030))
    assert delta.delta_map["perfect"] == 1
    assert delta.event_interval_s == pytest.approx((2.010, 2.030))


def test_visible_reset_is_observed_as_reset_not_negative_judgement() -> None:
    renderer = VisibleCounterRenderer(_contract())
    observer = JudgementObserver(renderer.contract.labels)
    observer.observe(renderer.capture(0.0))
    renderer.notify(StockJudgementNotification(0.1, "perfect"))
    assert observer.observe(renderer.capture(0.1)).total_increment == 1
    renderer.reset_visible_counters()
    reset = observer.observe(renderer.capture(0.2))
    assert reset.status is JudgementDeltaStatus.RESET
    assert reset.total_increment == 0


def test_unverified_automatic_failmiss_can_be_kept_out_of_visible_contract() -> None:
    renderer = VisibleCounterRenderer(_contract(fail_miss=False))
    diagnostic = renderer.notify(StockJudgementNotification(3.25, "fail_miss"))
    assert not diagnostic.accepted
    assert renderer.capture(3.26).counters == (("perfect", 0), ("very_late", 0))


def test_frames_do_not_expose_privileged_floor_or_target_fields() -> None:
    renderer = VisibleCounterRenderer(_contract())
    frame = renderer.capture(1.0)
    assert not hasattr(frame, "target_index")
    assert not hasattr(frame, "current_floor")
    assert not hasattr(frame, "target_time_s")


def test_contract_rejects_unmapped_labels_and_bad_events() -> None:
    with pytest.raises(ValueError):
        VisibleCounterContract.build(
            labels=("perfect",),
            category_to_label={"perfect": "missing"},
            evidence={"perfect": CounterVisibilityEvidence.TEST_FIXTURE_ONLY},
        )
    renderer = VisibleCounterRenderer(_contract())
    with pytest.raises(ValueError):
        renderer.notify(StockJudgementNotification(0.0, "perfect", count=0))


def test_unconfirmed_categories_never_increment_visible_counters() -> None:
    renderer = VisibleCounterRenderer(_contract(fail_miss=True))
    diagnostic = renderer.notify(StockJudgementNotification(0.05, "fail_miss"))
    assert not diagnostic.accepted
    assert "user-confirmed" in diagnostic.reason
    assert dict(renderer.capture(0.10).counters)["fail_miss"] == 0

    internal = VisibleCounterContract.build(
        labels=("internal",),
        category_to_label={"internal": "internal"},
        evidence={"internal": CounterVisibilityEvidence.DLL_INTERNAL_COUNT_ONLY},
    )
    internal_renderer = VisibleCounterRenderer(internal)
    assert not internal_renderer.notify(
        StockJudgementNotification(0.05, "internal")
    ).accepted
    assert dict(internal_renderer.capture(0.10).counters)["internal"] == 0


def test_user_confirmed_category_still_reaches_visible_counter() -> None:
    renderer = VisibleCounterRenderer(_contract(fail_miss=True))
    assert renderer.notify(
        StockJudgementNotification(0.05, "perfect", count=2)
    ).accepted
    assert dict(renderer.capture(0.10).counters)["perfect"] == 2
