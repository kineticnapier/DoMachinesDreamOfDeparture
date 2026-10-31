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


def test_visible_reset_restarts_capture_clock_and_discards_old_notifications() -> None:
    renderer = VisibleCounterRenderer(_contract())
    renderer.notify(StockJudgementNotification(9.9, "perfect"))
    assert dict(renderer.capture(10.0).counters)["perfect"] == 1
    renderer.notify(StockJudgementNotification(10.1, "perfect"))
    renderer.reset_visible_counters()

    first = renderer.capture(0.0)
    assert dict(first.counters) == {
        "perfect": 0, "very_late": 0, "fail_miss": 0,
    }
    renderer.notify(StockJudgementNotification(0.05, "perfect"))
    assert dict(renderer.capture(0.1).counters)["perfect"] == 1

    with pytest.raises(ValueError, match="capture timestamp moved backward"):
        renderer.capture(0.09)


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


class _StringifiesToPerfect:
    def __str__(self) -> str:
        return "perfect"


@pytest.mark.parametrize("invalid_category", [1, True, _StringifiesToPerfect()])
def test_visible_notify_rejects_nonstring_categories_without_queuing(invalid_category) -> None:
    renderer = VisibleCounterRenderer(_contract(fail_miss=False))
    # An object whose __str__ returns "perfect" previously passed notify(),
    # but capture() then looked up the original object and raised KeyError.
    with pytest.raises(ValueError, match="judgement category must be a string"):
        renderer.notify(StockJudgementNotification(0.05, invalid_category))
    assert dict(renderer.capture(0.10).counters)["perfect"] == 0


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


def test_visible_pending_heap_preserves_out_of_order_delayed_counts() -> None:
    renderer = VisibleCounterRenderer(_contract(fail_miss=False), display_latency_s=0.01)
    assert renderer.notify(StockJudgementNotification(0.30, "perfect", 2)).accepted
    assert renderer.notify(StockJudgementNotification(0.20, "very_late", 3)).accepted
    assert renderer.notify(StockJudgementNotification(0.20, "perfect", 1)).accepted

    assert dict(renderer.capture(0.20).counters) == {"perfect": 0, "very_late": 0}
    assert dict(renderer.capture(0.21).counters) == {"perfect": 1, "very_late": 3}
    assert dict(renderer.capture(0.29).counters) == {"perfect": 1, "very_late": 3}
    assert dict(renderer.capture(0.31).counters) == {"perfect": 3, "very_late": 3}
    assert dict(renderer.capture(0.31).counters) == {"perfect": 3, "very_late": 3}


def test_visible_pending_heap_preserves_one_picosecond_boundary() -> None:
    renderer = VisibleCounterRenderer(_contract(fail_miss=False))
    renderer.notify(StockJudgementNotification(0.2 + 0.5e-12, "perfect"))
    renderer.notify(StockJudgementNotification(0.2 + 2.0e-12, "very_late"))
    assert dict(renderer.capture(0.2).counters) == {"perfect": 1, "very_late": 0}
    assert dict(renderer.capture(0.21).counters) == {"perfect": 1, "very_late": 1}
    with pytest.raises(ValueError, match="capture timestamp moved backward"):
        renderer.capture(0.20)


def test_visible_pending_heap_large_batch_and_reset() -> None:
    renderer = VisibleCounterRenderer(_contract(fail_miss=False))
    expected = {"perfect": 0, "very_late": 0}
    notifications = []
    for index in range(2500):
        category = "perfect" if index % 2 == 0 else "very_late"
        count = 1 if category == "perfect" else 2
        timestamp = 0.2 + (index % 17) * 0.01
        notifications.append((timestamp, category, count))
        assert renderer.notify(
            StockJudgementNotification(timestamp, category, count)
        ).accepted

    for displayed in (0.1, 0.25, 0.35, 0.5):
        expected = {
            category: sum(
                count for timestamp, label, count in notifications
                if label == category and timestamp <= displayed + 1e-12
            )
            for category in expected
        }
        assert dict(renderer.capture(displayed).counters) == expected
    renderer.reset_visible_counters()
    assert dict(renderer.capture(0.0).counters) == {"perfect": 0, "very_late": 0}
    renderer.notify(StockJudgementNotification(0.01, "perfect"))
    assert dict(renderer.capture(0.02).counters) == {"perfect": 1, "very_late": 0}


@pytest.mark.parametrize("invalid_time", [True, False, "0.1", float("nan"), float("inf"), float("-inf"), 10**1000])
def test_visible_capture_rejects_malformed_timestamps_without_consuming_pending(invalid_time) -> None:
    renderer = VisibleCounterRenderer(_contract(fail_miss=False))
    renderer.notify(StockJudgementNotification(0.05, "perfect"))
    with pytest.raises(ValueError, match="display timestamp must be a finite number"):
        renderer.capture(invalid_time)
    assert dict(renderer.capture(0.1).counters)["perfect"] == 1


@pytest.mark.parametrize("invalid_time", [True, False, "0.05", float("nan"), float("inf"), float("-inf"), 10**1000])
def test_visible_notify_rejects_malformed_timestamps_without_queuing(invalid_time) -> None:
    renderer = VisibleCounterRenderer(_contract(fail_miss=False))
    with pytest.raises(ValueError, match="judgement occurrence time must be a finite number"):
        renderer.notify(StockJudgementNotification(invalid_time, "perfect"))
    assert dict(renderer.capture(0.1).counters)["perfect"] == 0


@pytest.mark.parametrize("invalid_time", [True, "0.01", float("nan"), float("inf"), 10**1000])
@pytest.mark.parametrize("parameter", ["display_latency_s", "capture_latency_s"])
def test_visible_renderer_rejects_malformed_latency_parameters(parameter, invalid_time) -> None:
    with pytest.raises(ValueError, match="latency must be a finite number"):
        VisibleCounterRenderer(_contract(), **{parameter: invalid_time})


def test_visible_renderer_preserves_valid_integer_timestamps() -> None:
    renderer = VisibleCounterRenderer(_contract(fail_miss=False), display_latency_s=0, capture_latency_s=0)
    assert renderer.notify(StockJudgementNotification(1, "perfect")).accepted
    assert dict(renderer.capture(1).counters)["perfect"] == 1


@pytest.mark.parametrize("displayed_at_s", [1e308, float.fromhex("0x1.fffffffffffffp+1023")])
def test_visible_capture_rejects_overflowed_observation_before_consuming_pending(displayed_at_s) -> None:
    renderer = VisibleCounterRenderer(_contract(fail_miss=False), capture_latency_s=1e308)
    renderer.notify(StockJudgementNotification(0.05, "perfect"))
    with pytest.raises(ValueError, match="observation timestamp must be finite"):
        renderer.capture(displayed_at_s)

    # Failed capture must not advance the episode clock or pop a pending hit.
    frame = renderer.capture(0.1)
    assert dict(frame.counters)["perfect"] == 1
    assert frame.observed_at_s == 1e308


def test_visible_capture_keeps_large_but_finite_observation_timestamps() -> None:
    renderer = VisibleCounterRenderer(_contract(fail_miss=False))
    renderer.notify(StockJudgementNotification(0.05, "perfect"))
    frame = renderer.capture(1e308)
    assert frame.observed_at_s == 1e308
    assert dict(frame.counters)["perfect"] == 1


@pytest.mark.parametrize(
    ("labels", "mapping", "evidence", "message"),
    [
        ([1], {1: "perfect"}, {1: CounterVisibilityEvidence.USER_CONFIRMED_VISIBLE}, "counter labels"),
        ([""], {"perfect": ""}, {"perfect": CounterVisibilityEvidence.USER_CONFIRMED_VISIBLE}, "counter labels"),
        (["perfect"], {1: "perfect"}, {1: CounterVisibilityEvidence.USER_CONFIRMED_VISIBLE}, "counter mappings"),
        (["perfect"], {"perfect": 1}, {"perfect": CounterVisibilityEvidence.USER_CONFIRMED_VISIBLE}, "counter mappings"),
        (["perfect"], {"": "perfect"}, {"": CounterVisibilityEvidence.USER_CONFIRMED_VISIBLE}, "counter mappings"),
        (["perfect"], {"perfect": "perfect"}, {1: CounterVisibilityEvidence.USER_CONFIRMED_VISIBLE}, "visibility evidence keys"),
        (["perfect"], {"perfect": "perfect"}, {"perfect": "user_confirmed_visible"}, "visibility evidence must use"),
    ],
)
def test_visible_contract_rejects_coerced_identifiers_and_evidence(
    labels, mapping, evidence, message
) -> None:
    # Previously str() silently accepted non-string labels and categories;
    # string-valued evidence could also silently disable a visible category.
    with pytest.raises(ValueError, match=message):
        VisibleCounterContract.build(
            labels=labels, category_to_label=mapping, evidence=evidence
        )


def test_visible_contract_preserves_valid_string_identifiers_and_enum_evidence() -> None:
    contract = VisibleCounterContract.build(
        labels=["perfect"],
        category_to_label={"perfect": "perfect"},
        evidence={"perfect": CounterVisibilityEvidence.USER_CONFIRMED_VISIBLE},
    )
    renderer = VisibleCounterRenderer(contract)
    assert renderer.notify(StockJudgementNotification(0.05, "perfect")).accepted
    assert dict(renderer.capture(0.1).counters) == {"perfect": 1}


def test_visible_renderer_reuses_contract_mapping_across_events_and_captures(monkeypatch) -> None:
    # The contract is immutable; resolving its mapping once is sufficient.
    # Count actual property reads instead of relying on timing measurements.
    original_mapping = VisibleCounterContract.mapping.fget
    assert original_mapping is not None
    mapping_reads = []

    def tracked_mapping(contract):
        mapping_reads.append(1)
        return original_mapping(contract)

    monkeypatch.setattr(VisibleCounterContract, "mapping", property(tracked_mapping))
    renderer = VisibleCounterRenderer(_contract(fail_miss=False))
    assert len(mapping_reads) == 1
    assert renderer.notify(StockJudgementNotification(0.05, "perfect", 2)).accepted
    assert renderer.notify(StockJudgementNotification(0.06, "very_late", 3)).accepted
    assert not renderer.notify(StockJudgementNotification(0.07, "unknown")).accepted
    assert dict(renderer.capture(0.05).counters) == {"perfect": 2, "very_late": 0}
    assert dict(renderer.capture(0.10).counters) == {"perfect": 2, "very_late": 3}
    renderer.reset_visible_counters()
    assert dict(renderer.capture(0.0).counters) == {"perfect": 0, "very_late": 0}
    assert renderer.notify(StockJudgementNotification(0.01, "perfect")).accepted
    assert dict(renderer.capture(0.02).counters) == {"perfect": 1, "very_late": 0}
    assert len(mapping_reads) == 1


@pytest.mark.parametrize(
    "label_values",
    [(), ("perfect", "perfect"), (1,), ("",)],
)
def test_visible_contract_rejects_invalid_one_shot_label_iterables(label_values) -> None:
    # Validation must inspect the same labels that are stored in the contract.
    with pytest.raises(ValueError, match="counter labels"):
        VisibleCounterContract.build(
            labels=(label for label in label_values),
            category_to_label={"perfect": "perfect"},
            evidence={"perfect": CounterVisibilityEvidence.USER_CONFIRMED_VISIBLE},
        )


def test_visible_contract_rejects_scalar_string_as_label_sequence() -> None:
    # Without this check, "perfect" becomes seven separate HUD labels.
    with pytest.raises(ValueError, match="counter labels"):
        VisibleCounterContract.build(
            labels="perfect",
            category_to_label={"perfect": "p"},
            evidence={"perfect": CounterVisibilityEvidence.USER_CONFIRMED_VISIBLE},
        )


def test_visible_contract_preserves_valid_one_shot_label_iterable() -> None:
    contract = VisibleCounterContract.build(
        labels=(label for label in ("perfect", "very_late")),
        category_to_label={"perfect": "perfect", "very_late": "very_late"},
        evidence={
            "perfect": CounterVisibilityEvidence.USER_CONFIRMED_VISIBLE,
            "very_late": CounterVisibilityEvidence.USER_CONFIRMED_VISIBLE,
        },
    )
    assert contract.labels == ("perfect", "very_late")
    renderer = VisibleCounterRenderer(contract)
    assert renderer.notify(StockJudgementNotification(0.05, "perfect")).accepted
    assert dict(renderer.capture(0.1).counters) == {"perfect": 1, "very_late": 0}
