from __future__ import annotations

import pytest

from dmdod.features.judgement_observer_v01 import (
    JudgementCounterFrame,
    JudgementDeltaStatus,
    JudgementObserver,
)


LABELS = ("perfect", "very_late", "too_early")


def frame(observed: float, counters: dict[str, int], *, displayed: float | None = None):
    return JudgementCounterFrame.from_mapping(
        observed_at_s=observed,
        displayed_at_s=observed if displayed is None else displayed,
        counters=counters,
    )


def test_first_observation_establishes_a_distinct_baseline():
    observer = JudgementObserver(LABELS)
    result = observer.observe(frame(1.00, {"perfect": 120, "very_late": 8, "too_early": 2}))

    assert result.status is JudgementDeltaStatus.BASELINE
    assert result.delta_available is False
    assert result.deltas == ()
    assert result.event_interval_s is None


def test_unchanged_frame_reports_a_zero_vector_not_a_new_baseline():
    observer = JudgementObserver(LABELS)
    observer.observe(frame(1.00, dict.fromkeys(LABELS, 0)))
    result = observer.observe(frame(1.01, dict.fromkeys(LABELS, 0)))

    assert result.status is JudgementDeltaStatus.UNCHANGED
    assert result.delta_available is True
    assert result.delta_map == dict.fromkeys(LABELS, 0)
    assert result.total_increment == 0
    assert result.event_interval_s == pytest.approx((1.00, 1.01))


def test_single_and_multiple_judgements_preserve_integer_multiplicity():
    observer = JudgementObserver(LABELS)
    observer.observe(frame(1.00, dict.fromkeys(LABELS, 0)))
    single = observer.observe(frame(1.01, {"perfect": 0, "very_late": 1, "too_early": 0}))
    multiple = observer.observe(frame(1.02, {"perfect": 2, "very_late": 2, "too_early": 0}))

    assert single.status is JudgementDeltaStatus.DELTA
    assert single.delta_map == {"perfect": 0, "very_late": 1, "too_early": 0}
    assert multiple.delta_map == {"perfect": 2, "very_late": 1, "too_early": 0}
    assert multiple.total_increment == 3
    assert multiple.within_interval_order_known is False


def test_counter_decrease_is_a_reset_and_never_a_negative_event():
    observer = JudgementObserver(LABELS)
    observer.observe(frame(1.00, {"perfect": 5, "very_late": 2, "too_early": 0}))
    reset = observer.observe(frame(1.01, {"perfect": 0, "very_late": 0, "too_early": 0}))
    resumed = observer.observe(frame(1.02, {"perfect": 1, "very_late": 0, "too_early": 0}))

    assert reset.status is JudgementDeltaStatus.RESET
    assert reset.delta_available is False
    assert reset.deltas == ()
    assert "decreased" in reset.reason
    assert resumed.status is JudgementDeltaStatus.DELTA
    assert resumed.delta_map["perfect"] == 1


@pytest.mark.parametrize(
    "bad_counters",
    [
        {"perfect": -1, "very_late": 0, "too_early": 0},
        {"perfect": 0, "very_late": 0},
        {"perfect": 0, "very_late": 0, "too_early": 0, "unknown": 0},
        {"perfect": True, "very_late": 0, "too_early": 0},
    ],
)
def test_invalid_or_missing_values_fail_closed_and_require_a_new_baseline(bad_counters):
    observer = JudgementObserver(LABELS)
    observer.observe(frame(1.00, dict.fromkeys(LABELS, 0)))
    invalid = observer.observe(frame(1.01, bad_counters))
    baseline = observer.observe(frame(1.02, {"perfect": 9, "very_late": 4, "too_early": 1}))

    assert invalid.status is JudgementDeltaStatus.INVALID
    assert invalid.delta_available is False
    assert baseline.status is JudgementDeltaStatus.BASELINE
    assert baseline.delta_available is False


def test_observation_time_is_separate_from_visible_update_interval():
    observer = JudgementObserver(LABELS)
    observer.observe(frame(2.010, dict.fromkeys(LABELS, 0), displayed=2.004))
    result = observer.observe(
        frame(2.020, {"perfect": 1, "very_late": 0, "too_early": 0}, displayed=2.014)
    )

    assert result.event_interval_s == pytest.approx((2.004, 2.014))
    assert result.observed_at_s == pytest.approx(2.020)
    assert result.available_at_s == pytest.approx(2.020)


def test_noncausal_or_out_of_order_frame_fails_closed():
    observer = JudgementObserver(LABELS)
    observer.observe(frame(1.00, dict.fromkeys(LABELS, 0), displayed=0.99))
    invalid = observer.observe(frame(0.98, dict.fromkeys(LABELS, 0), displayed=0.98))
    future_display = observer.observe(frame(1.10, dict.fromkeys(LABELS, 0), displayed=1.11))

    assert invalid.status is JudgementDeltaStatus.INVALID
    assert future_display.status is JudgementDeltaStatus.INVALID


def test_observer_contract_has_no_scorer_identity_fields():
    frame_fields = set(JudgementCounterFrame.__dataclass_fields__)
    observer = JudgementObserver(LABELS)

    assert frame_fields == {"observed_at_s", "displayed_at_s", "counters"}
    assert not hasattr(observer, "target_index")
    assert not hasattr(observer, "judgement_trace")
    assert not hasattr(observer, "_used")
    assert not hasattr(observer, "_missed")


class _StringifiesToPerfect:
    def __str__(self) -> str:
        return "perfect"


@pytest.mark.parametrize("invalid_label", [1, True, _StringifiesToPerfect(), ""])
def test_observer_rejects_coerced_or_empty_configured_labels(invalid_label) -> None:
    # Stringification used to silently convert 1, True, or an arbitrary
    # object into a purported HUD schema label.
    with pytest.raises(ValueError, match="counter labels must be non-empty strings"):
        JudgementObserver((invalid_label, "very_late"))


@pytest.mark.parametrize("invalid_label", [1, True, _StringifiesToPerfect(), ""])
def test_frame_builder_rejects_coerced_or_empty_counter_labels(invalid_label) -> None:
    with pytest.raises(ValueError, match="counter labels must be non-empty strings"):
        JudgementCounterFrame.from_mapping(
            observed_at_s=0.1,
            displayed_at_s=0.1,
            counters={invalid_label: 1},
        )


@pytest.mark.parametrize("invalid_label", [1, True, _StringifiesToPerfect(), "", []])
def test_direct_counter_frame_rejects_malformed_labels_and_resets_baseline(invalid_label) -> None:
    observer = JudgementObserver(LABELS)
    assert observer.observe(frame(0.0, dict.fromkeys(LABELS, 0))).status is JudgementDeltaStatus.BASELINE
    # A direct dataclass constructor bypasses from_mapping. Even an
    # unhashable label must produce INVALID, not an uncaught TypeError.
    malformed = JudgementCounterFrame(
        observed_at_s=0.1,
        displayed_at_s=0.1,
        counters=((invalid_label, 1), ("very_late", 0), ("too_early", 0)),
    )
    invalid = observer.observe(malformed)
    assert invalid.status is JudgementDeltaStatus.INVALID
    assert not invalid.delta_available
    assert observer.observe(
        frame(0.2, {"perfect": 1, "very_late": 0, "too_early": 0})
    ).status is JudgementDeltaStatus.BASELINE


@pytest.mark.parametrize("bad_timestamp", [True, "0.1", float("nan"), float("inf"), 10**400])
@pytest.mark.parametrize("field", ("observed_at_s", "displayed_at_s"))
def test_frame_builder_rejects_coerced_nonfinite_or_overflowing_timestamps(field, bad_timestamp):
    timestamps = {"observed_at_s": 0.1, "displayed_at_s": 0.1}
    timestamps[field] = bad_timestamp
    with pytest.raises(ValueError, match="counter timestamps must be finite numbers"):
        JudgementCounterFrame.from_mapping(
            **timestamps, counters=dict.fromkeys(LABELS, 0),
        )


@pytest.mark.parametrize("bad_timestamp", [True, "0.1", float("nan"), 10**400])
@pytest.mark.parametrize("field", ("observed_at_s", "displayed_at_s"))
def test_direct_frame_bad_timestamps_invalidate_and_reset_baseline(field, bad_timestamp):
    observer = JudgementObserver(LABELS)
    assert observer.observe(frame(0.0, dict.fromkeys(LABELS, 0))).status is JudgementDeltaStatus.BASELINE
    timestamps = {"observed_at_s": 0.1, "displayed_at_s": 0.1}
    timestamps[field] = bad_timestamp
    malformed = JudgementCounterFrame(
        **timestamps,
        counters=tuple((label, 1) for label in LABELS),
    )
    invalid = observer.observe(malformed)
    assert invalid.status is JudgementDeltaStatus.INVALID
    assert not invalid.delta_available
    assert invalid.deltas == ()
    assert observer.observe(frame(0.2, dict.fromkeys(LABELS, 1))).status is JudgementDeltaStatus.BASELINE


def test_integer_clock_values_remain_valid_for_hud_observation():
    observer = JudgementObserver(LABELS)
    first = JudgementCounterFrame.from_mapping(
        observed_at_s=1, displayed_at_s=1, counters=dict.fromkeys(LABELS, 0),
    )
    second = JudgementCounterFrame.from_mapping(
        observed_at_s=2, displayed_at_s=2, counters={"perfect": 1, "very_late": 0, "too_early": 0},
    )
    assert observer.observe(first).status is JudgementDeltaStatus.BASELINE
    delta = observer.observe(second)
    assert delta.status is JudgementDeltaStatus.DELTA
    assert delta.delta_map["perfect"] == 1
    assert delta.event_interval_s == (1.0, 2.0)
