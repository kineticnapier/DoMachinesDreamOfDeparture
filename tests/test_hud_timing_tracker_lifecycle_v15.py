from __future__ import annotations

from dmdod.planning.hud_timing_tracker_lifecycle_v15 import (
    QuantizedLifecycleCausalTimingTracker,
)
from dmdod.planning.hud_timing_tracker_transition_v14 import (
    CausalTimingTracker,
    WindowForecast,
)


HUD = (0.0,) * 263


def _predict(sequence):
    iterator = iter(sequence)
    return lambda _hud: next(iterator)


def test_quantized_boundary_marks_elapsed_without_changing_due_or_id():
    forecasts = [
        WindowForecast((1,), (0.0100002,), "eligible"),
        WindowForecast((), (), "eligible"),
    ]
    baseline = CausalTimingTracker(predictor=_predict(forecasts))
    first = baseline.observe(0, HUD)
    second = baseline.observe(1, HUD)
    assert first.pending[0].due_s == 0.0100002
    assert not second.elapsed
    assert [item.local_id for item in second.canceled] == [0]

    improved = QuantizedLifecycleCausalTimingTracker(
        predictor=_predict(forecasts)
    )
    first = improved.observe(0, HUD)
    second = improved.observe(1, HUD)
    assert first.pending[0].due_s == 0.0100002
    assert [item.local_id for item in second.elapsed] == [0]
    assert not second.canceled


def test_quantized_boundary_does_not_finalize_outside_existing_match_tolerance():
    forecasts = [
        WindowForecast((1,), (0.0108,), "eligible"),
        WindowForecast((), (), "eligible"),
    ]
    tracker = QuantizedLifecycleCausalTimingTracker(
        predictor=_predict(forecasts)
    )
    tracker.observe(0, HUD)
    update = tracker.observe(1, HUD)
    assert not update.elapsed
    assert [item.local_id for item in update.canceled] == [0]


def test_abstention_cannot_finalize_a_future_boundary_event_early():
    forecasts = [
        WindowForecast((1,), (0.0100002,), "eligible"),
        WindowForecast((), (), "special_event"),
    ]
    tracker = QuantizedLifecycleCausalTimingTracker(
        predictor=_predict(forecasts)
    )
    tracker.observe(0, HUD)
    update = tracker.observe(1, HUD)
    assert not update.elapsed
    assert not update.canceled
    assert [item.local_id for item in update.pending] == [0]
    assert update.pending[0].confirmed_now is False
