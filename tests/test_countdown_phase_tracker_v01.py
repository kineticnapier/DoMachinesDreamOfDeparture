from __future__ import annotations

import pytest

from dmdod.features.countdown_hud_v3 import (
    CountdownCueKind,
    CountdownStudentObservation,
)
from dmdod.planning.countdown_phase_tracker_v01 import CountdownPhaseTracker


def cue(step: int, kind: CountdownCueKind, digit=None, *, valid=True,
        rendered=None, latency=.0):
    rendered = step * .01 if rendered is None else rendered
    return CountdownStudentObservation(
        step=step, observed_at_s=rendered + latency,
        rendered_at_s=rendered, valid=valid, kind=kind, digit=digit,
    )


def test_two_observed_numeric_onsets_bound_tick_period() -> None:
    tracker = CountdownPhaseTracker()
    tracker.observe(cue(0, CountdownCueKind.READY, rendered=0.00))
    tracker.observe(cue(1, CountdownCueKind.READY, rendered=.25))
    first = tracker.observe(cue(2, CountdownCueKind.NUMBER, 2, rendered=.26))
    tracker.observe(cue(3, CountdownCueKind.NUMBER, 2, rendered=1.25))
    second = tracker.observe(cue(4, CountdownCueKind.NUMBER, 1, rendered=1.26))
    assert first.measured_tick_period_s is None
    assert second.transition_confirmations == 2
    assert second.measured_tick_period_s.lower_s == pytest.approx(.99)
    assert second.measured_tick_period_s.upper_s == pytest.approx(1.01)
    assert second.remaining_transitions_to_go == 1


def test_latency_is_not_subtracted_as_hidden_truth() -> None:
    tracker = CountdownPhaseTracker()
    tracker.observe(cue(0, CountdownCueKind.READY, rendered=0.0, latency=.004))
    tracker.observe(cue(1, CountdownCueKind.READY, rendered=.25, latency=.002))
    tracker.observe(cue(2, CountdownCueKind.NUMBER, 3, rendered=.26, latency=.007))
    tracker.observe(cue(3, CountdownCueKind.NUMBER, 3, rendered=.50, latency=.003))
    update = tracker.observe(
        cue(4, CountdownCueKind.NUMBER, 2, rendered=.51, latency=.009)
    )
    assert update.measured_tick_period_s.lower_s == pytest.approx(.24)
    assert update.measured_tick_period_s.upper_s == pytest.approx(.26)


@pytest.mark.parametrize("bad", [
    cue(1, CountdownCueKind.UNKNOWN, valid=False),
    cue(1, CountdownCueKind.UNKNOWN, valid=False, rendered=0.0, latency=.1),
])
def test_missing_or_stale_observation_forces_abstention(bad) -> None:
    tracker = CountdownPhaseTracker()
    tracker.observe(cue(0, CountdownCueKind.READY))
    failed = tracker.observe(bad)
    recovered = tracker.observe(
        cue(2, CountdownCueKind.NUMBER, 2, rendered=.11)
    )
    assert not failed.valid
    assert not recovered.valid
    assert recovered.status == "invalid_or_stale_frame"
    assert recovered.measured_tick_period_s is None


def test_contradictory_digit_sequence_forces_abstention() -> None:
    tracker = CountdownPhaseTracker()
    tracker.observe(cue(0, CountdownCueKind.READY))
    tracker.observe(cue(1, CountdownCueKind.NUMBER, 4))
    failed = tracker.observe(cue(2, CountdownCueKind.NUMBER, 2))
    assert not failed.valid
    assert failed.status == "contradictory_digit_sequence"


def test_go_requires_observed_digit_one_and_hidden_after_go_is_valid() -> None:
    tracker = CountdownPhaseTracker()
    tracker.observe(cue(0, CountdownCueKind.READY))
    tracker.observe(cue(1, CountdownCueKind.NUMBER, 2))
    tracker.observe(cue(2, CountdownCueKind.NUMBER, 1))
    go = tracker.observe(cue(3, CountdownCueKind.GO))
    hidden = tracker.observe(cue(4, CountdownCueKind.HIDDEN))
    assert go.valid and go.remaining_transitions_to_go == 0
    assert hidden.valid and hidden.status == "hidden"


def test_hidden_idle_and_hidden_to_ready_are_valid() -> None:
    tracker = CountdownPhaseTracker()
    idle = tracker.observe(cue(0, CountdownCueKind.HIDDEN))
    ready = tracker.observe(cue(1, CountdownCueKind.READY))
    assert idle.valid and idle.status == "hidden"
    assert ready.valid and ready.status == "ready"


def test_tracker_rejects_nonsequential_or_backward_observation() -> None:
    tracker = CountdownPhaseTracker()
    with pytest.raises(ValueError, match="consecutive"):
        tracker.observe(cue(1, CountdownCueKind.READY))
    tracker.observe(cue(0, CountdownCueKind.READY, rendered=1.0))
    with pytest.raises(ValueError, match="backward"):
        tracker.observe(cue(1, CountdownCueKind.READY, rendered=.9))
