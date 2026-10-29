from __future__ import annotations

from dataclasses import dataclass
import random

import pytest

from dmdod.planning.due_event_window_v06 import DueEventWindow


@dataclass(frozen=True)
class Event:
    event_index: int
    episode_time_s: float


def legacy_due(events, now, tail, prediction, selected):
    upcoming = tuple(
        event for event in events
        if event.event_index not in selected
        and event.episode_time_s + tail >= now
    )
    return tuple(event for event in upcoming
                 if event.episode_time_s <= now + prediction + 1e-9)


def test_exact_expiry_and_prediction_boundaries_and_deferred_retry():
    events = tuple(Event(i, t) for i, t in enumerate((0., .1, .1, .2, .4)))
    window = DueEventWindow(events)
    selected = {0, 2}
    for now in (0., .05, .1, .15, .2, .3, .4, .5):
        assert window.due(now, preview_tail_s=.1, prediction_s=.05,
                          selected=selected) == legacy_due(
            events, now, .1, .05, selected)
    # A deferred event remains eligible on a later frame.
    assert Event(3, .2) in window.due(.15, preview_tail_s=.1,
                                      prediction_s=.05, selected=set())
    assert Event(3, .2) in window.due(.16, preview_tail_s=.1,
                                      prediction_s=.05, selected=set())


def test_many_sorted_events_random_monotonic_frames_match_old_filter():
    rng = random.Random(301)
    times = sorted(rng.randrange(0, 2000) / 1000 for _ in range(500))
    events = tuple(Event(i, t) for i, t in enumerate(times))
    window = DueEventWindow(events)
    selected = set()
    for frame in range(250):
        now = frame / 100
        if frame % 7 == 0:
            selected.update(rng.sample(range(len(events)), 4))
        expected = legacy_due(events, now, .07, .044, selected)
        assert window.due(now, preview_tail_s=.07, prediction_s=.044,
                          selected=selected) == expected


def test_invalid_unsorted_or_nonfinite_timeline_rejected():
    with pytest.raises(ValueError):
        DueEventWindow((Event(0, 2.), Event(1, 1.)))
    with pytest.raises(ValueError):
        DueEventWindow((Event(0, float("nan")),))
