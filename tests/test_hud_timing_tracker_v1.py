from __future__ import annotations

import pytest
from dmdod.planning.hud_timing_tracker_v1 import CausalTimingTracker, forecast_visible_window
from dmdod.training import prestige1_autoplay_future_events_audit as legacy
from dmdod.training import prestige1_autoplay_high_bpm_simulator_audit as sim


def _fixture_hud(bpm=20000., reverse=False):
    segment, _ = sim._fixture(bpm, reverse=reverse)
    samples, done = sim._capture(segment)
    assert done
    return segment, samples


def test_extended_forecast_equals_frozen_10ms_predictor():
    for bpm in (4000., 20000.):
        _, samples = _fixture_hud(bpm)
        for sample in samples[1::10]:
            actual = forecast_visible_window(sample.hud, horizon_s=.010)
            expected = legacy.forecast(sample.hud)
            assert (actual.floor_offsets, actual.times_s, actual.status) == (expected.floor_offsets, expected.times_s, expected.status)


def test_30ms_forecast_is_never_wrong_when_eligible_on_simple_chart():
    from bisect import bisect_right
    for bpm, reverse in ((4000., False), (8000., True), (20000.,False), (20000.,True)):
        segment, samples = _fixture_hud(bpm, reverse)
        times = tuple(float(t.episode_time_s) for t in segment.targets)
        eligible = multi = 0
        for sample in samples[1::10]:
            f = forecast_visible_window(sample.hud, horizon_s=.030)
            if not f.eligible:
                continue
            eligible += 1
            lo = bisect_right(times, sample.seconds+1e-7)
            hi = bisect_right(times, sample.seconds+.030+1e-7)
            expected = tuple(t-sample.seconds for t in times[lo:hi])
            if len(expected) >= 2:
                multi += 1
            assert len(expected) == len(f.times_s), (bpm,reverse,sample.seconds,expected,f)
            assert all(abs(a-b) <= 5e-5 for a,b in zip(expected,f.times_s))
        assert eligible > 0
        assert multi > 0


def test_tracker_deduplicates_overlapping_real_predictions():
    _, samples = _fixture_hud(4000.)
    tracker = CausalTimingTracker()
    reused = 0
    for index, sample in enumerate(samples[::10]):
        result = tracker.observe(index, sample.hud)
        assert len({x.local_id for x in result.pending}) == len(result.pending)
        assert all(a.due_s < b.due_s for a,b in zip(result.pending,result.pending[1:]))
        reused += len(result.reused_ids)
    assert reused > 0


def test_tracker_preserves_abstention_without_hallucinated_event():
    forecasts = iter([
        legacy.Forecast((1,2),(.015,.027),"eligible"),
        legacy.Forecast((),(),"special_event"),
        legacy.Forecast((1,),(.007,),"eligible"),
        legacy.Forecast((),(),"eligible"),
    ])
    tracker=CausalTimingTracker(predictor=lambda _: next(forecasts))
    r0=tracker.observe(0,())
    assert r0.new_ids==(0,1)
    r1=tracker.observe(1,())
    assert not r1.new_ids and not any(x.confirmed_now for x in r1.pending)
    r2=tracker.observe(2,())
    assert {x.local_id for x in r2.elapsed}=={0}
    assert r2.reused_ids==(1,)
    r3=tracker.observe(3,())
    assert [x.local_id for x in r3.elapsed]==[1]
    assert not r3.pending


def test_invalid_stream_and_corrupt_hud_fail_closed():
    tracker=CausalTimingTracker()
    with pytest.raises(ValueError):
        tracker.observe(1,())
    tracker.observe(0,(0.,))
    with pytest.raises(ValueError):
        tracker.observe(0,())
    assert forecast_visible_window((0.,),horizon_s=.030).status == "invalid_hud_dimension"


@pytest.mark.parametrize("bpm,reverse",[
    (6000.,False),(6000.,True),(12000.,False),(12000.,True),
    (20000.,False),(20000.,True),(4000.,False),
])
def test_boundary_locked_notes_are_recovered_without_duplicate_predictions(bpm,reverse):
    segment, samples = _fixture_hud(bpm,reverse)
    tracker = CausalTimingTracker()
    seen = set()
    elapsed = []
    for i,sample in enumerate(samples[::10]):
        update=tracker.observe(i,sample.hud)
        for ev in update.elapsed:
            assert ev.local_id not in seen
            seen.add(ev.local_id)
            elapsed.append(ev)
    true_times = [float(x.episode_time_s) for x in segment.targets]
    assert elapsed
    for ev in elapsed:
        assert any(abs(t-ev.due_s)<5e-5 for t in true_times)
    if bpm in (6000.,12000.):
        assert len(elapsed) >= (25 if bpm==6000. else 40)


def test_predictor_does_not_import_training_modules():
    import ast
    import inspect
    from dmdod.planning import hud_timing_tracker_v1 as module
    tree = ast.parse(inspect.getsource(module))
    assert all(not (node.module or "").startswith("dmdod.training")
               for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))


def test_malformed_forecast_does_not_create_phantom_events():
    from dmdod.planning.hud_timing_tracker_v1 import WindowForecast
    tracker = CausalTimingTracker(predictor=lambda _: WindowForecast((1, 2), (.015, .015), "eligible"))
    with pytest.raises(ValueError):
        tracker.observe(0, ())
