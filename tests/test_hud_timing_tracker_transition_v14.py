"""Independent causal HUD SetSpeed/Pause research, no Student training."""
from __future__ import annotations

from bisect import bisect_right

import pytest

from dmdod.envs.n_key import (
    DiagnosticHudNKeyRealChartMotorEnv, encode_n_key_hud_real_chart_observation,
)
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG as FLOORS
from dmdod.motor.n_key import NKeyAction
from dmdod.planning.hud_timing_tracker_transition_v14 import (
    CausalTimingTracker, forecast_visible_window, visible_slot, CURRENT_SLOT,
)
from scripts.audit_hud_timing_tracker_transition_v14_matrix import generated


def hud_trace(segment):
    env = DiagnosticHudNKeyRealChartMotorEnv(
        segment, key_count=8, control_dt_s=.01, physics_dt_s=.001,
        behind_floors=FLOORS.behind_floors, ahead_floors=FLOORS.ahead_floors,
        tail_s=.04,
    )
    obs = env.reset()
    idle = NKeyAction((0.,) * 8)
    result = []
    for _ in range(1200):
        result.append(tuple(encode_n_key_hud_real_chart_observation(obs)))
        transition = env.step(idle)
        obs = transition.observation
        if transition.done:
            return result
    raise AssertionError("unbounded chart HUD")


@pytest.mark.parametrize("variant", (
    "speed_double", "speed_half", "speed_offset", "pause_1",
    "pause_half", "twirl_speed", "twirl_pause", "speed_pause",
))
@pytest.mark.parametrize("bpm", (496., 4000.))
def test_special_transition_predictions_match_compiled_events(variant, bpm):
    segment = generated((0., 90., 180., 270.), bpm, variant)
    samples = hud_trace(segment)
    tracker = CausalTimingTracker()
    forecasts = []
    observed = []
    previous = None
    for i, sample in enumerate(samples):
        forecasts.append(forecast_visible_window(sample, previous_hud=previous))
        observed.extend(tracker.observe(i, sample).elapsed)
        previous = sample
    # All oracle labels are consulted *after* all predictions are finished.
    true = tuple(float(t.episode_time_s) for t in segment.targets)
    for i, forecast in enumerate(forecasts):
        if not forecast.eligible:
            continue
        now = i * .01
        covered = forecast.covered_until_s if forecast.covered_until_s is not None else .03
        lo = bisect_right(true, now + .000001)
        hi = bisect_right(true, now + covered - .000001)
        expected = tuple(t - now for t in true[lo:hi])
        predicted = tuple(t for t in forecast.times_s if .000001 < t < covered - .000001)
        assert len(predicted) == len(expected), (variant, bpm, i, forecast)
        assert all(abs(x - y) <= .00005 for x, y in zip(expected, predicted))
    for ev in observed:
        assert any(abs(t - ev.due_s) <= .00005 for t in true)


def test_current_setspeed_never_extrapolates_across_exit():
    chart = generated((0., 90., 180., 270.), 4000., "speed_offset")
    samples = hud_trace(chart)
    forecast = forecast_visible_window(samples[5], previous_hud=samples[4])
    assert forecast.status == "eligible_partial"
    assert len(forecast.times_s) == 1
    assert forecast.covered_until_s == pytest.approx(forecast.times_s[0])


def test_pause_duration_not_in_visible_hud_before_motion_resumes():
    short = generated((0., 180.), 496., "pause_half")
    long = generated((0., 180.), 496., "pause_1")
    short_hud = hud_trace(short)
    long_hud = hud_trace(long)
    # Before the shorter pause finishes, same actions/geometry/BPM means
    # visible HUDs are identical even though subsequent target times differ.
    same_prefix = 0
    for a, b in zip(short_hud, long_hud):
        if max(abs(x - y) for x, y in zip(a, b)) > 1e-8:
            break
        same_prefix += 1
    assert same_prefix > 10
    assert tuple(t.episode_time_s for t in short.targets) != tuple(
        t.episode_time_s for t in long.targets)


def test_pause_stationary_abstains_but_resumed_motion_can_predict():
    chart = generated((0., 180.), 496., "pause_1")
    samples = hud_trace(chart)
    status = []
    for i, sample in enumerate(samples):
        floor = visible_slot(sample, CURRENT_SLOT)
        if floor[11] > .5:
            status.append(forecast_visible_window(
                sample, previous_hud=samples[i - 1] if i > 0 else None).status)
    assert "pause_motion_unobservable" in status
    assert any(s in ("eligible", "eligible_partial") for s in status)


def test_countdown_is_not_mistaken_for_speed_transition():
    import json
    from dmdod.adofai.chart import parse_adofai_text
    from dmdod.adofai.timing import compile_adofai
    from dmdod.adofai.playable import build_playable_segment
    raw = json.dumps({
        "angleData": [0., 90., 180., 270.] * 9,
        "settings": {"bpm": 124., "pitch": 100., "countdownTicks": 3},
        "actions": [{"floor": 1, "eventType": "SetSpeed", "speedType": "Bpm",
                     "beatsPerMinute": 248.}],
    })
    chart = build_playable_segment(compile_adofai(parse_adofai_text(raw)), end_s=3.)
    samples = hud_trace(chart)
    tracker = CausalTimingTracker()
    elapsed = []
    for i, sample in enumerate(samples):
        elapsed.extend(tracker.observe(i, sample).elapsed)
    truth = tuple(t.episode_time_s for t in chart.targets)
    assert truth[0] > 1.
    assert all(any(abs(t - e.due_s) < .00005 for t in truth) for e in elapsed)


def test_causal_source_imports_no_evaluator_or_chart_truth():
    import ast
    import inspect
    from dmdod.planning import hud_timing_tracker_transition_v14 as m
    tree = ast.parse(inspect.getsource(m))
    modules = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any("adofai" in name or "training" in name or "envs" in name for name in modules)
