from __future__ import annotations

import ast
import inspect
import json

import pytest

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.n_key import (
    DiagnosticHudNKeyRealChartMotorEnv,
    encode_n_key_hud_real_chart_observation,
)
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG as FLOORS
from dmdod.motor.n_key import NKeyAction
from dmdod.planning.hud_timing_tracker_transition_v14 import (
    CURRENT_SLOT,
    FLOOR_DIM,
    HUD_OFFSET,
    MOTOR_FEATURES,
    SLOT_START,
    forecast_visible_window,
    visible_slot,
)
from dmdod.planning.hud_timing_tracker_twirl_v17 import (
    TwirlRecoveryLifecycleCausalTimingTracker,
    normalize_visible_twirl_transition,
)
from dmdod.planning.hud_v3_composite_forecast_source_v30 import (
    HUDV3CompositeForecastSource,
    LIFECYCLE_V14,
    LIFECYCLE_V15,
)


def _capture_twirl_chart():
    raw = json.dumps({
        "angleData": [0.0, 0.0, 30.0, 180.0, 0.0, 180.0, 0.0],
        "settings": {"bpm": 1200.0, "pitch": 100.0, "countdownTicks": 0},
        "actions": [{"floor": 2, "eventType": "Twirl"}],
    })
    segment = build_playable_segment(
        compile_adofai(parse_adofai_text(raw))
    )
    env = DiagnosticHudNKeyRealChartMotorEnv(
        segment,
        key_count=8,
        control_dt_s=.010,
        physics_dt_s=.001,
        behind_floors=FLOORS.behind_floors,
        ahead_floors=FLOORS.ahead_floors,
        tail_s=.040,
    )
    observation = env.reset()
    idle = NKeyAction((0.0,) * 8)
    rows = []
    for _ in range(200):
        rows.append(tuple(encode_n_key_hud_real_chart_observation(observation)))
        transition = env.step(idle)
        observation = transition.observation
        if transition.done:
            break
    return segment, rows


def _future_twirl_pair(rows):
    for previous, hud in zip(rows, rows[1:]):
        slots = tuple(
            slot for slot in range(
                CURRENT_SLOT + 1, CURRENT_SLOT + FLOORS.ahead_floors + 1
            )
            if visible_slot(hud, slot)[-1] > .5
            and visible_slot(hud, slot)[8] > .5
        )
        if slots:
            result = normalize_visible_twirl_transition(hud, previous)
            if result.recovered_floor_slots:
                return previous, hud, slots[0]
    raise AssertionError("fixture did not expose a causally recoverable Twirl")


def test_visible_twirl_is_normalized_only_after_direction_observation():
    _, rows = _capture_twirl_chart()
    previous, hud, slot = _future_twirl_pair(rows)
    missing = normalize_visible_twirl_transition(hud, None)
    normalized = normalize_visible_twirl_transition(hud, previous)

    assert missing.recovered_floor_slots == ()
    assert missing.reason == "twirl_no_motion_history"
    assert normalized.recovered_floor_slots == (slot,)
    assert normalized.observed_direction in (-1, 1)
    assert visible_slot(hud, slot)[8] == 1.0
    assert visible_slot(normalized.hud, slot)[8] == 0.0
    assert visible_slot(normalized.hud, slot)[2:4] == visible_slot(hud, slot)[4:6]
    assert visible_slot(normalized.hud, slot)[4:6] == visible_slot(hud, slot)[2:4]


def test_twirl_marker_counterfactual_is_not_mislabeled_same_observation():
    _, rows = _capture_twirl_chart()
    _, hud, slot = _future_twirl_pair(rows)
    no_marker = list(hud)
    no_marker[SLOT_START + slot * FLOOR_DIM + 8] = 0.0

    differences = tuple(
        index for index, (left, right) in enumerate(zip(hud, no_marker))
        if left != right
    )
    assert differences == (SLOT_START + slot * FLOOR_DIM + 8,)


@pytest.mark.parametrize(
    "unsafe",
    ("mixed_marker", "multiplanet", "motion_alias"),
)
def test_twirl_rule_fails_closed_for_ambiguous_visible_conditions(unsafe):
    _, rows = _capture_twirl_chart()
    previous, hud, slot = _future_twirl_pair(rows)
    changed = list(hud)
    changed_previous = list(previous)
    if unsafe == "mixed_marker":
        changed[SLOT_START + slot * FLOOR_DIM + 9] = 1.0
    elif unsafe == "multiplanet":
        changed[SLOT_START + slot * FLOOR_DIM + 7] = 1.0
    else:
        # 3000 BPM exceeds the <90 degree/10ms anti-alias contract.
        import math
        encoded = math.log2(3000.0 / 120.0)
        changed[HUD_OFFSET] = encoded
        changed_previous[HUD_OFFSET] = encoded

    result = normalize_visible_twirl_transition(
        tuple(changed), tuple(changed_previous)
    )
    assert result.recovered_floor_slots == ()


def test_later_twirl_remains_a_gate_and_is_not_crossed():
    _, rows = _capture_twirl_chart()
    previous, hud, slot = _future_twirl_pair(rows)
    changed = list(hud)
    later = slot + 2
    changed[SLOT_START + later * FLOOR_DIM + 8] = 1.0
    normalized = normalize_visible_twirl_transition(tuple(changed), previous)

    assert normalized.recovered_floor_slots == (slot,)
    assert visible_slot(normalized.hud, slot)[8] == 0.0
    assert visible_slot(normalized.hud, later)[8] == 1.0
    assert visible_slot(normalized.hud, later)[2:6] == visible_slot(hud, later)[2:6]
    forecast = forecast_visible_window(
        normalized.hud, horizon_s=.080, previous_hud=previous
    )
    assert all(
        offset <= later - CURRENT_SLOT for offset in forecast.floor_offsets
    )


def test_twirl_recovery_predicts_post_twirl_event_once_with_44ms_lead():
    segment, rows = _capture_twirl_chart()
    target = next(
        target for target in segment.targets
        if "Twirl" in segment.chart.floors[target.floor_index - 1].event_markers
    )
    tracker = TwirlRecoveryLifecycleCausalTimingTracker(horizon_s=.080)
    ids = set()
    first_seen = None
    all_created = []
    for step, hud in enumerate(rows):
        update = tracker.observe(step, hud)
        all_created.extend(update.new_ids)
        if step * .010 < target.episode_time_s:
            for item in update.pending:
                if abs(item.due_s - target.episode_time_s) <= .00005:
                    ids.add(item.local_id)
                    if first_seen is None:
                        first_seen = step * .010

    assert len(ids) == 1
    assert len(all_created) == len(set(all_created))
    assert first_seen is not None
    assert target.episode_time_s - first_seen >= .044


def test_transformed_history_does_not_break_full_turn_before_twirl():
    # Regression shape from ALL MY FELLAS event 48: a full-turn current floor
    # immediately precedes a visible Twirl. Using raw prior angles in the
    # frozen full-turn consistency check caused full_turn_changed_floor.
    segment, rows = _capture_twirl_chart()
    previous, hud, slot = _future_twirl_pair(rows)
    full_turn_previous = list(previous)
    full_turn_hud = list(hud)
    for frame in (full_turn_previous, full_turn_hud):
        start = SLOT_START + CURRENT_SLOT * FLOOR_DIM
        frame[start + 4:start + 6] = frame[start + 2:start + 4]
    tracker = TwirlRecoveryLifecycleCausalTimingTracker(horizon_s=.080)
    tracker.observe(0, tuple(full_turn_previous))
    update = tracker.observe(1, tuple(full_turn_hud))

    assert update.status != "full_turn_changed_floor"


def test_future_setspeed_and_pause_gates_remain_in_frozen_forecaster():
    _, rows = _capture_twirl_chart()
    previous, hud, slot = _future_twirl_pair(rows)
    for marker_index in (9, 11):
        changed = list(hud)
        changed[SLOT_START + (slot + 1) * FLOOR_DIM + marker_index] = 1.0
        normalized = normalize_visible_twirl_transition(tuple(changed), previous)
        forecast = forecast_visible_window(
            normalized.hud, horizon_s=.080, previous_hud=previous
        )
        # Entry to a special floor can be emitted, but no event after that
        # floor's hidden numeric effect may be emitted.
        special_relative = slot + 1 - CURRENT_SLOT
        assert all(
            offset <= special_relative for offset in forecast.floor_offsets
        )


def test_forecast_source_twirl_opt_in_is_independent_and_v15_only():
    baseline = HUDV3CompositeForecastSource(lifecycle_version=LIFECYCLE_V15)
    twirl = HUDV3CompositeForecastSource(
        lifecycle_version=LIFECYCLE_V15, twirl_recovery=True
    )
    combined = HUDV3CompositeForecastSource(
        lifecycle_version=LIFECYCLE_V15,
        twirl_recovery=True,
        midspin_recovery=True,
    )
    assert baseline.twirl_recovery is False
    assert twirl.twirl_recovery is True and twirl.midspin_recovery is False
    assert combined.twirl_recovery is True and combined.midspin_recovery is True
    with pytest.raises(ValueError, match="requires lifecycle v1.5"):
        HUDV3CompositeForecastSource(
            lifecycle_version=LIFECYCLE_V14, twirl_recovery=True
        )


def test_twirl_module_imports_no_chart_truth_or_evaluator_modules():
    from dmdod.planning import hud_timing_tracker_twirl_v17 as module

    tree = ast.parse(inspect.getsource(module))
    imported = [
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    ]
    assert not any(
        name.startswith(("dmdod.adofai", "dmdod.envs", "dmdod.training"))
        for name in imported
    )
