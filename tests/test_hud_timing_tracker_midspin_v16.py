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
from dmdod.planning.hud_timing_tracker_midspin_v16 import (
    MidspinRecoveryLifecycleCausalTimingTracker,
    normalize_visible_midspin_transition,
)
from dmdod.planning.hud_timing_tracker_transition_v14 import (
    CURRENT_SLOT,
    FLOOR_DIM,
    SLOT_START,
    forecast_visible_window,
    visible_slot,
)
from dmdod.planning.hud_v3_composite_forecast_source_v30 import (
    HUDV3CompositeForecastSource,
    LIFECYCLE_V14,
    LIFECYCLE_V15,
)


def _capture_midspin_chart():
    raw = json.dumps({
        "angleData": [0.0, 90.0, 999.0, 180.0, 270.0, 0.0, 90.0],
        "settings": {"bpm": 240.0, "pitch": 100.0, "countdownTicks": 0},
        "actions": [],
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
    for _ in range(300):
        rows.append(tuple(encode_n_key_hud_real_chart_observation(observation)))
        transition = env.step(idle)
        observation = transition.observation
        if transition.done:
            break
    return segment, rows


def _first_hud_with_future_midspin(rows, *, blocked=False):
    for hud in rows:
        for slot in range(CURRENT_SLOT + 1, CURRENT_SLOT + FLOORS.ahead_floors):
            if (
                visible_slot(hud, slot)[6] > .5
                and (not blocked or forecast_visible_window(
                    hud, horizon_s=.080
                ).status == "special_future_floor")
            ):
                return hud, slot
    raise AssertionError("fixture did not expose a future Midspin")


def test_visible_midspin_pair_is_opt_in_and_original_hud_is_unchanged():
    _, rows = _capture_midspin_chart()
    hud, slot = _first_hud_with_future_midspin(rows, blocked=True)
    baseline = forecast_visible_window(hud, horizon_s=.080)
    normalized = normalize_visible_midspin_transition(hud)

    assert baseline.status == "special_future_floor"
    assert normalized.recovered_floor_slots == (slot,)
    assert visible_slot(hud, slot)[6] == 1.0
    assert visible_slot(normalized.hud, slot)[6] == 0.0
    assert visible_slot(normalized.hud, slot + 1) == visible_slot(hud, slot + 1)


@pytest.mark.parametrize("unsafe_feature", ("missing_following", "marker", "multiplanet"))
def test_midspin_rule_fails_closed_for_unsafe_visible_patterns(unsafe_feature):
    _, rows = _capture_midspin_chart()
    hud, slot = _first_hud_with_future_midspin(rows)
    changed = list(hud)
    if unsafe_feature == "missing_following":
        start = SLOT_START + (slot + 1) * FLOOR_DIM
        changed[start:start + FLOOR_DIM] = [0.0] * FLOOR_DIM
    elif unsafe_feature == "marker":
        changed[SLOT_START + slot * FLOOR_DIM + 8] = 1.0
    else:
        changed[SLOT_START + slot * FLOOR_DIM + 7] = 1.0
    assert normalize_visible_midspin_transition(tuple(changed)).recovered_floor_slots == ()


def test_recovered_due_matches_playable_target_without_duplicate_id():
    segment, rows = _capture_midspin_chart()
    target = next(
        target for target in segment.targets
        if segment.chart.floors[target.floor_index].previous_midspin
    )
    tracker = MidspinRecoveryLifecycleCausalTimingTracker(horizon_s=.080)
    matching_ids = set()
    all_created = []
    all_retired = []
    for step, hud in enumerate(rows):
        update = tracker.observe(step, hud)
        all_created.extend(update.new_ids)
        all_retired.extend(item.local_id for item in update.elapsed + update.canceled)
        if step * .010 < target.episode_time_s:
            matching_ids.update(
                item.local_id for item in update.pending
                if abs(item.due_s - target.episode_time_s) <= .00005
            )

    assert len(matching_ids) == 1
    assert len(all_created) == len(set(all_created))
    assert len(all_retired) == len(set(all_retired))


def test_forecast_source_requires_explicit_v15_opt_in_and_reset_is_safe():
    baseline = HUDV3CompositeForecastSource(
        lifecycle_version=LIFECYCLE_V15
    )
    candidate = HUDV3CompositeForecastSource(
        lifecycle_version=LIFECYCLE_V15,
        midspin_recovery=True,
    )
    assert baseline.midspin_recovery is False
    assert candidate.midspin_recovery is True
    candidate.reset()
    with pytest.raises(ValueError, match="requires lifecycle v1.5"):
        HUDV3CompositeForecastSource(
            lifecycle_version=LIFECYCLE_V14,
            midspin_recovery=True,
        )


def test_midspin_module_imports_no_chart_truth_or_evaluator_modules():
    from dmdod.planning import hud_timing_tracker_midspin_v16 as module

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
