from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dmdod.planning.gesture_chunk_planner_v08 import GestureChunkPlannerV08
from audit_dmdogo_v09_h012_synthetic_counterexamples import (
    BASE_SEED, DEFAULT_TRIALS, MODES, QUOTA, OBSERVE_CHUNKS,
    generate, selected_seams, select_alternative, original_precise_cost,
    classify, summarize, audit_one,
)
from audit_dmdogo_v09_h04_beam_decisions import _chunk_record


@pytest.mark.parametrize("trial", [0, 1, 2, 3, 5, 13])
def test_generated_events_and_provenance_are_reproducible(trial):
    first, a = generate(trial)
    second, b = generate(trial)
    assert first == second
    assert a == b
    assert a["seed"] == BASE_SEED + 7919*trial
    assert a["mode"] in MODES
    assert len(a["events_sha256"]) == 64
    assert [x.event_index for x in first] == list(range(len(first)))
    assert all(y.due_s > x.due_s for x, y in zip(first, first[1:]))
    assert min(x.chord_size for x in first) == 1
    assert max(x.chord_size for x in first) <= 2


def test_selected_seams_are_fixed_before_observing_any_outcome():
    assert selected_seams(8) == ()
    assert selected_seams(9) == (1, 2)
    assert selected_seams(24) == (1, 7)
    assert all(join + OBSERVE_CHUNKS < 24 for join in selected_seams(24))


def test_candidate_preselection_uses_only_original_prefix_and_join():
    events, spec = generate(1)
    planner = GestureChunkPlannerV08()
    plan = planner.plan(events)
    assert plan.feasible
    chunks = [_chunk_record(c) for c in plan.chunks]
    windows = planner._windows(events)
    original = original_precise_cost(events, plan, windows, chunks)
    assert round(original, 6) == plan.metrics["beam_cost"]
    seams = selected_seams(len(windows))
    assert seams
    picked = select_alternative(events, chunks, windows, seams[0])
    assert picked is not None
    assert picked["window_index"] == seams[0]
    assert picked["nonquick_join_expanded_rank"] >= 1
    assert picked["selection"] in (
        "first_rank_outside_top24",
        "first_nonoriginal_in_original_predecessor",
    )
    assert len(picked["flag_pair"]) == 2
    assert len(picked["nonquick_axes"]) > 0
    assert picked == select_alternative(events, chunks, windows, seams[0])


def test_counterexample_categories_are_not_confused():
    full = {
        "window_index": 1,
        "beam_winner": {
            "final_cost": 9.0,
            "is_from_protected_lineage": True,
            "delta_cost_vs_frozen_v08": -1.0,
        },
    }
    adaptive = {
        "quota_released": True,
        "beam_winner": {
            "final_cost": 10.0,
            "is_from_protected_lineage": False,
            "delta_cost_vs_frozen_v08": 0.0,
        },
    }
    late = {"summary": {"first_tagged_rank_one_window": 9}}
    outcome = classify(full, adaptive, late)
    assert outcome["late_rank1_after_grace"] is True
    assert outcome["released_then_late_rank1"] is True
    assert outcome["rule_regret_after_release"] is True
    assert outcome["missed_improvement_vs_v08"] is True
    assert outcome["first_rank1_offset"] == 8


def test_late_rank1_without_regret_is_not_a_falsifying_example():
    full = {
        "window_index": 1,
        "beam_winner": {
            "final_cost": 11.0,
            "is_from_protected_lineage": False,
            "delta_cost_vs_frozen_v08": 0.0,
        },
    }
    adaptive = {
        "quota_released": True,
        "beam_winner": {
            "final_cost": 11.0,
            "is_from_protected_lineage": False,
            "delta_cost_vs_frozen_v08": 0.0,
        },
    }
    late = {"summary": {"first_tagged_rank_one_window": 7}}
    result = classify(full, adaptive, late)
    assert result["late_rank1_after_grace"]
    assert result["rule_regret"] is False
    assert result["missed_improvement_vs_v08"] is False


def test_trial_smoke_preserves_width_and_source_and_reports_every_target():
    result = audit_one(1)
    assert result["trial_index"] == 1
    assert result["generator"]["mode"] == "accelerate"
    assert result["frozen_plan_feasible"]
    assert len(result["targets"]) + len(result["skipped_targets"]) == len(
        selected_seams(result["chunk_count"]))
    for item in result["targets"]:
        assert item["baseline"]["adaptive_rescue_windows"] <= item[
            "baseline"]["full_quota8_rescue_windows"]
        assert item["target"]["nonquick_join_expanded_rank"] >= 1
        assert item["counterexample"]["quota_released"] in (True, False)
        assert item["diagnostic"]["first_five_chunks"]
    total = summarize([result])
    assert total["synthetic_trials_generated"] == 1
    assert total["targets_evaluated"] == len(result["targets"])


def test_generator_rejects_invalid_index():
    with pytest.raises(ValueError):
        generate(-1)


def test_experiment_protocol_is_fixed():
    assert QUOTA == 8
    assert OBSERVE_CHUNKS == 5
    assert DEFAULT_TRIALS == 32
    assert GestureChunkPlannerV08().beam_width == 24
