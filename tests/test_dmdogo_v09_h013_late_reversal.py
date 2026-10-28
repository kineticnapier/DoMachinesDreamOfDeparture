from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_dmdogo_v09_h013_late_reversal as h013
from audit_dmdogo_v09_h012_synthetic_counterexamples import (
    generate, summarize,
)


def fake_trial(idx, *, with_late=False, with_regret=False):
    _, spec = generate(idx)
    c = {
        "late_rank1_after_grace": with_late,
        "released_then_late_rank1": with_late,
        "rule_regret": with_regret,
        "rule_regret_after_release": with_regret,
        "quota_released": True,
        "missed_improvement_vs_v08": with_regret,
        "full_quota8_winner_protected": with_late,
        "adaptive_minus_full_final_cost": 1.0 if with_regret else 0.0,
        "first_rank1_offset": 7 if with_late else None,
    }
    return {
        "trial_index": idx,
        "generator": spec,
        "frozen_plan_feasible": True,
        "frozen_failure_reason": None,
        "chunk_count": 20,
        "skipped_targets": [],
        "targets": ([{
            "window_index": 1,
            "target": {
                "nonquick_join_expanded_rank": 26},
            "counterexample": c,
            "baseline": {
                "full_quota8_displaced_slots": 48,
                "adaptive_displaced_slots": 8,
                "full_quota8_rescue_windows": 6,
                "adaptive_rescue_windows": 1,
            },
            "diagnostic": {
                "first_five_chunks": [],
                "last_five_chunks": [],
            },
        }] if with_late or with_regret else []),
    }


def fake_h012_source():
    trials = [fake_trial(i) for i in range(32)]
    return {
        "format": h013.H012_FORMAT,
        "generator_version": h013.GENERATOR_VERSION,
        "base_seed": h013.BASE_SEED,
        "tested_trial_count": 32,
        "generator_modes": list(h013.MODES),
        "beam_width": h013.BEAM_WIDTH,
        "quota": h013.QUOTA,
        "grace_chunks": h013.OBSERVE_CHUNKS,
        "uses_saved_train_charts": False,
        "physical_motor_executed": False,
        "trials": trials,
        "aggregate": summarize(trials),
    }


def test_complete_h012_source_census_is_verified():
    source = fake_h012_source()
    rows = h013.check_h012_source(source)
    assert len(rows) == 32
    assert [r["trial_index"] for r in rows] == list(range(32))
    source["aggregate"]["synthetic_trials_generated"] = 31
    with pytest.raises(ValueError, match="aggregate"):
        h013.check_h012_source(source)


def test_historical_changed_seed_or_missing_trial_is_rejected():
    source = fake_h012_source()
    source["trials"][2]["generator"]["seed"] += 1
    with pytest.raises(ValueError, match="generator"):
        h013.check_h012_source(source)
    source = fake_h012_source()
    source["trials"].pop()
    with pytest.raises(ValueError, match="census"):
        h013.check_h012_source(source)


def test_historical_late_reversal_selection_is_not_cost_regret():
    trials = [fake_trial(0, with_late=True), fake_trial(1, with_regret=True),
              fake_trial(2)]
    cases = h013._historical_late(trials)
    assert [(c["trial_index"], c["window_index"]) for c in cases] == [(0, 1)]
    assert cases[0]["saved"]["counterexample"]["rule_regret"] is False


def test_independent_cohorts_disjoint_and_nonhistorical():
    assert h013.DISCOVERY_START > 31
    assert h013.EVALUATION_START > h013.DISCOVERY_START+64
    assert not (set(range(h013.DISCOVERY_START,
                          h013.DISCOVERY_START+64))
                & set(range(h013.EVALUATION_START,
                            h013.EVALUATION_START+64)))
    _, a = generate(h013.DISCOVERY_START)
    _, b = generate(h013.EVALUATION_START)
    assert a["seed"] != b["seed"]
    assert a["events_sha256"] != b["events_sha256"]


def test_cohort_and_counterexamples_do_not_drop_adverse_outcomes(monkeypatch):
    def stub(idx):
        return fake_trial(idx, with_late=(idx % 2 == 0),
                          with_regret=(idx % 3 == 0))
    monkeypatch.setattr(h013, "audit_one", stub)
    output = h013._cohort([4096, 4097, 4098], name="discovery")
    assert output["aggregate"]["targets_evaluated"] == 2
    classified = h013.evaluate_counterexamples(output)
    assert classified["regret_count"] == 1
    assert classified["late_rank1_count"] == 2
    assert classified["counterexamples"][0]["trial_index"] == 4098


def test_cost_regret_and_late_reversal_are_separate_taxonomies():
    values = [fake_trial(4096, with_late=True),
              fake_trial(4097, with_regret=True)]
    report = h013.evaluate_counterexamples({"trials": values})
    assert report["late_rank1_count"] == 1
    assert report["regret_count"] == 1
    assert report["counterexamples"][0]["trial_index"] == 4097
    assert report["late_rank1_cases"][0]["trial_index"] == 4096


def test_segment_retains_expanded_rank_and_gap_around_selected_window():
    rows = [{
        "window_index": i + 10, "event_range": [i*4, i*4+3],
        "best_tagged_expanded_rank": 1 if i == 4 else 35,
        "natural_tagged_count": 8 if i == 4 else 0,
        "retained_tagged_count": 8, "replaced_slots": 0 if i == 4 else 8,
        "best_tagged_minus_best_untagged": -1. if i == 4 else 2.,
    } for i in range(9)]
    selected = h013._segment(rows, 4, radius=2)
    assert [r["offset_chunks"] for r in selected] == [2, 3, 4, 5, 6]
    assert selected[2]["expanded_rank"] == 1
    assert selected[2]["tagged_minus_untagged_cost"] == -1.


def test_frozen_horizon_and_quota_unchanged():
    assert h013.QUOTA == 8
    assert h013.OBSERVE_CHUNKS == 5
    assert h013.DEFAULT_DISCOVERY == h013.DEFAULT_EVALUATION == 32


def test_fresh_seed_real_smoke_replays_without_source_data():
    # One synthetic trial fully exercises frozen planner, full quota8,
    # per-window parity and online adaptive comparison. No real Train.
    sample = h013._cohort([h013.DISCOVERY_START], name="smoke")
    assert sample["aggregate"]["synthetic_trials_generated"] == 1
    assert sample["aggregate"]["feasible_frozen_trials"] == 1
    assert sample["aggregate"]["targets_evaluated"] >= 1
    assert all(row["generator"]["seed"] == h013.BASE_SEED + 7919*row["trial_index"]
               for row in sample["trials"])
