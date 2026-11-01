"""Synthetic contract tests for research A001b's precommitted measurement."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from a001_periodic_motor_probe import Trial, target_times
from a001b_steady_motor_survey import (
    WARMUP_S, JUDGEMENT_S, INTERVALS_MS, DUTIES, LEADS_MS, MODES,
    CONFIRM_STARTS_S, score_trace, run_trial, candidate_order, survey,
)


def test_fixed_protocol_constants_are_not_silent_copies_of_old_posthoc_cutoff():
    assert WARMUP_S == 3.0
    assert JUDGEMENT_S == .020
    assert INTERVALS_MS == (60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,220,260)
    assert DUTIES == (.25, .40, .55, .70)
    assert LEADS_MS == (20,50,80,110,140,170,200,230,260,290,320,350)
    assert MODES == ("single_left", "single_right", "alternating")
    assert CONFIRM_STARTS_S == (1.137,1.273,1.419)


def test_all_target_matching_and_phase_scoring_have_no_double_count():
    trial = Trial("alternating", 100, .4, 80, 1.13, 5.0)
    targets = target_times(trial)
    downs = [(t, "left" if i % 2 == 0 else "right") for i,t in enumerate(targets)]
    score = score_trace(trial, targets, downs)
    assert score["warmup"]["targets"] + score["steady"]["targets"] == len(targets)
    assert score["warmup"]["hits"] + score["steady"]["hits"] == len(targets)
    assert score["steady"]["extra_keydowns"] == 0
    assert score["steady"]["wrong_axis"] == 0
    assert score["passed"] is True


def test_wrong_axis_does_not_pass_as_alternating_even_with_perfect_timing():
    trial = Trial("alternating", 100, .4, 80, 1.13, 5.0)
    targets = target_times(trial)
    downs = [(t, "left") for t in targets]
    score = score_trace(trial, targets, downs)
    assert score["steady"]["hit_rate"] == 1
    assert score["steady"]["wrong_axis"] > 0
    assert score["passed"] is False


def test_single_key_uses_only_one_key_and_extra_is_fatal():
    trial = Trial("single_left", 180, .4, 80, 1.13, 5.0)
    targets = target_times(trial)
    downs = [(t, "left") for t in targets]
    steady_i = next(i for i,t in enumerate(targets) if t >= trial.start_s + WARMUP_S)
    # In between two expected hits this keydown cannot be matched.
    extra = .5 * (targets[steady_i] + targets[steady_i+1])
    score = score_trace(trial, targets, downs + [(extra, "right")])
    assert score["steady"]["extra_keydowns"] == 1
    assert score["passed"] is False


def test_boundary_keydown_never_counted_twice():
    trial = Trial("single_right", 100, .4, 80, 1.13, 5.0)
    targets = target_times(trial)
    downs = [(t, "right") for t in targets]
    start = trial.start_s + WARMUP_S
    downs.append((start - .030, "right"))
    score = score_trace(trial, targets, downs)
    assert score["warmup"]["hits"] + score["steady"]["hits"] == len(targets)
    assert score["warmup"]["extra_keydowns"] + score["steady"]["extra_keydowns"] <= 1


def test_selection_scores_correct_phase_not_startup():
    trial = Trial("alternating", 100, .4, 80, 1.13, 5.0)
    targets = target_times(trial)
    steady_i = next(i for i,t in enumerate(targets) if t >= trial.start_s + WARMUP_S)
    downs = [(t,"left" if i%2==0 else "right") for i,t in enumerate(targets) if i >= steady_i]
    score = score_trace(trial,targets,downs)
    assert score["warmup"]["hits"] == 0
    assert score["steady"]["hit_rate"] == 1
    assert score["passed"] is True


def test_periodic_motor_trial_reproducible_and_not_learning():
    a=run_trial("single_left",180,.55,80,1.13,1.1)
    b=run_trial("single_left",180,.55,80,1.13,1.1)
    assert a==b
    assert "score" in a and "end_fatigue" in a


def test_candidate_ranking_prefers_clean_steady_not_warmup():
    base=run_trial("single_left",180,.55,80,1.13,1.1)
    better = {**base, "score": {**base["score"], "passed": True}}
    assert candidate_order(better)>candidate_order(base) or base["score"]["passed"]


def test_quick_survey_is_explicitly_not_full_validation():
    result=survey(quick=True)
    assert len(result["rows"]) == 4
    assert result["conditions"]["confirm_starts_s"] == (1.137,)
    assert result["status"].startswith("precommitted exploratory")
    assert not result["claims"]["student_training_performed"]
