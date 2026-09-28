from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v062 as trainer  # noqa: E402


def _result(
    *,
    hits: int = 90,
    misses: int = 10,
    xacc: float = 70.0,
    pp: float = 0.50,
    early: int = 10,
    overloaded: bool = False,
    targets: int = 100,
):
    stats = SimpleNamespace(
        hits=hits,
        misses=misses,
        x_accuracy_percent=xacc,
        perfect_rate=pp,
        too_early_presses=early,
        overloaded=overloaded,
        targets=targets,
    )
    return trainer.v054.StudentEvalResult(stats, hits + early)


def test_default_bootstrap_windows_cover_three_distinct_train_regions():
    windows = trainer._even_windows(0.0, 90.0, 30.0, 3)
    assert windows == (
        trainer.SegmentWindow(0.0, 30.0),
        trainer.SegmentWindow(30.0, 60.0),
        trainer.SegmentWindow(60.0, 90.0),
    )


def test_sampled_training_window_stays_inside_pool():
    rng = random.Random(123)
    for _ in range(20):
        window = trainer._sample_window(rng, 0.0, 90.0, 30.0)
        assert 0.0 <= window.start_s <= 60.0
        assert window.end_s - window.start_s == 30.0
        assert window.end_s <= 90.0


def test_ranges_must_not_overlap_validation_or_sight():
    trainer._validate_disjoint_ranges(
        trainer.SegmentWindow(0.0, 90.0),
        trainer.SegmentWindow(90.0, 110.0),
        trainer.SegmentWindow(110.0, 130.0),
    )
    try:
        trainer._validate_disjoint_ranges(
            trainer.SegmentWindow(0.0, 95.0),
            trainer.SegmentWindow(90.0, 110.0),
            trainer.SegmentWindow(110.0, 130.0),
        )
    except ValueError as exc:
        assert "must not overlap" in str(exc)
    else:
        raise AssertionError("overlapping train/validation ranges must fail")


def test_validation_guard_rejects_safe_to_overload():
    reference = _result(overloaded=False)
    candidate = _result(hits=95, xacc=80.0, overloaded=True)
    decision = trainer._validation_guard(reference, candidate)
    assert not decision.accepted
    assert decision.reason == "validation safe->overload"


def test_validation_guard_allows_small_non_regression_without_requiring_improvement():
    reference = _result(hits=90, xacc=70.0, early=10, overloaded=False)
    candidate = _result(hits=88, xacc=68.5, early=13, overloaded=False)
    decision = trainer._validation_guard(reference, candidate)
    assert decision.accepted
    assert decision.reason == "validation preserved"


def test_validation_guard_uses_fixed_floor_to_reject_large_xacc_loss():
    reference = _result(hits=90, xacc=70.0, early=10, overloaded=False)
    candidate = _result(hits=90, xacc=67.9, early=10, overloaded=False)
    decision = trainer._validation_guard(reference, candidate)
    assert not decision.accepted
    assert decision.reason == "validation XAcc regression>2pt"


def test_validation_guard_allows_overload_escape_and_nonworsening_progress():
    overloaded = _result(hits=20, xacc=50.0, early=8, overloaded=True)
    still_overloaded = _result(hits=21, xacc=50.5, early=9, overloaded=True)
    escaped = _result(hits=40, xacc=55.0, early=9, overloaded=False)
    assert trainer._validation_guard(overloaded, still_overloaded).accepted
    decision = trainer._validation_guard(overloaded, escaped)
    assert decision.accepted
    assert decision.reason == "validation escaped overload"


def test_dual_choice_requires_both_train_and_validation_guards():
    base_train = _result(hits=90, xacc=68.0)
    good_train = _result(hits=91, xacc=70.0)
    better_train = _result(hits=92, xacc=71.0)
    validation = _result(hits=90, xacc=70.0)

    accepted = trainer.DualCandidate(
        0.25,
        good_train,
        validation,
        trainer.v057.ConservativeDecision(True, "accuracy-first better"),
        trainer.v057.ConservativeDecision(True, "validation preserved"),
    )
    rejected_validation = trainer.DualCandidate(
        0.5,
        better_train,
        validation,
        trainer.v057.ConservativeDecision(True, "accuracy-first better"),
        trainer.v057.ConservativeDecision(False, "validation safe->overload"),
    )
    choice = trainer._choose_dual_candidate(base_train, [accepted, rejected_validation])
    assert choice.accepted
    assert choice.alpha == 0.25


def test_bootstrap_selection_requires_all_segments_safe_before_unsafe_high_score():
    safe_train = [_result(hits=80, xacc=60.0), _result(hits=82, xacc=62.0)]
    safe_val = _result(hits=70, xacc=58.0)
    unsafe_train = [_result(hits=99, xacc=95.0), _result(hits=99, xacc=95.0, overloaded=True)]
    unsafe_val = _result(hits=99, xacc=95.0)
    assert (
        trainer._bootstrap_selection_key(safe_train, safe_val)
        > trainer._bootstrap_selection_key(unsafe_train, unsafe_val)
    )


def test_v062_defaults_keep_sight_outside_train_and_validation():
    assert trainer.CHECKPOINT_FORMAT_VERSION == 10
    assert trainer.DEFAULT_CHECKPOINT.endswith("real_chart_v062_multisegment.pt")
    assert trainer.DEFAULT_TRAIN_POOL_END <= trainer.DEFAULT_VALIDATION_START
    assert trainer.DEFAULT_VALIDATION_END <= trainer.DEFAULT_SIGHT_START
