from __future__ import annotations

import pytest

from dmdod.evaluation.evaluator import TargetHit, TimingEvaluator
from dmdod.motor.env import TimedKeyEvent
from dmdod.motor.keyboard import KeyEvent


@pytest.mark.parametrize("second_key", ["left", "right"])
def test_equal_time_targets_preserve_individual_miss_records(second_key: str) -> None:
    evaluator = TimingEvaluator([TargetHit(1.0, "left"), TargetHit(1.0, second_key)])
    evaluator.record((TimedKeyEvent(1.0, "left", KeyEvent.DOWN),))

    summary = evaluator.finalize()
    assert (summary.targets, summary.hits, summary.misses) == (2, 1, 1)
    assert len(evaluator.judgements) == 2
    assert sum(row.event_time_s is None for row in evaluator.judgements) == 1
    assert sum(row.event_time_s is not None for row in evaluator.judgements) == 1

    assert evaluator.finalize() == summary
    assert len(evaluator.judgements) == 2


def test_equal_time_all_missed_are_recorded_exactly_once() -> None:
    evaluator = TimingEvaluator([TargetHit(1.0), TargetHit(1.0)])
    summary = evaluator.finalize()
    assert (summary.targets, summary.hits, summary.misses) == (2, 0, 2)
    assert len(evaluator.judgements) == 2
    assert all(row.event_time_s is None for row in evaluator.judgements)

    assert evaluator.finalize() == summary
    assert len(evaluator.judgements) == 2


def test_distinct_time_targets_preserve_existing_scoring() -> None:
    evaluator = TimingEvaluator([TargetHit(1.0, "left"), TargetHit(2.0, "right")])
    evaluator.record((TimedKeyEvent(1.025, "left", KeyEvent.DOWN),))
    summary = evaluator.finalize()
    assert (summary.targets, summary.hits, summary.misses) == (2, 1, 1)
    assert summary.mean_abs_error_ms == pytest.approx(25.0)
    assert len(evaluator.judgements) == 2
