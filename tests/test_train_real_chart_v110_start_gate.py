from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v090_turbo as turbo
import train_real_chart_v110_start_gate as v110


def _named(role: str):
    return SimpleNamespace(role=role)


def _eval(early: int):
    return SimpleNamespace(stats=SimpleNamespace(too_early_presses=int(early)))


def test_start_micro_early_counts_only_start_micro_anchors(monkeypatch) -> None:
    monkeypatch.setattr(
        turbo,
        "_CAPTURED_ANCHORS",
        [_named("anchor-1"), _named("start-micro-4"), _named("start-micro-4")],
    )
    assert v110._start_micro_early([_eval(9), _eval(1), _eval(2)]) == 3


def test_clean_start_outranks_dirty_start_at_equal_safety(monkeypatch) -> None:
    monkeypatch.setattr(turbo, "_CAPTURED_ANCHORS", [_named("start-micro-4")])
    monkeypatch.setattr(
        v110,
        "_BASE_BOOTSTRAP_KEY",
        lambda anchors, validations: (1, 0.50, 50.0, 0.50, -10),
    )
    clean = v110._bootstrap_key_start_gate([_eval(0)], [])

    monkeypatch.setattr(
        v110,
        "_BASE_BOOTSTRAP_KEY",
        lambda anchors, validations: (1, 1.0, 100.0, 1.0, 0),
    )
    dirty = v110._bootstrap_key_start_gate([_eval(1)], [])

    assert clean > dirty


def test_safe_dirty_start_outranks_unsafe_clean_start(monkeypatch) -> None:
    monkeypatch.setattr(turbo, "_CAPTURED_ANCHORS", [_named("start-micro-4")])
    monkeypatch.setattr(
        v110,
        "_BASE_BOOTSTRAP_KEY",
        lambda anchors, validations: (0, 1.0, 100.0, 1.0, 0),
    )
    unsafe_clean = v110._bootstrap_key_start_gate([_eval(0)], [])

    monkeypatch.setattr(
        v110,
        "_BASE_BOOTSTRAP_KEY",
        lambda anchors, validations: (1, 0.50, 50.0, 0.50, -10),
    )
    safe_dirty = v110._bootstrap_key_start_gate([_eval(2)], [])

    assert safe_dirty > unsafe_clean


def test_fewer_start_early_is_fallback_order_while_both_dirty(monkeypatch) -> None:
    monkeypatch.setattr(turbo, "_CAPTURED_ANCHORS", [_named("start-micro-4")])
    monkeypatch.setattr(
        v110,
        "_BASE_BOOTSTRAP_KEY",
        lambda anchors, validations: (1, 1.0, 100.0, 1.0, 0),
    )
    one_early = v110._bootstrap_key_start_gate([_eval(1)], [])
    two_early = v110._bootstrap_key_start_gate([_eval(2)], [])

    assert one_early > two_early


def test_aggregate_pruning_is_disabled_until_equal_safety_best_is_start_clean(monkeypatch) -> None:
    dirty_best = (1, 0, -3, 0.99, 99.0, 0.99, -3)
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        return "completion"

    monkeypatch.setattr(v110, "_BASE_BOOTSTRAP_PRUNE_REASON", fail_if_called)
    reason = v110._bootstrap_prune_reason_start_gate(
        dirty_best,
        any_overloaded=False,
        partial_hits=0,
        evaluated_targets=10,
        total_targets=100,
    )

    assert reason is None
    assert not called


def test_safety_pruning_remains_active_before_start_gate(monkeypatch) -> None:
    safe_dirty_best = (1, 0, -3, 0.99, 99.0, 0.99, -3)
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(v110, "_BASE_BOOTSTRAP_PRUNE_REASON", fail_if_called)
    reason = v110._bootstrap_prune_reason_start_gate(
        safe_dirty_best,
        any_overloaded=True,
        partial_hits=100,
        evaluated_targets=100,
        total_targets=100,
    )

    assert reason == "safety"
    assert not called


def test_aggregate_pruning_resumes_after_clean_best(monkeypatch) -> None:
    clean_best = (1, 1, 0, 0.99, 99.0, 0.99, 0)
    seen = None

    def record_base(best_key, **kwargs):
        nonlocal seen
        seen = best_key
        return "completion"

    monkeypatch.setattr(v110, "_BASE_BOOTSTRAP_PRUNE_REASON", record_base)
    reason = v110._bootstrap_prune_reason_start_gate(
        clean_best,
        any_overloaded=False,
        partial_hits=0,
        evaluated_targets=10,
        total_targets=100,
    )

    assert reason == "completion"
    assert seen == (1, 0.99, 99.0, 0.99, 0)
