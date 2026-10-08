from __future__ import annotations

from types import SimpleNamespace

from dmdod.training.real_chart import (
    aggregate,
    safe_anchor_count,
    selection_key,
    survival_selection_key,
    train_safety_guard,
    train_survival_guard,
)


def _stats(
    *,
    hits: int,
    targets: int = 100,
    x: float = 50.0,
    early: int = 5,
    overloaded: bool = False,
):
    return SimpleNamespace(
        hits=hits,
        targets=targets,
        too_early_presses=early,
        overloaded=overloaded,
        x_accuracy_points=x,
        x_accuracy_denominator=100.0,
    )


def test_safety_guard_rejects_safe_to_overload() -> None:
    reference = [(_stats(hits=80, overloaded=False), 90)]
    candidate = [(_stats(hits=81, overloaded=True), 91)]

    safe, reasons = train_safety_guard(reference, candidate)

    assert not safe
    assert reasons == ("anchor 1 safe->overload",)


def test_selection_key_prefers_hits_then_x_then_lower_noise() -> None:
    a = [(_stats(hits=80, x=50, early=10), 100)]
    b = [(_stats(hits=81, x=1, early=100), 1000)]
    c = [(_stats(hits=80, x=51, early=20), 100)]

    assert selection_key(b) > selection_key(a)
    assert selection_key(c) > selection_key(a)
    assert "H=80/100" in aggregate(a)


def test_survival_guard_rejects_only_safe_to_overload() -> None:
    references = [
        (_stats(hits=80, overloaded=False), 90),
        (_stats(hits=80, overloaded=True), 90),
    ]
    hit_drop_only = [
        (_stats(hits=1, overloaded=False), 2),
        (_stats(hits=0, overloaded=True), 1),
    ]

    safe, reasons = train_survival_guard(references, hit_drop_only)

    assert safe
    assert reasons == ()

    dead_candidate = [
        (_stats(hits=81, overloaded=True), 91),
        (_stats(hits=0, overloaded=True), 1),
    ]
    safe, reasons = train_survival_guard(references, dead_candidate)

    assert not safe
    assert reasons == ("anchor 1 safe->overload",)


def test_survival_selection_prioritizes_safe_anchor_recovery() -> None:
    baseline = [
        (_stats(hits=90, overloaded=False), 100),
        (_stats(hits=90, overloaded=True), 100),
    ]
    recovered_with_fewer_hits = [
        (_stats(hits=70, overloaded=False), 80),
        (_stats(hits=1, overloaded=False), 2),
    ]

    assert safe_anchor_count(baseline) == 1
    assert safe_anchor_count(recovered_with_fewer_hits) == 2
    assert (
        survival_selection_key(recovered_with_fewer_hits)
        > survival_selection_key(baseline)
    )
