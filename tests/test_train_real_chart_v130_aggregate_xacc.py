from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v130_aggregate_xacc as v130


def _eval(
    *,
    targets: int = 100,
    hits: int = 100,
    xacc: float = 95.0,
    early: int = 0,
    overloaded: bool = False,
):
    return SimpleNamespace(
        stats=SimpleNamespace(
            targets=targets,
            hits=hits,
            x_accuracy_percent=xacc,
            too_early_presses=early,
            overloaded=overloaded,
        )
    )


def test_local_validation_guard_defers_xacc_only() -> None:
    reference = _eval(xacc=99.0)
    candidate = _eval(xacc=50.0)

    decision = v130._validation_guard_without_xacc(reference, candidate)

    assert decision.accepted
    assert "XAcc deferred" in decision.reason


def test_local_validation_guard_keeps_overload_hard() -> None:
    reference = _eval(overloaded=False)
    candidate = _eval(overloaded=True)

    decision = v130._validation_guard_without_xacc(reference, candidate)

    assert not decision.accepted
    assert "safe->overload" in decision.reason


def test_local_validation_guard_keeps_hit_and_early_floors() -> None:
    reference = _eval(targets=100, hits=100, early=0)

    hit_bad = v130._validation_guard_without_xacc(
        reference,
        _eval(targets=100, hits=96, early=0),
    )
    early_bad = v130._validation_guard_without_xacc(
        reference,
        _eval(targets=100, hits=100, early=5),
    )

    assert not hit_bad.accepted
    assert "hit regression" in hit_bad.reason
    assert not early_bad.accepted
    assert "early regression" in early_bad.reason


def test_aggregate_xacc_is_target_weighted_across_role() -> None:
    references = (
        _eval(targets=100, xacc=100.0),
        _eval(targets=300, xacc=80.0),
    )
    candidates = (
        _eval(targets=100, xacc=90.0),
        _eval(targets=300, xacc=84.0),
    )

    reference_x, candidate_x, weight = v130._weighted_xacc_pairs(references, candidates)
    decision = v130._aggregate_xacc_guard(references, candidates, stage="validation")

    assert weight == 400
    assert abs(reference_x - 85.0) < 1e-9
    assert abs(candidate_x - 85.5) < 1e-9
    assert decision.accepted


def test_aggregate_xacc_rejects_role_drop_over_two_points() -> None:
    references = (_eval(xacc=95.0), _eval(xacc=95.0))
    candidates = (_eval(xacc=94.0), _eval(xacc=91.0))

    decision = v130._aggregate_xacc_guard(references, candidates, stage="anchor")

    assert not decision.accepted
    assert "aggregate XAcc regression>2pt" in decision.reason


def test_aggregate_xacc_excludes_overload_escape_pair() -> None:
    references = (
        _eval(targets=100, xacc=99.0, overloaded=True),
        _eval(targets=100, xacc=90.0, overloaded=False),
    )
    candidates = (
        _eval(targets=100, xacc=10.0, overloaded=False),
        _eval(targets=100, xacc=89.0, overloaded=False),
    )

    reference_x, candidate_x, weight = v130._weighted_xacc_pairs(references, candidates)
    decision = v130._aggregate_xacc_guard(references, candidates, stage="validation")

    assert weight == 100
    assert reference_x == 90.0
    assert candidate_x == 89.0
    assert decision.accepted
