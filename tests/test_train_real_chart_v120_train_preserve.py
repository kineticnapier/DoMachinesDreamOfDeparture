from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v120_train_preserve as v120


def _eval(
    *,
    hits: int = 100,
    targets: int = 100,
    x: float = 95.0,
    pp: float = 90.0,
    early: int = 0,
    misses: int | None = None,
    overloaded: bool = False,
):
    if misses is None:
        misses = max(0, targets - hits)
    return SimpleNamespace(
        stats=SimpleNamespace(
            hits=hits,
            targets=targets,
            x_accuracy_percent=x,
            perfect_rate=pp,
            too_early_presses=early,
            misses=misses,
            overloaded=overloaded,
        )
    )


def test_train_preservation_accepts_identical_policy() -> None:
    base = _eval()
    candidate = _eval()
    decision = v120._train_preservation_guard(base, candidate)
    assert decision.accepted
    assert decision.reason == "train preserved"


def test_train_preservation_rejects_real_regressions() -> None:
    base = _eval(hits=100, targets=100, x=95.0, early=1)

    assert not v120._train_preservation_guard(
        base, _eval(hits=100, targets=100, x=95.0, early=1, overloaded=True)
    ).accepted
    assert not v120._train_preservation_guard(
        base, _eval(hits=96, targets=100, x=95.0, early=1)
    ).accepted
    assert not v120._train_preservation_guard(
        base, _eval(hits=100, targets=100, x=92.9, early=1)
    ).accepted
    assert not v120._train_preservation_guard(
        base, _eval(hits=100, targets=100, x=95.0, early=6)
    ).accepted


def test_train_preservation_allows_overloaded_policy_to_escape() -> None:
    base = _eval(hits=20, targets=100, x=50.0, overloaded=True)
    candidate = _eval(hits=10, targets=100, x=40.0, overloaded=False)
    decision = v120._train_preservation_guard(base, candidate)
    assert decision.accepted
    assert decision.reason == "train escaped overload"


def test_aggregate_key_keeps_start_clean_ahead_of_completion() -> None:
    normal = SimpleNamespace(role="anchor-1")
    start = SimpleNamespace(role="start-micro-4")

    clean_key = v120._aggregate_key(
        _eval(hits=90),
        (_eval(hits=90),),
        (_eval(hits=90), _eval(hits=4, targets=4, early=0)),
        (normal, start),
    )
    dirty_but_more_complete_key = v120._aggregate_key(
        _eval(hits=100),
        (_eval(hits=100),),
        (_eval(hits=100), _eval(hits=4, targets=4, early=1)),
        (normal, start),
    )

    assert clean_key > dirty_but_more_complete_key


def test_aggregate_key_prefers_completion_when_start_gate_ties() -> None:
    normal = SimpleNamespace(role="anchor-1")
    start = SimpleNamespace(role="start-micro-4")

    lower = v120._aggregate_key(
        _eval(hits=90),
        (_eval(hits=90),),
        (_eval(hits=90), _eval(hits=4, targets=4)),
        (normal, start),
    )
    higher = v120._aggregate_key(
        _eval(hits=91),
        (_eval(hits=90),),
        (_eval(hits=90), _eval(hits=4, targets=4)),
        (normal, start),
    )

    assert higher > lower


def test_aggregate_delta_reports_first_lexicographic_loss() -> None:
    base = (1, 0, -2, 0.9500, 93.0, 80.0, -10, -20)
    candidate = (1, 0, -2, 0.9490, 94.0, 82.0, -8, -19)

    delta = v120._aggregate_key_delta(base, candidate)

    assert delta["first"] == "completion"
    assert delta["relation"] == "LOSS"
    assert abs(delta["completion_pp"] + 0.1) < 1e-9
    assert delta["xacc_pt"] == 1.0
    assert delta["pp_pt"] == 2.0
    assert delta["early"] == -2
    assert delta["miss"] == -1


def test_aggregate_delta_uses_human_direction_for_start_early() -> None:
    base = (1, 0, -2, 0.95, 93.0, 80.0, -10, -20)
    candidate = (1, 0, -3, 0.96, 94.0, 82.0, -10, -20)

    delta = v120._aggregate_key_delta(base, candidate)

    assert delta["first"] == "start-early"
    assert delta["relation"] == "LOSS"
    assert delta["start_early"] == 1
    text = v120._format_aggregate_delta(0.125, base, candidate)
    assert "a=0.125" in text
    assert "first=start-early:LOSS" in text
    assert "dStartEarly=+1" in text
