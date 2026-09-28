from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v061 as trainer  # noqa: E402


def _result(
    *,
    hits: int = 0,
    misses: int = 0,
    xacc: float = 15.0,
    pp: float = 0.0,
    early: int = 4,
    overloaded: bool = True,
    targets: int = 107,
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


def test_guard_prefers_escape_from_overload():
    best = _result(hits=8, xacc=60.0, pp=0.53, early=6, overloaded=True)
    candidate = _result(hits=84, misses=23, xacc=61.5, pp=0.48, early=22, overloaded=False)
    decision = trainer._safety_guard(best, candidate)
    assert decision.accepted
    assert decision.reason == "escaped overload"


def test_guard_rejects_both_overloaded_even_when_xacc_improves():
    best = _result(hits=0, xacc=13.33, early=3, overloaded=True)
    candidate = _result(hits=8, xacc=60.0, pp=0.53, early=6, overloaded=True)
    decision = trainer._safety_guard(best, candidate)
    assert not decision.accepted
    assert decision.reason == "both overloaded"


def test_guard_rejects_safe_to_overload():
    best = _result(hits=90, misses=17, xacc=68.0, pp=0.50, early=13, overloaded=False)
    candidate = _result(hits=10, xacc=80.0, pp=0.70, early=5, overloaded=True)
    decision = trainer._safety_guard(best, candidate)
    assert not decision.accepted
    assert decision.reason == "safe->overload"


def test_guard_keeps_existing_conservative_rules_when_both_safe():
    best = _result(hits=90, misses=17, xacc=68.0, pp=0.50, early=13, overloaded=False)
    candidate = _result(hits=91, misses=16, xacc=70.0, pp=0.52, early=14, overloaded=False)
    decision = trainer._safety_guard(best, candidate)
    assert decision.accepted
    assert decision.reason == "accuracy-first better"


def test_escape_selection_prefers_completion_over_tiny_high_xacc_run():
    best = _result(hits=0, xacc=13.33, early=3, overloaded=True)
    tiny = _result(hits=4, misses=1, xacc=90.0, pp=0.90, early=2, overloaded=False)
    complete = _result(hits=90, misses=17, xacc=64.0, pp=0.47, early=22, overloaded=False)
    candidates = [
        trainer.v058.TrustCandidate(
            0.25,
            tiny,
            trainer.v057.ConservativeDecision(True, "escaped overload"),
        ),
        trainer.v058.TrustCandidate(
            0.5,
            complete,
            trainer.v057.ConservativeDecision(True, "escaped overload"),
        ),
    ]
    choice = trainer._choose_line_search_candidate(best, candidates)
    assert choice.accepted
    assert choice.alpha == 0.5
    assert choice.evaluation is complete


def test_bootstrap_safe_key_prefers_more_complete_safe_policy():
    accurate_short = _result(hits=40, misses=0, xacc=95.0, pp=0.90, early=0, overloaded=False)
    complete = _result(hits=90, misses=17, xacc=65.0, pp=0.50, early=10, overloaded=False)
    assert trainer._safe_escape_key(complete) > trainer._safe_escape_key(accurate_short)


def test_v061_checkpoint_format_and_name():
    assert trainer.CHECKPOINT_FORMAT_VERSION == 9
    assert trainer.DEFAULT_CHECKPOINT.endswith("real_chart_v061_safety_selected.pt")
