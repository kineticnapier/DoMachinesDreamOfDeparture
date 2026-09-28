from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v064 as trainer  # noqa: E402


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


def _decision(accepted: bool, reason: str = "ok"):
    return trainer.v057.ConservativeDecision(accepted, reason)


def _candidate(
    *,
    alpha: float,
    train,
    validation,
    anchors,
    train_ok: bool = True,
    validation_ok: bool = True,
    anchor_ok: tuple[bool, ...] | None = None,
):
    if anchor_ok is None:
        anchor_ok = tuple(True for _ in anchors)
    return trainer.AnchorCandidate(
        alpha=alpha,
        train_eval=train,
        validation_eval=validation,
        anchor_evals=tuple(anchors),
        train_decision=_decision(train_ok, "train"),
        validation_decision=_decision(validation_ok, "validation"),
        anchor_decisions=tuple(
            _decision(ok, f"anchor-{index}")
            for index, ok in enumerate(anchor_ok)
        ),
    )


def test_anchor_guard_rejects_safe_to_overload():
    reference = _result(overloaded=False)
    candidate = _result(hits=95, xacc=80.0, overloaded=True)
    decision = trainer._anchor_guard(reference, candidate)
    assert not decision.accepted
    assert decision.reason == "anchor safe->overload"


def test_anchor_guard_rejects_large_hit_regression():
    reference = _result(hits=90, targets=100, xacc=70.0)
    candidate = _result(hits=86, targets=100, xacc=72.0)
    decision = trainer._anchor_guard(reference, candidate)
    assert not decision.accepted
    assert decision.reason == "anchor hit regression>3"


def test_anchor_guard_rejects_large_xacc_regression():
    reference = _result(hits=90, xacc=70.0)
    candidate = _result(hits=90, xacc=67.9)
    decision = trainer._anchor_guard(reference, candidate)
    assert not decision.accepted
    assert decision.reason == "anchor XAcc regression>2pt"


def test_anchor_guard_allows_small_preserved_drift():
    reference = _result(hits=90, xacc=70.0, early=10)
    candidate = _result(hits=88, xacc=68.5, early=13)
    decision = trainer._anchor_guard(reference, candidate)
    assert decision.accepted
    assert decision.reason == "anchor preserved"


def test_candidate_requires_every_anchor_guard():
    train = _result(hits=90, xacc=70.0)
    validation = _result(hits=90, xacc=70.0)
    anchors = (_result(), _result(), _result())
    candidate = _candidate(
        alpha=0.25,
        train=train,
        validation=validation,
        anchors=anchors,
        anchor_ok=(True, False, True),
    )
    assert not candidate.accepted


def test_choice_rejects_better_train_candidate_that_forgets_anchor():
    base = _result(hits=90, xacc=68.0)
    validation = _result(hits=90, xacc=70.0)
    anchors = (_result(), _result(), _result())

    preserved = _candidate(
        alpha=0.125,
        train=_result(hits=91, xacc=69.0),
        validation=validation,
        anchors=anchors,
    )
    forgetting = _candidate(
        alpha=0.5,
        train=_result(hits=96, xacc=80.0),
        validation=validation,
        anchors=anchors,
        anchor_ok=(True, False, True),
    )

    choice = trainer._choose_anchor_candidate(base, [preserved, forgetting])
    assert choice.accepted
    assert choice.alpha == 0.125
    assert choice.train_eval is preserved.train_eval


def test_reference_update_never_accepts_worse_lexicographic_reference():
    reference = _result(hits=90, xacc=70.0, pp=0.50)
    worse = _result(hits=90, xacc=69.0, pp=0.80)
    better = _result(hits=90, xacc=71.0, pp=0.40)
    assert trainer._update_reference(reference, worse) is reference
    assert trainer._update_reference(reference, better) is better


def test_checkpoint_payload_contains_anchor_floors_and_v064_format():
    args = SimpleNamespace(hidden=96, rounds=4, keep_round_checkpoints=False)
    reference = _result(hits=90, xacc=70.0)
    anchors = (_result(hits=80, xacc=60.0), _result(hits=82, xacc=62.0))
    payload = trainer._checkpoint_payload(
        model_state={},
        args=args,
        chart_path="chart.adofai",
        signature={"anchor_guard": trainer.ANCHOR_GUARD_VERSION},
        completed_round=2,
        rng_state=(3, (), None),
        validation_reference=reference,
        anchor_references=anchors,
        bootstrap_history=[],
        round_history=[],
        finalized=False,
    )
    assert payload["format_version"] == 12
    assert payload["anchor_guard"] == trainer.ANCHOR_GUARD_VERSION
    assert len(payload["anchor_references"]) == 2
    assert payload["anchor_references"][0]["hits"] == 80


def test_v064_checkpoint_name_and_guard_version():
    assert trainer.CHECKPOINT_FORMAT_VERSION == 12
    assert trainer.DEFAULT_CHECKPOINT.endswith("real_chart_v064_anchor_guard.pt")
    assert trainer.ANCHOR_GUARD_VERSION == "per-anchor-v1"
