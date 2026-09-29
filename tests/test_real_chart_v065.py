from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v065 as trainer  # noqa: E402


def _result(
    *,
    hits: int = 90,
    misses: int = 10,
    targets: int = 100,
    xacc: float = 70.0,
    pp: float = 0.5,
    early: int = 10,
    overloaded: bool = False,
):
    stats = SimpleNamespace(
        hits=hits,
        misses=misses,
        targets=targets,
        x_accuracy_percent=xacc,
        perfect_rate=pp,
        too_early_presses=early,
        overloaded=overloaded,
    )
    return trainer.v064.v054.StudentEvalResult(stats, hits + early)


def _segment(offset: float = 0.0):
    return SimpleNamespace(
        targets=(
            SimpleNamespace(
                ordinal=0,
                floor_index=1,
                chart_time_s=1.0 + offset,
                episode_time_s=0.5,
            ),
            SimpleNamespace(
                ordinal=1,
                floor_index=2,
                chart_time_s=1.5 + offset,
                episode_time_s=1.0,
            ),
        )
    )


def test_line_search_memo_runs_identical_key_once():
    memo = trainer.LineSearchMemo()
    calls = 0

    def run():
        nonlocal calls
        calls += 1
        return ("choice", (), None)

    first, first_hit = memo.get_or_run(("same", 1), run)
    second, second_hit = memo.get_or_run(("same", 1), run)

    assert first == second
    assert not first_hit
    assert second_hit
    assert calls == 1
    assert memo.hits == 1
    assert memo.misses == 1


def test_state_digest_changes_when_parameter_changes():
    a = {"weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32)}
    b = {"weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32)}
    c = {"weight": torch.tensor([[1.0, 2.001]], dtype=torch.float32)}

    assert trainer._state_digest(a) == trainer._state_digest(b)
    assert trainer._state_digest(a) != trainer._state_digest(c)


def test_proposal_cache_key_covers_state_segment_and_guard_references():
    kwargs = dict(
        base_state={"w": torch.tensor([1.0])},
        proposal_state={"w": torch.tensor([2.0])},
        base_train_eval=_result(),
        validation_reference=_result(hits=91),
        anchor_references=(_result(hits=80), _result(hits=82)),
        alphas=(1.0, 0.5, 0.25),
        train_segment=_segment(0.0),
        validation_segment=_segment(10.0),
        anchor_segments=[_segment(20.0), _segment(30.0)],
        same_hand=True,
        control_dt_s=0.01,
        device=torch.device("cpu"),
    )
    key = trainer._proposal_cache_key(**kwargs)
    same = trainer._proposal_cache_key(**kwargs)
    assert key == same

    changed_state = dict(kwargs)
    changed_state["proposal_state"] = {"w": torch.tensor([2.1])}
    assert trainer._proposal_cache_key(**changed_state) != key

    changed_segment = dict(kwargs)
    changed_segment["train_segment"] = _segment(0.25)
    assert trainer._proposal_cache_key(**changed_segment) != key

    changed_reference = dict(kwargs)
    changed_reference["validation_reference"] = _result(hits=92)
    assert trainer._proposal_cache_key(**changed_reference) != key


def test_cached_line_search_skips_second_expensive_evaluation(monkeypatch):
    calls = 0

    def fake_line_search(model, **kwargs):
        nonlocal calls
        calls += 1
        return "choice", (), None

    class FakeModel:
        def __init__(self):
            self.loads = 0

        def load_state_dict(self, state):
            self.loads += 1

    memo = trainer.LineSearchMemo()
    monkeypatch.setattr(trainer, "_PROPOSAL_CACHE", memo)
    monkeypatch.setattr(trainer, "_ORIGINAL_ANCHOR_LINE_SEARCH", fake_line_search)

    model = FakeModel()
    kwargs = dict(
        base_state={"w": torch.tensor([1.0])},
        proposal_state={"w": torch.tensor([2.0])},
        base_train_eval=_result(),
        validation_reference=_result(hits=91),
        anchor_references=(_result(hits=80),),
        alphas=(1.0, 0.5),
        train_segment=_segment(0.0),
        validation_segment=_segment(10.0),
        anchor_segments=[_segment(20.0)],
        same_hand=True,
        control_dt_s=0.01,
        device=torch.device("cpu"),
        label_prefix="test e01",
        verbose=True,
    )

    trainer._cached_evaluate_anchor_line_search(model, **kwargs)
    trainer._cached_evaluate_anchor_line_search(model, **kwargs)

    assert calls == 1
    assert memo.misses == 1
    assert memo.hits == 1
    assert model.loads == 1  # cache hit restores the base state for a rollback


def test_v065_identity_constants():
    assert trainer.TRAINER_VERSION == "0.6.5-exact-proposal-cache"
    assert trainer.CHECKPOINT_FORMAT_VERSION == 13
    assert trainer.DEFAULT_CHECKPOINT.endswith("real_chart_v065_proposal_cache.pt")
    assert trainer.PROPOSAL_CACHE_VERSION == "exact-state-v1"
