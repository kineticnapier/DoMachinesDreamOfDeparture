from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v080 as v080  # noqa: E402
import train_real_chart_v090_turbo as turbo  # noqa: E402


def test_bootstrap_prunes_immediately_when_safe_best_meets_overloaded_candidate():
    best = (1, 0.80, 90.0, 0.9, -2)
    assert turbo._bootstrap_prune_reason(
        best,
        any_overloaded=True,
        partial_hits=20,
        evaluated_targets=25,
        total_targets=100,
    ) == "safety"


def test_bootstrap_prunes_when_even_perfect_remainder_cannot_match_completion():
    best = (1, 0.95, 90.0, 0.9, -2)
    assert turbo._bootstrap_prune_reason(
        best,
        any_overloaded=False,
        partial_hits=35,
        evaluated_targets=50,
        total_targets=100,
    ) == "completion"


def test_bootstrap_does_not_prune_unsafe_best_while_candidate_can_still_be_safe():
    best = (0, 0.99, 99.0, 0.99, 0)
    assert turbo._bootstrap_prune_reason(
        best,
        any_overloaded=False,
        partial_hits=0,
        evaluated_targets=90,
        total_targets=100,
    ) is None


def test_teacher_cache_key_is_stable_and_depends_on_timing(tmp_path, monkeypatch):
    monkeypatch.setenv("DMDOD_TEACHER_CACHE_DIR", str(tmp_path))
    named = SimpleNamespace(
        chart_sha256="abc123",
        start_s=10.0,
        end_s=40.0,
    )
    first = turbo._teacher_cache_path(
        named,
        lead_s=0.043,
        same_hand=True,
        control_dt_s=0.010,
    )
    same = turbo._teacher_cache_path(
        named,
        lead_s=0.043,
        same_hand=True,
        control_dt_s=0.010,
    )
    changed = turbo._teacher_cache_path(
        named,
        lead_s=0.044,
        same_hand=True,
        control_dt_s=0.010,
    )
    assert first == same
    assert first != changed
    assert first.parent == tmp_path


def test_turbo_keeps_v080_checkpoint_format_identity():
    assert v080.CHECKPOINT_FORMAT_VERSION == 15
    assert turbo.TURBO_VERSION.startswith("v090-")
