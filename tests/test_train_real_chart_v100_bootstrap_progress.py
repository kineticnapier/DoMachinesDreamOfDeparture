from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v100_bootstrap_progress as bp


def _named(role: str, key: str, *, hits: int, targets: int, overloaded: bool = False):
    return SimpleNamespace(
        role=role,
        key=(key,),
        segment=SimpleNamespace(targets=tuple(range(targets))),
        fake_stats=SimpleNamespace(
            hits=hits,
            targets=targets,
            overloaded=overloaded,
        ),
    )


def _install_fake_eval(monkeypatch, events):
    monkeypatch.setattr(bp, "emit_progress", lambda kind, **values: events.append((kind, values)))
    monkeypatch.setattr(bp.turbo.v080, "_configured_workers", lambda: 2)
    monkeypatch.setattr(
        bp.turbo,
        "_bootstrap_segment_order",
        lambda segments: list(range(len(segments))),
    )
    monkeypatch.setattr(bp.turbo, "_BOOTSTRAP_FAILURE_COUNTS", {})

    def fake_eval(model, states, segments, *, same_hand, control_dt_s):
        marker = next(iter(states))
        return {
            (marker, named.key): SimpleNamespace(stats=named.fake_stats)
            for named in segments
        }

    monkeypatch.setattr(bp.turbo.v080, "_evaluate_states_on_segments", fake_eval)


def test_bootstrap_progress_reports_each_worker_wave(monkeypatch):
    events = []
    _install_fake_eval(monkeypatch, events)
    monkeypatch.setattr(
        bp.turbo.v080,
        "_bootstrap_key",
        lambda anchors, validations: (1, 1.0, 100.0, 1.0, 0),
    )

    anchors = [
        _named("start-micro-4", "a", hits=4, targets=4),
        _named("anchor-1", "b", hits=10, targets=10),
    ]
    validations = [_named("validation", "v", hits=8, targets=8)]

    got_anchors, got_validations, key, pruned = bp._evaluate_bootstrap_candidate_with_progress(
        object(),
        {},
        anchors,
        validations,
        best_key=(1, 0.50, 0.0, 0.0, 0),
        same_hand=True,
        control_dt_s=0.010,
    )

    assert got_anchors is not None and len(got_anchors) == 2
    assert got_validations is not None and len(got_validations) == 1
    assert key == (1, 1.0, 100.0, 1.0, 0)
    assert pruned is None

    kinds = [kind for kind, _ in events]
    assert kinds == [
        "bootstrap_eval_start",
        "bootstrap_eval_step",
        "bootstrap_eval_step",
        "bootstrap_eval_done",
    ]
    assert events[0][1]["total"] == 3
    assert events[0][1]["waves"] == 2
    assert events[0][1]["start_micro"] == 1
    assert [events[1][1]["current"], events[2][1]["current"]] == [2, 3]
    assert events[-1][1]["status"] == "FULL"


def test_bootstrap_progress_reports_early_prune(monkeypatch):
    events = []
    _install_fake_eval(monkeypatch, events)

    anchors = [
        _named("start-micro-4", "a", hits=3, targets=4, overloaded=True),
        _named("start-micro-4", "b", hits=4, targets=4),
        _named("anchor-1", "c", hits=10, targets=10),
    ]

    got_anchors, got_validations, key, pruned = bp._evaluate_bootstrap_candidate_with_progress(
        object(),
        {},
        anchors,
        [],
        best_key=(1, 0.50, 0.0, 0.0, 0),
        same_hand=True,
        control_dt_s=0.010,
    )

    assert got_anchors is None
    assert got_validations is None
    assert key is None
    assert pruned is not None and pruned["reason"] == "safety"
    assert pruned["evaluated"] == 2
    assert events[-1][0] == "bootstrap_eval_done"
    assert events[-1][1]["status"] == "PRUNE"
    assert events[-1][1]["reason"] == "safety"
    assert events[-1][1]["current"] == 2
