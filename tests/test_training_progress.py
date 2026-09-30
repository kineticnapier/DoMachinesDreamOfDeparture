from __future__ import annotations

from types import SimpleNamespace

from dmdod.training_progress import emit_progress, subscribe, unsubscribe

import train_real_chart_v090_progress as progress


def test_progress_bus_subscribe_emit_unsubscribe() -> None:
    received = []

    def listener(event):
        received.append(event)

    subscribe(listener)
    try:
        emit_progress("bc_chunk", current=3, total=10)
    finally:
        unsubscribe(listener)

    emit_progress("bc_chunk", current=4, total=10)

    assert len(received) == 1
    assert received[0].kind == "bc_chunk"
    assert received[0].values == {"current": 3, "total": 10}


def test_progress_listener_failure_never_escapes() -> None:
    def broken(_event):
        raise RuntimeError("display failure")

    subscribe(broken)
    try:
        emit_progress("guard_start", alphas=6)
    finally:
        unsubscribe(broken)


def test_eval_label_identifies_major_roles() -> None:
    anchors = [SimpleNamespace(role="anchor-1"), SimpleNamespace(role="anchor-2")]
    validation = [SimpleNamespace(role="validation")]
    final = [SimpleNamespace(role="final-holdout")]

    assert progress._eval_label(anchors) == "Anchor eval"
    assert progress._eval_label(validation) == "Validation eval"
    assert progress._eval_label(final) == "Final holdout"
