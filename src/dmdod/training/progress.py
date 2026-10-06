from __future__ import annotations

"""Tiny process-local event bus for optional training progress UIs.

Training code emits best-effort observational events through this module.  With
no listeners installed the cost is a small function call; listener failures are
swallowed deliberately so a terminal frontend can never change training
semantics or abort a long run.
"""

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class TrainingProgressEvent:
    kind: str
    values: dict[str, Any] = field(default_factory=dict)


ProgressListener = Callable[[TrainingProgressEvent], None]

_LISTENERS: list[ProgressListener] = []
_LOCK = RLock()


def subscribe(listener: ProgressListener) -> None:
    with _LOCK:
        if listener not in _LISTENERS:
            _LISTENERS.append(listener)


def unsubscribe(listener: ProgressListener) -> None:
    with _LOCK:
        try:
            _LISTENERS.remove(listener)
        except ValueError:
            pass


def emit_progress(kind: str, **values: Any) -> None:
    with _LOCK:
        listeners = tuple(_LISTENERS)
    if not listeners:
        return

    event = TrainingProgressEvent(str(kind), dict(values))
    for listener in listeners:
        try:
            listener(event)
        except Exception:
            # Progress display is execution-only.  Never let a rendering bug
            # affect model updates, guards, checkpoints, or reproducibility.
            continue
