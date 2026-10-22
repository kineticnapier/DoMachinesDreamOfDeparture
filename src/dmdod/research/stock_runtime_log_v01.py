from __future__ import annotations

"""Strict evaluator-only ingestion for a future Stock runtime probe."""

from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Iterable


AUDITED_DLL_SHA256 = "0D1524DABF0F8C67AA58BA1992D6ED858522B9094490EBBB00E5E8AEFF7F40CD"


class RuntimeEventKind(str, Enum):
    FRAME_BEGIN = "frame_begin"
    CHECK_POST_HOLD_FAIL = "check_post_hold_fail"
    INPUT_DISPATCH = "input_dispatch"
    KEY_DOWN = "key_down"
    JUDGEMENT = "judgement"
    AUTO_FAIL_MISS = "auto_fail_miss"
    ADD_HIT = "add_hit"
    COUNTER_UPDATE = "counter_update"
    GAME_STOP = "game_stop"
    FRAME_END = "frame_end"


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    session_id: str
    sequence: int
    frame: int
    realtime_s: float
    kind: RuntimeEventKind
    no_fail: bool
    payload: tuple[tuple[str, object], ...]

    def value(self, key: str, default=None):
        return dict(self.payload).get(key, default)


@dataclass(frozen=True, slots=True)
class RuntimeCapture:
    evidence_kind: str
    dll_sha256: str | None
    events: tuple[RuntimeEvent, ...]

    @property
    def is_live(self) -> bool:
        return self.evidence_kind == "live_runtime"


def parse_runtime_jsonl(lines: Iterable[str]) -> RuntimeCapture:
    header = None
    events: list[RuntimeEvent] = []
    prior_sequence = -1
    prior_frame = -1
    prior_time = -math.inf
    for line_number, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if header is None:
            if row.get("type") != "stock_runtime_probe_header":
                raise ValueError("first JSONL row must be a probe header")
            header = row
            continue
        if row.get("type") != "event":
            raise ValueError(f"line {line_number}: expected event")
        sequence = int(row["sequence"])
        frame = int(row["frame"])
        realtime = float(row["realtime_s"])
        if sequence <= prior_sequence or frame < prior_frame or realtime < prior_time:
            raise ValueError("runtime log must be monotonic in sequence/frame/time")
        known = {"type", "session_id", "sequence", "frame", "realtime_s", "kind", "no_fail"}
        payload = tuple(sorted((key, value) for key, value in row.items() if key not in known))
        events.append(RuntimeEvent(
            str(row["session_id"]), sequence, frame, realtime,
            RuntimeEventKind(row["kind"]), bool(row["no_fail"]), payload,
        ))
        prior_sequence, prior_frame, prior_time = sequence, frame, realtime
    if header is None:
        raise ValueError("empty runtime log")
    if not events:
        raise ValueError("runtime log has no events")
    sessions = {event.session_id for event in events}
    if sessions != {str(header["session_id"])}:
        raise ValueError("header/event session mismatch")
    dll = header.get("assembly_csharp_sha256")
    if dll is not None:
        dll = str(dll).upper()
        if len(dll) != 64 or any(char not in "0123456789ABCDEF" for char in dll):
            raise ValueError("invalid DLL SHA256")
    return RuntimeCapture(str(header.get("evidence_kind", "unknown")), dll, tuple(events))


def load_runtime_jsonl(path: Path) -> RuntimeCapture:
    return parse_runtime_jsonl(path.read_text(encoding="utf-8").splitlines())


def audit_runtime_capture(capture: RuntimeCapture) -> dict:
    kinds = {event.kind for event in capture.events}
    required = {
        RuntimeEventKind.FRAME_BEGIN,
        RuntimeEventKind.CHECK_POST_HOLD_FAIL,
        RuntimeEventKind.INPUT_DISPATCH,
        RuntimeEventKind.KEY_DOWN,
        RuntimeEventKind.JUDGEMENT,
        RuntimeEventKind.ADD_HIT,
        RuntimeEventKind.FRAME_END,
    }
    missing = sorted(kind.value for kind in required - kinds)
    frame_orders: dict[int, list[str]] = {}
    for event in capture.events:
        frame_orders.setdefault(event.frame, []).append(event.kind.value)
    dll_match = capture.dll_sha256 == AUDITED_DLL_SHA256
    sufficiently_instrumented = not missing
    # A synthetic or exported fixture can prove parser behaviour only.
    if not capture.is_live:
        verdict = "RUNTIME_PARITY_UNVERIFIED"
        reason = "capture is not declared live runtime evidence"
    elif not dll_match or not sufficiently_instrumented:
        verdict = "RUNTIME_PARITY_UNVERIFIED"
        reason = "live evidence lacks audited DLL identity or required hook coverage"
    else:
        # Reference equality still needs a separate expected-transition input;
        # this parser never upgrades presence of logs into parity by itself.
        verdict = "RUNTIME_PARITY_UNVERIFIED"
        reason = "live trace is ingestible but has not been compared with a fixed reference trace"
    return {
        "verdict": verdict,
        "reason": reason,
        "evidence_kind": capture.evidence_kind,
        "dll_sha256": capture.dll_sha256,
        "audited_dll_match": dll_match,
        "event_count": len(capture.events),
        "missing_required_kinds": missing,
        "frame_orders": frame_orders,
        "student_visible_fields": [],
        "evaluator_only": True,
    }
