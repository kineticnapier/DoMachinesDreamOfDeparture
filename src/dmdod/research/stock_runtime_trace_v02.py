from __future__ import annotations

"""Ingest evaluator-only JSONL emitted by the Phase 31 runtime probe.

This module never exposes a Student observation.  Internal floor and counter
payloads remain evidence for reference-model comparison only.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable


REQUIRED_HOOKS = frozenset({
    "Simulated_PlayerControl_Update",
    "CheckPostHoldFail",
    "ProcessKeyInputs",
    "HitInputEvent",
    "SwitchChosen",
    "AddHit",
    "ShowHitText",
})


@dataclass(frozen=True, slots=True)
class TraceEvent:
    session_id: str
    sequence: int
    frame: int
    realtime_s: float
    kind: str
    no_fail: bool
    hook: str
    payload: tuple[tuple[str, object], ...]

    def value(self, key: str, default=None):
        return dict(self.payload).get(key, default)


@dataclass(frozen=True, slots=True)
class TraceSession:
    session_id: str
    evidence_kind: str
    dll_sha256: str | None
    game_build: str | None
    events: tuple[TraceEvent, ...]
    issues: tuple[str, ...]

    @property
    def is_live(self) -> bool:
        return self.evidence_kind == "live_runtime"


def parse_runtime_jsonl_v02(lines: Iterable[str]) -> tuple[TraceSession, ...]:
    sessions: list[TraceSession] = []
    header: dict[str, object] | None = None
    events: list[TraceEvent] = []
    issues: list[str] = []
    seen_sequences: set[int] = set()
    prior_sequence: int | None = None
    prior_frame: int | None = None
    prior_time: float | None = None

    def finish() -> None:
        nonlocal header, events, issues, seen_sequences, prior_sequence, prior_frame, prior_time
        if header is None:
            return
        if not events:
            issues.append("EMPTY_SESSION")
        sessions.append(TraceSession(
            session_id=str(header["session_id"]),
            evidence_kind=str(header.get("evidence_kind", "unknown")),
            dll_sha256=None if header.get("assembly_csharp_sha256") is None else str(header["assembly_csharp_sha256"]).upper(),
            game_build=None if header.get("game_build") is None else str(header["game_build"]),
            events=tuple(events),
            issues=tuple(issues),
        ))
        header, events, issues = None, [], []
        seen_sequences, prior_sequence, prior_frame, prior_time = set(), None, None, None

    for line_number, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number}: object required")
        if row.get("type") == "stock_runtime_probe_header":
            finish()
            if not str(row.get("session_id", "")):
                raise ValueError(f"line {line_number}: header session_id required")
            header = row
            continue
        if row.get("type") != "event" or header is None:
            raise ValueError(f"line {line_number}: event must follow a header")
        required = ("session_id", "sequence", "frame", "realtime_s", "kind", "no_fail", "hook")
        missing = [name for name in required if name not in row]
        if missing:
            raise ValueError(f"line {line_number}: missing {','.join(missing)}")
        if str(row["session_id"]) != str(header["session_id"]):
            raise ValueError(f"line {line_number}: session mismatch")
        sequence, frame, timestamp = int(row["sequence"]), int(row["frame"]), float(row["realtime_s"])
        if sequence < 0 or frame < 0 or not math.isfinite(timestamp):
            raise ValueError(f"line {line_number}: invalid sequence/frame/time")
        if sequence in seen_sequences:
            issues.append(f"DUPLICATE_SEQUENCE:{sequence}")
        elif prior_sequence is not None and sequence > prior_sequence + 1:
            issues.append(f"SEQUENCE_GAP:{prior_sequence + 1}-{sequence - 1}")
        elif prior_sequence is not None and sequence < prior_sequence:
            issues.append(f"SEQUENCE_REVERSAL:{prior_sequence}->{sequence}")
        if prior_frame is not None and frame < prior_frame:
            issues.append(f"FRAME_REVERSAL:{prior_frame}->{frame}")
        if prior_time is not None and timestamp < prior_time:
            issues.append(f"TIME_REVERSAL:{prior_time}->{timestamp}")
        known = {"type", "session_id", "sequence", "frame", "realtime_s", "kind", "no_fail", "hook"}
        events.append(TraceEvent(
            str(row["session_id"]), sequence, frame, timestamp, str(row["kind"]),
            bool(row["no_fail"]), str(row["hook"]),
            tuple(sorted((key, value) for key, value in row.items() if key not in known)),
        ))
        seen_sequences.add(sequence)
        prior_sequence, prior_frame, prior_time = sequence, frame, timestamp
    finish()
    if not sessions:
        raise ValueError("runtime trace contains no session")
    ids = [session.session_id for session in sessions]
    if len(ids) != len(set(ids)):
        raise ValueError("session_id must be unique across headers")
    return tuple(sessions)


def load_runtime_jsonl_v02(path: Path) -> tuple[TraceSession, ...]:
    return parse_runtime_jsonl_v02(path.read_text(encoding="utf-8-sig").splitlines())


def audit_trace_sessions(sessions: tuple[TraceSession, ...]) -> dict[str, object]:
    results: list[dict[str, object]] = []
    any_live = False
    all_complete = True
    for session in sessions:
        hooks = {event.hook for event in session.events}
        missing_hooks = sorted(REQUIRED_HOOKS - hooks)
        all_complete &= not missing_hooks and not session.issues
        any_live |= session.is_live
        no_fail_values = sorted({event.no_fail for event in session.events})
        keydown_pairs: list[dict[str, object]] = []
        auto_transitions: list[dict[str, object]] = []
        hit_text_delays: list[dict[str, object]] = []
        # Keep the earliest strictly later matching event for each event type.
        # A reverse pass avoids repeated tuple slicing and forward rescans.
        events = session.events
        next_judgement_or_miss: list[TraceEvent | None] = [None] * len(events)
        next_judgement: list[TraceEvent | None] = [None] * len(events)
        next_visible: list[TraceEvent | None] = [None] * len(events)
        later_judgement_or_miss: TraceEvent | None = None
        later_judgement: TraceEvent | None = None
        later_visible: TraceEvent | None = None
        for index in range(len(events) - 1, -1, -1):
            event = events[index]
            next_judgement_or_miss[index] = later_judgement_or_miss
            next_judgement[index] = later_judgement
            next_visible[index] = later_visible
            if event.kind in {"judgement", "auto_fail_miss"}:
                later_judgement_or_miss = event
            if event.kind == "judgement":
                later_judgement = event
            if event.kind in {"visible_judgement_signal", "counter_update"}:
                later_visible = event
        for index, event in enumerate(events):
            if event.kind == "key_down":
                judged = next_judgement_or_miss[index]
                keydown_pairs.append({
                    "keydown_sequence": event.sequence,
                    "keydown_frame": event.frame,
                    "judgement_sequence": None if judged is None else judged.sequence,
                    "judgement": None if judged is None else judged.value("judgement"),
                    "target_floor_evaluator_only": None if judged is None else judged.value("floor_evaluator_only"),
                })
            if event.kind == "auto_fail_miss":
                transitioned = next_judgement[index]
                auto_transitions.append({
                    "sequence": event.sequence,
                    "frame": event.frame,
                    "realtime_s": event.realtime_s,
                    "next_judgement_sequence": None if transitioned is None else transitioned.sequence,
                })
            if event.kind == "add_hit":
                visible = next_visible[index]
                hit_text_delays.append({
                    "add_hit_sequence": event.sequence,
                    "visible_sequence": None if visible is None else visible.sequence,
                    "delay_s": None if visible is None else visible.realtime_s - event.realtime_s,
                    "signal_kind": None if visible is None else visible.kind,
                })
        results.append({
            "session_id": session.session_id,
            "evidence_kind": session.evidence_kind,
            "event_count": len(session.events),
            "issues": list(session.issues),
            "no_fail_values": no_fail_values,
            "missing_hooks": missing_hooks,
            "hook_coverage": sorted(hooks),
            "auto_transitions": auto_transitions,
            "keydown_judgement_pairs": keydown_pairs,
            "hit_text_signal_delays": hit_text_delays,
            "visible_cumulative_counter_delay": "UNKNOWN_NOT_HOOKED",
        })
    return {
        "format": "dmdod-phase32-runtime-trace-ingestion-v0.2",
        "session_count": len(sessions),
        "live_runtime_present": any_live,
        "all_sessions_complete": all_complete,
        "sessions": results,
        "student_visible_fields": [],
        "evaluator_only": True,
        "runtime_parity": "RUNTIME_PARITY_UNVERIFIED",
    }
