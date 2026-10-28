from __future__ import annotations

"""Compare live Stock probe traces with the Phase 20C reference evidence."""

from dataclasses import dataclass
import math
from typing import Mapping

from .stock_runtime_log_v01 import AUDITED_DLL_SHA256
from .stock_runtime_trace_v02 import REQUIRED_HOOKS, TraceEvent, TraceSession


def _norm_judgement(value: object) -> str | None:
    if value is None:
        return None
    text = "".join(char for char in str(value).lower() if char.isalnum())
    aliases = {
        "failmiss": "fail_miss",
        "tooearly": "too_early",
        "veryearly": "very_early",
        "earlyperfect": "early_perfect",
        "lateperfect": "late_perfect",
        "verylate": "very_late",
        "toolate": "too_late",
    }
    return aliases.get(text, text or None)


@dataclass(frozen=True, slots=True)
class RuntimeTransition:
    kind: str
    frame: int
    song_time_s: float | None
    realtime_s: float
    floor_before: int | None
    floor_after: int | None
    target_ordinal: int | None
    judgement: str | None
    source_sequences: tuple[int, ...]


def _integer(value: object) -> int | None:
    return None if value is None else int(value)


def _song_time(event: TraceEvent) -> float | None:
    value = event.value("song_time_evaluator_only")
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def extract_runtime_transitions(
    session: TraceSession,
    *,
    floor_to_target: Mapping[int, int] | None = None,
) -> tuple[RuntimeTransition, ...]:
    floor_map = {} if floor_to_target is None else dict(floor_to_target)
    result: list[RuntimeTransition] = []
    pending_keydown: TraceEvent | None = None
    pending_category: tuple[TraceEvent, str] | None = None
    pending_auto: TraceEvent | None = None
    for event in session.events:
        if event.kind == "key_down":
            pending_keydown = event
        if event.kind == "auto_fail_miss":
            pending_auto = event
            pending_category = (event, "fail_miss")
        elif event.kind in {"judgement", "add_hit"} and event.value("judgement") is not None:
            pending_category = (event, _norm_judgement(event.value("judgement")) or "")
        if event.kind == "judgement" and event.hook == "SwitchChosen" and event.value("hook_phase") == "postfix":
            before = _integer(event.value("floor_before_evaluator_only"))
            after = _integer(event.value("floor_after_evaluator_only"))
            category_event, category = pending_category or (event, None)
            is_auto = pending_auto is not None and pending_auto.frame == event.frame
            key_event = pending_keydown if pending_keydown is not None and pending_keydown.frame == event.frame else None
            source = tuple(row.sequence for row in (pending_auto if is_auto else None, key_event, category_event, event) if row is not None)
            transition_song_time = _song_time(event)
            if transition_song_time is None:
                transition_song_time = _song_time(category_event)
            result.append(RuntimeTransition(
                kind="auto_miss" if is_auto else "key_down",
                frame=event.frame,
                song_time_s=transition_song_time,
                realtime_s=event.realtime_s,
                floor_before=before,
                floor_after=after,
                target_ordinal=None if after is None else floor_map.get(after),
                judgement=category,
                source_sequences=source,
            ))
            pending_category = None
            if is_auto:
                pending_auto = None
            if key_event is not None:
                pending_keydown = None
    for event in session.events:
        if event.kind == "game_stop":
            floor = _integer(event.value("floor_evaluator_only"))
            result.append(RuntimeTransition(
                kind="game_stop", frame=event.frame, song_time_s=_song_time(event),
                realtime_s=event.realtime_s, floor_before=floor, floor_after=floor,
                target_ordinal=None, judgement=None, source_sequences=(event.sequence,),
            ))
    return tuple(sorted(result, key=lambda row: (row.realtime_s, row.source_sequences)))


def reference_transitions(phase20c: dict[str, object], *, no_fail: bool) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    if no_fail:
        arm = phase20c["no_fail"]
        for row in arm["automatic_transitions"]:
            rows.append({
                "kind": "auto_miss", "song_time_s": row["observed_time_s"],
                "target_ordinal": row["target_ordinal"], "judgement": row["judgement"],
                "floor_before": row["current_floor_before"], "floor_after": row["current_floor_after"],
            })
        for row in arm["event_37_90"]:
            rows.append({
                "kind": "key_down", "song_time_s": row["keydown_at_s"],
                "target_ordinal": row["stock_target_ordinal"], "judgement": row["stock_judgement"],
                "floor_before": row["stock_current_floor_before"], "floor_after": row["stock_current_floor_after"],
            })
    else:
        for update in phase20c["fail_snapshot"]["updates"]:
            for row in update["transitions"]:
                if row["kind"] == "ignored_after_fail":
                    continue
                rows.append({
                    "kind": "auto_miss" if row["automatic_miss"] else "key_down",
                    "song_time_s": row["observed_time_s"], "target_ordinal": row["target_ordinal"],
                    "judgement": row["judgement"], "floor_before": row["current_floor_before"],
                    "floor_after": row["current_floor_after"],
                })
        failure = phase20c["fail_snapshot"].get("failure")
        if failure:
            rows.append({
                "kind": "game_stop", "song_time_s": failure["observed_time_s"],
                "target_ordinal": None, "judgement": None,
                "floor_before": failure["current_floor_before"], "floor_after": failure["current_floor_after"],
            })
    return tuple(sorted(rows, key=lambda row: float(row["song_time_s"])))


def compare_runtime_to_reference(
    session: TraceSession,
    phase20c: dict[str, object],
    *,
    floor_to_target: Mapping[int, int] | None = None,
    song_time_offset_s: float | None = None,
    song_time_tolerance_s: float | None = None,
) -> dict[str, object]:
    """Compare evaluator-only transitions, failing closed without time alignment.

    Offset and tolerance must be independently calibrated against live timing
    evidence before use. The virtual 10ms Phase 20C grid is not calibration.
    """
    if (song_time_offset_s is None) != (song_time_tolerance_s is None):
        raise ValueError("song-time offset and tolerance must be supplied together")
    if song_time_offset_s is not None:
        if not math.isfinite(song_time_offset_s):
            raise ValueError("song-time offset must be finite")
        if not math.isfinite(song_time_tolerance_s) or song_time_tolerance_s < 0:
            raise ValueError("song-time tolerance must be finite and non-negative")
    no_fail_values = {event.no_fail for event in session.events}
    if len(no_fail_values) != 1:
        return {"verdict": "RUNTIME_PARITY_UNVERIFIED", "reason": "session mixes NoFail modes", "comparisons": []}
    no_fail = next(iter(no_fail_values))
    observed = extract_runtime_transitions(session, floor_to_target=floor_to_target)
    expected = reference_transitions(phase20c, no_fail=no_fail)
    comparisons: list[dict[str, object]] = []
    explicit_mismatch = False
    complete = len(observed) == len(expected) and bool(observed)
    for index in range(max(len(observed), len(expected))):
        actual = observed[index] if index < len(observed) else None
        wanted = expected[index] if index < len(expected) else None
        row: dict[str, object] = {"index": index, "observed": None, "reference": wanted, "fields": {}}
        if actual is not None:
            row["observed"] = {
                "kind": actual.kind, "frame": actual.frame, "song_time_s": actual.song_time_s,
                "realtime_s": actual.realtime_s, "target_ordinal": actual.target_ordinal,
                "judgement": actual.judgement, "floor_before": actual.floor_before,
                "floor_after": actual.floor_after, "source_sequences": list(actual.source_sequences),
            }
        if actual is None or wanted is None:
            row["fields"] = {"row": "MISSING"}
            explicit_mismatch = True
        else:
            fields: dict[str, str] = {}
            for name in ("kind", "judgement", "floor_before", "floor_after", "target_ordinal"):
                left, right = getattr(actual, name), wanted[name]
                if left is None or right is None:
                    fields[name] = "UNKNOWN"
                    complete = False
                elif left == right:
                    fields[name] = "MATCH"
                else:
                    fields[name] = "MISMATCH"
                    explicit_mismatch = True
            reference_time = wanted.get("song_time_s")
            if actual.song_time_s is None or reference_time is None:
                fields["song_time_s"] = "UNKNOWN"
                complete = False
            else:
                try:
                    expected_time = float(reference_time)
                except (TypeError, ValueError):
                    expected_time = math.nan
                if not math.isfinite(expected_time):
                    fields["song_time_s"] = "UNKNOWN"
                    complete = False
                else:
                    delta = actual.song_time_s - expected_time
                    row["song_time_delta_s"] = delta
                    if song_time_offset_s is None:
                        # An observed timestamp is not evidence of parity.
                        fields["song_time_s"] = "UNKNOWN"
                        complete = False
                    elif abs(delta - song_time_offset_s) <= song_time_tolerance_s:
                        fields["song_time_s"] = "MATCH"
                    else:
                        fields["song_time_s"] = "MISMATCH"
                        explicit_mismatch = True
            row["fields"] = fields
        comparisons.append(row)
    hooks = {event.hook for event in session.events}
    live_qualified = (
        session.is_live
        and session.dll_sha256 == AUDITED_DLL_SHA256
        and not session.issues
        and REQUIRED_HOOKS <= hooks
    )
    if live_qualified and explicit_mismatch:
        verdict, reason = "RUNTIME_PARITY_MISMATCH", "live qualified evidence contains an explicit state/category/order/time mismatch"
    elif live_qualified and complete and not explicit_mismatch:
        verdict, reason = "RUNTIME_PARITY_CONFIRMED", "all declared reference transitions and fields match live evidence"
    else:
        verdict, reason = "RUNTIME_PARITY_UNVERIFIED", "evidence is synthetic, incomplete, unqualified, or contains UNKNOWN fields"
    return {
        "verdict": verdict,
        "reason": reason,
        "session_id": session.session_id,
        "no_fail": no_fail,
        "live_qualified": live_qualified,
        "reference_count": len(expected),
        "observed_count": len(observed),
        "complete": complete,
        "explicit_mismatch": explicit_mismatch,
        "comparisons": comparisons,
        "uses_virtual_10ms_as_runtime_time": False,
        "song_time_alignment": {
            "offset_s": song_time_offset_s,
            "tolerance_s": song_time_tolerance_s,
            "explicit_alignment_parameters_supplied": song_time_offset_s is not None,
        },
        "student_visible_fields": [],
    }
