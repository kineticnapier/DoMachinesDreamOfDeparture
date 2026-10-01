from __future__ import annotations

"""Execution-only telemetry for v1.1 round candidate rejection.

This module does not change guards, trust alphas, candidate ranking, optimizer
updates, or checkpoint semantics. It wraps the already-installed fast line
search and records where each interpolation candidate dies, plus the concrete
guard failure events that caused those deaths.
"""

from collections import Counter

import train_real_chart_v080 as v080
import train_real_chart_v080_fast as v080_fast


TELEMETRY_VERSION = "v110-round-rejection-telemetry-v1"

_ORIGINAL_FAST_LINE_SEARCH = None
_INSTALLED = False
_ANCHOR_GUARD_DEPTH = 0
_LINE_SEARCHES = 0
_CANDIDATES = 0
_COMPLETE_CANDIDATES = 0
_CHOSEN_UPDATES = 0
_TERMINAL_STAGES: Counter[str] = Counter()
_GUARD_FAILURES: Counter[str] = Counter()


def _reason_bucket(reason: str) -> str:
    text = str(reason).strip().casefold()
    if "safe->overload" in text or "both overloaded" in text:
        return "overload"
    if "hit regression" in text:
        return "hit"
    if "xacc regression" in text:
        return "XAcc"
    if "early regression" in text:
        return "early"
    if "not better" in text or "no accuracy-first improvement" in text:
        return "not-better"
    if not text:
        return "unknown"
    return text.replace(" ", "-")


def _anchor_stage(segment) -> str:
    role = str(getattr(segment, "role", ""))
    return "start-micro" if role.startswith("start-micro-") else "anchor"


def _terminal_stage(candidate, *, validation_count: int, anchor_count: int) -> str:
    if not candidate.train_decision.accepted:
        return "train"
    if (
        len(candidate.validation_decisions) != int(validation_count)
        or any(not decision.accepted for decision in candidate.validation_decisions)
    ):
        return "validation"
    if (
        len(candidate.anchor_decisions) != int(anchor_count)
        or any(not decision.accepted for decision in candidate.anchor_decisions)
    ):
        return "anchor"
    return "complete"


def _record_failure(stage: str, reason: str) -> None:
    _GUARD_FAILURES[f"{stage}:{_reason_bucket(reason)}"] += 1


def _telemetry_line_search(*args, **kwargs):
    global _ANCHOR_GUARD_DEPTH
    global _LINE_SEARCHES, _CANDIDATES, _COMPLETE_CANDIDATES, _CHOSEN_UPDATES

    assert _ORIGINAL_FAST_LINE_SEARCH is not None

    validation_references = tuple(kwargs.get("validation_references", ()))
    anchor_references = tuple(kwargs.get("anchor_references", ()))
    anchor_segments = tuple(kwargs.get("anchor_segments", ()))
    anchor_stage_by_reference = {
        id(reference): _anchor_stage(segment)
        for reference, segment in zip(anchor_references, anchor_segments)
    }

    original_train_guard = v080.v063.v062.v061._safety_guard
    original_validation_guard = v080.v062._validation_guard
    original_anchor_guard = v080.v064._anchor_guard

    def train_guard(best, candidate):
        decision = original_train_guard(best, candidate)
        if not decision.accepted:
            _record_failure("train", decision.reason)
        return decision

    def validation_guard(reference, candidate):
        decision = original_validation_guard(reference, candidate)
        # _anchor_guard delegates to _validation_guard internally. Suppress that
        # nested observation so one anchor failure is not also reported as a
        # validation failure.
        if _ANCHOR_GUARD_DEPTH == 0 and not decision.accepted:
            _record_failure("validation", decision.reason)
        return decision

    def anchor_guard(reference, candidate):
        global _ANCHOR_GUARD_DEPTH
        _ANCHOR_GUARD_DEPTH += 1
        try:
            decision = original_anchor_guard(reference, candidate)
        finally:
            _ANCHOR_GUARD_DEPTH -= 1
        if not decision.accepted:
            stage = anchor_stage_by_reference.get(id(reference), "anchor")
            _record_failure(stage, decision.reason)
        return decision

    v080.v063.v062.v061._safety_guard = train_guard
    v080.v062._validation_guard = validation_guard
    v080.v064._anchor_guard = anchor_guard
    try:
        result = _ORIGINAL_FAST_LINE_SEARCH(*args, **kwargs)
    finally:
        v080.v063.v062.v061._safety_guard = original_train_guard
        v080.v062._validation_guard = original_validation_guard
        v080.v064._anchor_guard = original_anchor_guard

    chosen, candidates, _chosen_state = result
    _LINE_SEARCHES += 1
    _CANDIDATES += len(candidates)
    validation_count = len(validation_references)
    anchor_count = len(anchor_references)
    for candidate in candidates:
        stage = _terminal_stage(
            candidate,
            validation_count=validation_count,
            anchor_count=anchor_count,
        )
        _TERMINAL_STAGES[stage] += 1
        if stage == "complete":
            _COMPLETE_CANDIDATES += 1
    if chosen is not None:
        _CHOSEN_UPDATES += 1
    return result


def install_rejection_telemetry() -> None:
    """Wrap the currently installed fast/timed line search without changing it."""

    global _ORIGINAL_FAST_LINE_SEARCH, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_FAST_LINE_SEARCH = v080_fast._fast_line_search
    v080_fast._fast_line_search = _telemetry_line_search
    _INSTALLED = True
    print(f"round-rejection-telemetry={TELEMETRY_VERSION}")


def rejection_telemetry() -> dict:
    return {
        "line_searches": int(_LINE_SEARCHES),
        "candidates": int(_CANDIDATES),
        "complete_candidates": int(_COMPLETE_CANDIDATES),
        "chosen_updates": int(_CHOSEN_UPDATES),
        "terminal_stages": dict(_TERMINAL_STAGES),
        "guard_failures": dict(_GUARD_FAILURES),
    }


def print_rejection_telemetry() -> None:
    stats = rejection_telemetry()
    print("=== Round rejection telemetry ===")
    print(
        f"line-searches={stats['line_searches']} candidates={stats['candidates']} "
        f"complete={stats['complete_candidates']} chosen={stats['chosen_updates']}"
    )

    terminal = stats["terminal_stages"]
    if terminal:
        ordered = ("train", "validation", "anchor", "complete")
        pieces = [f"{name}={terminal.get(name, 0)}" for name in ordered]
        extras = sorted(set(terminal) - set(ordered))
        pieces.extend(f"{name}={terminal[name]}" for name in extras)
        print("terminal-stage: " + " ".join(pieces))
    else:
        print("terminal-stage: none")

    failures = stats["guard_failures"]
    if failures:
        print("guard-fail-events:")
        for name, count in sorted(failures.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {name:<28} {count}")
    else:
        print("guard-fail-events: none")
