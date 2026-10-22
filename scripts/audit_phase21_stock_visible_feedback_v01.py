from __future__ import annotations

"""Replay Phase 20C events through the visible-counter observation boundary."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmdod.features.judgement_observer_v01 import JudgementObserver
from dmdod.features.stock_visible_judgement_feedback_v01 import (
    CounterVisibilityEvidence,
    StockJudgementNotification,
    VisibleCounterContract,
    VisibleCounterRenderer,
)


FORMAT = "dmdod-phase21-stock-visible-feedback-v0.1"
CATEGORIES = ("late_perfect", "perfect", "very_late", "fail_miss")


def _contract(*, include_fail_miss: bool) -> VisibleCounterContract:
    categories = CATEGORIES if include_fail_miss else CATEGORIES[:-1]
    return VisibleCounterContract.build(
        labels=categories,
        category_to_label={category: category for category in categories},
        evidence={
            category: (
                CounterVisibilityEvidence.TEST_FIXTURE_ONLY
                if category == "fail_miss"
                else CounterVisibilityEvidence.DLL_INTERNAL_COUNT_ONLY
            )
            for category in categories
        },
    )


def _visible_replay(source: dict[str, object], *, include_fail_miss: bool) -> dict[str, object]:
    renderer = VisibleCounterRenderer(
        _contract(include_fail_miss=include_fail_miss), capture_latency_s=0.010
    )
    observer = JudgementObserver(renderer.contract.labels)
    observations: list[dict[str, object]] = []
    baseline = observer.observe(renderer.capture(3.24))
    observations.append({"displayed_at_s": 3.24, "observed_at_s": baseline.observed_at_s, "status": baseline.status.value, "deltas": {}})

    events: list[tuple[float, str, str, int | None]] = []
    no_fail = source["no_fail"]
    assert isinstance(no_fail, dict)
    for row in no_fail["automatic_transitions"]:
        events.append((float(row["frame_time_s"]), str(row["judgement"]), "automatic", int(row["target_ordinal"])))
    for row in no_fail["event_37_90"]:
        events.append((float(row["processed_at_s"]), str(row["stock_judgement"]), "keydown", int(row["policy_local_id"])))
    events.sort(key=lambda item: (item[0], 0 if item[2] == "automatic" else 1))

    by_time: dict[float, list[tuple[str, str, int | None]]] = {}
    invisible: list[dict[str, object]] = []
    for when, category, source_kind, identity in events:
        by_time.setdefault(when, []).append((category, source_kind, identity))
    for displayed in sorted(by_time):
        for category, source_kind, identity in by_time[displayed]:
            diagnostic = renderer.notify(StockJudgementNotification(displayed, category))
            if not diagnostic.accepted:
                invisible.append({"occurred_at_s": displayed, "category": category, "source": source_kind, "identity": identity})
        delta = observer.observe(renderer.capture(displayed))
        if delta.total_increment:
            observations.append({
                "displayed_at_s": displayed,
                "observed_at_s": delta.observed_at_s,
                "event_interval_s": delta.event_interval_s,
                "status": delta.status.value,
                "deltas": {key: value for key, value in delta.deltas if value},
                "within_interval_order_known": delta.within_interval_order_known,
            })
    return {"observations": observations, "invisible_events": invisible}


def run(input_path: Path) -> dict[str, object]:
    source = json.loads(input_path.read_text(encoding="utf-8"))
    fixture = _visible_replay(source, include_fail_miss=True)
    conservative = _visible_replay(source, include_fail_miss=False)
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdicts": ["STOCK_VISIBLE_FEEDBACK_PARTIAL"],
        "source": str(input_path),
        "evidence_boundary": {
            "dll_verified": ["scrMarginTracker.AddHit mutates internal counts synchronously", "ShowHitText is invoked by judgement paths", "reset and checkpoint rollback mutate internal counts"],
            "user_confirmed_live": ["judgement-specific cumulative counters are visible", "a corresponding counter changes when judgement is shown"],
            "unverified": ["exact visible label schema", "automatic no-KeyDown FailMiss visibility", "render-to-capture latency", "same-frame ordering on the visible surface"],
        },
        "fixture_all_categories": fixture,
        "conservative_failmiss_unknown": conservative,
        "phase16_milestones": {
            "event36_automatic_transition_internal_s": 3.25,
            "fixture_failmiss_observed_s": 3.26,
            "id38_physical_keydown_s": 3.260999999999752,
            "id38_very_late_observed_s": 3.28,
            "warning_before_id38_physical_ms_if_failmiss_visible": 0.999999999752,
            "event89_automatic_transition_internal_s": 5.32,
            "id90_physical_keydown_s": 5.318999999999709,
            "fixture_event89_failmiss_observed_s": 5.33,
        },
        "causal_conclusions": {
            "category_delta_only": "contains no floor/target identity or hidden cursor",
            "id38": "FailMiss fixture would arrive only about 1 ms before the physical KeyDown; with unknown FailMiss visibility it is first observed as VeryLate after the KeyDown",
            "event89": "the automatic transition is observed after ID90's already-issued physical KeyDown",
            "safe_44ms_intervention": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("reports/phase20c-frame-stock-state-20261022-r1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {args.output}")
    result = run(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdicts": result["verdicts"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

