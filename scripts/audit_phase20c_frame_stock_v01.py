from __future__ import annotations

"""Frame-driven Stock/Phase-16 comparison on the fixed First town window."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.stock_floor_state_machine import (
    StockFloorStateMachine,
    StockRunMode,
    targets_from_playable_segment,
)
from dmdod.adofai.stock_frame_state_machine import (
    StockFrameStateMachine,
    StockQueuedKeyDown,
    first_frame_strictly_after,
)
from dmdod.adofai.stock_judgement_reference import stock_angle_boundaries
from dmdod.adofai.timing import compile_adofai
from scripts.audit_phase20b_stock_floor_state_v01 import (
    ARM,
    CHART,
    CHART_SHA256,
    SOURCE_HASHES,
    WINDOW_END_S,
    WINDOW_START_S,
    _edge_key,
    _hash,
    _local_ids,
    _logical_keydowns,
    _repository_rows,
)


FORMAT = "dmdod-phase20c-frame-stock-audit-v0.1"
FRAME_DT_S = 0.010
FRAME_PHASE_S = 0.0


def _jsonable(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _transition(row) -> dict:
    return _jsonable(asdict(row))


def _frame_range(end_s: float) -> range:
    return range(0, int(round((end_s - FRAME_PHASE_S) / FRAME_DT_S)) + 1)


def _queue_all(frames: StockFrameStateMachine, keydowns: list[dict], prefix: str) -> None:
    for index, edge in enumerate(keydowns):
        frames.queue_keydown(
            StockQueuedKeyDown(
                input_id=f"{prefix}:{index}",
                occurred_at_s=float(edge["time_s"]),
                contact=str(edge["contact"]),
            )
        )


def _run_no_fail(targets, keydowns, repository, local_ids) -> dict:
    floor = StockFloorStateMachine(targets, run_mode=StockRunMode.NO_FAIL)
    frames = StockFrameStateMachine(
        floor, frame_dt_s=FRAME_DT_S, phase_s=FRAME_PHASE_S
    )
    _queue_all(frames, keydowns, "nf")
    comparisons: list[dict] = []
    automatic: list[dict] = []
    for frame_index in _frame_range(WINDOW_END_S):
        update = frames.update(frame_index)
        if update.frame_time_s >= WINDOW_START_S:
            automatic.extend(
                {
                    "frame_index": frame_index,
                    "frame_time_s": update.frame_time_s,
                    **_transition(row),
                }
                for row in update.automatic_transitions
            )
        for processed in update.processed_keydowns:
            if processed.keydown.occurred_at_s < WINDOW_START_S:
                continue
            stock = processed.input_transition
            if stock is None:
                raise RuntimeError("NoFail fixed window must produce an input transition")
            key = _edge_key(
                processed.keydown.occurred_at_s, processed.keydown.contact
            )
            repo = repository[key]
            local_id = local_ids[key]
            comparisons.append(
                {
                    "frame_index": frame_index,
                    "processed_at_s": update.frame_time_s,
                    "keydown_at_s": processed.keydown.occurred_at_s,
                    "contact": processed.keydown.contact,
                    "policy_local_id": local_id,
                    "repository_target_ordinal": repo["target_index"],
                    "repository_judgement": repo["margin"],
                    "stock_target_ordinal": stock.target_ordinal,
                    "stock_judgement": stock.judgement.value,
                    "stock_current_floor_before": stock.current_floor_before,
                    "stock_current_floor_after": stock.current_floor_after,
                    "target_equal": stock.target_ordinal == repo["target_index"],
                    "judgement_equal": stock.judgement.value == repo["margin"],
                    "intended_equal_stock": local_id == stock.target_ordinal,
                }
            )
    backlog = [row for row in comparisons if not row["intended_equal_stock"]]
    resync = next(
        row
        for row in comparisons
        if backlog
        and row["processed_at_s"] >= backlog[-1]["processed_at_s"]
        and row["intended_equal_stock"]
        and row["policy_local_id"] == 90
    )
    return {
        "comparison_count": len(comparisons),
        "target_parity_count": sum(row["target_equal"] for row in comparisons),
        "category_parity_count": sum(row["judgement_equal"] for row in comparisons),
        "automatic_transitions": automatic,
        "backlog": {
            "count": len(backlog),
            "first": backlog[0],
            "last": backlog[-1],
            "resynchronized": resync,
        },
        "event_37_90": [
            row for row in comparisons if 37 <= row["policy_local_id"] <= 90
        ],
    }


def _run_fail_snapshot(targets, keydowns) -> dict:
    floor = StockFloorStateMachine(
        targets[36:],
        run_mode=StockRunMode.FAIL,
        initial_floor=targets[35].floor_index,
    )
    frames = StockFrameStateMachine(
        floor, frame_dt_s=FRAME_DT_S, phase_s=FRAME_PHASE_S
    )
    focused = [edge for edge in keydowns if edge["time_s"] > 3.174]
    _queue_all(frames, focused, "fail")
    updates: list[dict] = []
    for frame_index in range(318, _frame_range(WINDOW_END_S).stop):
        update = frames.update(frame_index)
        if update.transitions:
            updates.append(
                {
                    "frame_index": frame_index,
                    "frame_time_s": update.frame_time_s,
                    "current_floor_before": update.current_floor_before,
                    "current_floor_after": update.current_floor_after,
                    "transitions": [_transition(row) for row in update.transitions],
                }
            )
        if update.failed:
            break
    failure = next(
        row
        for update in updates
        for row in update["transitions"]
        if row["kind"] == "auto_miss_fail"
    )
    return {
        "initialization": "posthoc Event35 landing snapshot",
        "updates": updates,
        "failure": failure,
    }


def _deadline(target, *, no_fail: bool) -> dict:
    boundaries = stock_angle_boundaries(
        target.approach_bpm,
        pitch=target.pitch,
        margin_scale=target.margin_scale,
    )
    threshold_deg = (
        boundaries.counted_deg * 1.01
        if no_fail else max(180.0, boundaries.counted_deg * 2.0)
    )
    crossing_s = target.due_s + threshold_deg / (
        3.0 * target.approach_bpm * target.pitch
    )
    frame_index, frame_time_s = first_frame_strictly_after(
        crossing_s, frame_dt_s=FRAME_DT_S, phase_s=FRAME_PHASE_S
    )
    return {
        "target_ordinal": target.ordinal,
        "due_s": target.due_s,
        "threshold_deg": threshold_deg,
        "strict_crossing_s": crossing_s,
        "first_virtual_update_index": frame_index,
        "first_virtual_update_s": round(frame_time_s, 12),
    }


def run(source: Path, train_dir: Path, decompile_dir: Path) -> dict:
    raw = source.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    case = data["cases"][0]
    if case["chart"] != CHART or case["chart_sha256"] != CHART_SHA256:
        raise ValueError("fixed Phase 16 chart identity changed")
    chart_path = train_dir / CHART
    if _hash(chart_path) != CHART_SHA256:
        raise ValueError("fixed Train chart is absent or changed")
    for name, expected in SOURCE_HASHES.items():
        if _hash(decompile_dir / name).upper() != expected:
            raise ValueError(f"audited decompiled source changed: {name}")

    chart = compile_adofai(load_adofai(chart_path))
    targets = targets_from_playable_segment(
        build_playable_segment(chart, end_s=6.0)
    )
    arm = case["arms"][ARM]
    window = arm["diagnostic_window_posthoc"]
    keydowns = _logical_keydowns(window, end_s=WINDOW_END_S)
    repository = _repository_rows(window)
    local_ids = _local_ids(arm, window)
    no_fail = _run_no_fail(targets, keydowns, repository, local_ids)
    fail = _run_fail_snapshot(targets, keydowns)
    deadlines = {
        "event36_no_fail": _deadline(targets[36], no_fail=True),
        "event36_fail": _deadline(targets[36], no_fail=False),
        "event89_no_fail": _deadline(targets[89], no_fail=True),
    }
    auto_by_target = {
        row["target_ordinal"]: row for row in no_fail["automatic_transitions"]
    }
    checks = {
        "target_parity_54_of_54": no_fail["target_parity_count"] == 54,
        "category_parity_54_of_54": no_fail["category_parity_count"] == 54,
        "backlog_count_50": no_fail["backlog"]["count"] == 50,
        "event36_auto_miss_at_3_25": auto_by_target[36]["frame_time_s"] == 3.25,
        "event89_auto_miss_at_5_32": auto_by_target[89]["frame_time_s"] == 5.32,
        "id90_processed_after_event89_auto_miss": (
            no_fail["backlog"]["resynchronized"]["processed_at_s"] == 5.32
        ),
        "fail_stops_at_3_32": fail["failure"]["observed_time_s"]
        == 3.3200000000000003,
    }
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdicts": [
            "FRAME_ACCURATE_REFERENCE_VALIDATED"
            if all(checks.values()) else "FRAME_SCHEDULING_CHANGES_PARITY"
        ],
        "scope": {
            "frame_dt_s": FRAME_DT_S,
            "frame_phase_s": FRAME_PHASE_S,
            "frame_contract": (
                "virtual 10ms Update at phase 0; CheckPostHoldFail at frame time "
                "before queued KeyDowns; KeyDown category uses physical timestamp"
            ),
            "unity_runtime_equivalence": "UNVERIFIED",
            "window_s": [WINDOW_START_S, WINDOW_END_S],
            "truth_scope": "posthoc evaluator/reference-model only",
        },
        "source": {
            "path": str(source),
            "sha256": sha256(raw).hexdigest(),
            "chart_sha256": CHART_SHA256,
            "decompiled_source_sha256": SOURCE_HASHES,
        },
        "checks": checks,
        "deadlines": deadlines,
        "no_fail": no_fail,
        "fail_snapshot": fail,
        "repository_first_state_difference": {
            "no_fail": (
                "none on sampled 10ms frames for judged target/category; repository "
                "does not expose a per-frame floor trace in the saved artifact"
            ),
            "fail": {
                "time_s": 3.25,
                "reason": (
                    "repository expires Event36 after due+65ms, while audited fail "
                    "mode retains Event36 until the >130ms post-floor threshold"
                ),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=Path("reports/phase16-firsttown12-event89-causal-20261021-r2.json"),
    )
    parser.add_argument(
        "--train-dir", type=Path,
        default=Path("data/DMDOD-v140-tuf-p11-p20/Train"),
    )
    parser.add_argument("--decompile-dir", type=Path, default=Path("F:/adofai-code"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source, args.train_dir, args.decompile_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"verdicts": result["verdicts"], "output": str(args.output)}))
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
