from __future__ import annotations

"""Compare the Phase 16 input sequence with the audited-DLL floor model.

All chart targets and repository judgements are posthoc evaluator evidence.
The Stock reference generates its own current/next-floor allocation and never
uses repository ``target_index`` to select a target.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.stock_floor_state_machine import (
    StockFloorStateMachine,
    StockRunMode,
    StockStateTransition,
    targets_from_playable_segment,
)
from dmdod.adofai.timing import compile_adofai


FORMAT = "dmdod-phase20b-stock-floor-state-audit-v0.1"
CHART = "01645 - First town of this Journey remix - Irin.adofai"
CHART_SHA256 = "4a71411d454d3b10a10cd98165a2e9cce99e4b14c310d1e6847836fdc68fc0e1"
ARM = "v15_midspin_twirl_checkpoint_on"
WINDOW_START_S = 3.15
WINDOW_END_S = 5.36
AUDITED_DLL_SHA256 = "0D1524DABF0F8C67AA58BA1992D6ED858522B9094490EBBB00E5E8AEFF7F40CD"
SOURCE_HASHES = {
    "scrMisc.cs": "5DD7696235ED26476A01D9417ACEE4A7BE91D19567A9BD83FA510DC4225F62A5",
    "scrPlayer.cs": "21326189390DB431A4BBE2EA9E608D67048509AA2138526E5AD5179DA08EFC47",
    "scrPlanet.cs": "2250FEB39218F386233227E1BD2007BD1A9360C62C12749EBE0B8EFC07618C0E",
}


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


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


def _transition(row: StockStateTransition) -> dict:
    return _jsonable(asdict(row))


def _logical_keydowns(window: dict, *, end_s: float) -> list[dict]:
    return [
        dict(edge)
        for frame in window["frames"]
        for edge in frame["logical_edges"]
        if edge["edge"] == "down" and float(edge["time_s"]) <= end_s
    ]


def _edge_key(time_s: float, contact: str) -> tuple[float, str]:
    return round(float(time_s), 9), str(contact)


def _local_ids(arm: dict, window: dict) -> dict[tuple[float, str], int]:
    """Recover the policy's own reservation ID; no target truth is used."""

    result: dict[tuple[float, str], int] = {}
    for frame in window["frames"]:
        for edge in frame["logical_edges"]:
            if edge["edge"] != "down":
                continue
            reservations = [
                row
                for row in frame["reservations_after_physics"]
                if edge["contact"] in row["axes"]
                and edge["contact"] in row["down_axes"]
            ]
            if len(reservations) != 1:
                raise RuntimeError("KeyDown must have exactly one active reservation")
            result[_edge_key(edge["time_s"], edge["contact"])] = int(
                reservations[0]["local_id"]
            )
    decision_ids = {int(row["local_id"]) for row in arm["decisions"]}
    if not set(result.values()).issubset(decision_ids):
        raise RuntimeError("reservation ID has no Fingering decision")
    return result


def _repository_rows(window: dict) -> dict[tuple[float, str], dict]:
    return {
        _edge_key(row["time_s"], row["key"]): row
        for row in window["judgement_trace"]
    }


def _run_no_fail(targets, keydowns, repository, local_ids) -> dict:
    machine = StockFloorStateMachine(targets, run_mode=StockRunMode.NO_FAIL)
    comparisons: list[dict] = []
    for edge in keydowns:
        transitions = machine.key_down(float(edge["time_s"]))
        input_rows = [row for row in transitions if row.input_time_s is not None]
        if len(input_rows) != 1:
            raise RuntimeError("one Stock input transition is required per KeyDown")
        if edge["time_s"] < WINDOW_START_S:
            continue
        stock = input_rows[0]
        key = _edge_key(edge["time_s"], edge["contact"])
        repo = repository[key]
        local_id = local_ids[key]
        comparisons.append(
            {
                "time_s": edge["time_s"],
                "contact": edge["contact"],
                "policy_local_id": local_id,
                "repository_target_ordinal": repo["target_index"],
                "repository_judgement": repo["margin"],
                "stock_target_ordinal": stock.target_ordinal,
                "stock_target_floor": stock.target_floor,
                "stock_judgement": stock.judgement.value,
                "stock_current_floor_before": stock.current_floor_before,
                "stock_current_floor_after": stock.current_floor_after,
                "target_equal": stock.target_ordinal == repo["target_index"],
                "judgement_equal": stock.judgement.value == repo["margin"],
                "intended_equal_stock": local_id == stock.target_ordinal,
                "automatic_before_input": [
                    _transition(row) for row in transitions if row.automatic_miss
                ],
            }
        )

    automatic = [
        _transition(row)
        for row in machine.trace
        if row.automatic_miss
        and WINDOW_START_S <= row.observed_time_s <= WINDOW_END_S
    ]
    mismatches = [row for row in comparisons if not row["intended_equal_stock"]]
    resync = next(
        (
            row
            for row in comparisons
            if mismatches
            and row["time_s"] > mismatches[-1]["time_s"]
            and row["intended_equal_stock"]
        ),
        None,
    )
    return {
        "mode": "no_fail",
        "comparison_count": len(comparisons),
        "judged_target_parity_count": sum(row["target_equal"] for row in comparisons),
        "judgement_parity_count": sum(row["judgement_equal"] for row in comparisons),
        "automatic_misses": automatic,
        "intention_backlog": {
            "count": len(mismatches),
            "first": mismatches[0] if mismatches else None,
            "last": mismatches[-1] if mismatches else None,
            "resynchronized": resync,
        },
        "event_37_90": [
            row for row in comparisons if 37 <= row["policy_local_id"] <= 90
        ],
        "checks": {
            "all_repository_targets_equal_stock": all(
                row["target_equal"] for row in comparisons
            ),
            "all_repository_categories_equal_stock": all(
                row["judgement_equal"] for row in comparisons
            ),
            "id38_is_consumed_by_event37": any(
                row["policy_local_id"] == 38
                and row["stock_target_ordinal"] == 37
                for row in comparisons
            ),
            "id89_is_consumed_by_event88": any(
                row["policy_local_id"] == 89
                and row["stock_target_ordinal"] == 88
                for row in comparisons
            ),
            "id90_resynchronizes_after_stock_auto_miss89": resync is not None
            and resync["policy_local_id"] == 90
            and any(
                row["target_ordinal"] == 89
                for row in resync["automatic_before_input"]
            ),
        },
    }


def _run_fail_anchor(targets, keydowns) -> dict:
    # The snapshot is evaluator-only and begins after Event 35's successful
    # landing. It isolates fail-mode semantics without claiming that an earlier
    # default-Stock run would have survived all prior Phase 16 abstentions.
    machine = StockFloorStateMachine(
        targets[36:],
        run_mode=StockRunMode.FAIL,
        initial_floor=targets[35].floor_index,
    )
    trace: list[dict] = []
    for edge in keydowns:
        if edge["time_s"] <= 3.174:
            continue
        trace.extend(_transition(row) for row in machine.key_down(edge["time_s"]))
        if machine.failed:
            break
    first_input = next(row for row in trace if row["input_time_s"] is not None)
    failure = next(row for row in trace if row["kind"] == "auto_miss_fail")
    return {
        "mode": "fail",
        "initialization": {
            "kind": "posthoc_evaluator_snapshot",
            "current_floor": targets[35].floor_index,
            "next_target_ordinal": 36,
            "claim_limit": "isolates the audited fail branch; not a full Stock run",
        },
        "trace": trace,
        "first_input": first_input,
        "failure": failure,
        "checks": {
            "id38_time_still_targets_event36": first_input["target_ordinal"] == 36,
            "id38_time_is_too_late": first_input["judgement"] == "too_late",
            "invalid_input_does_not_advance": not first_input["transitioned"],
            "post_floor_auto_miss_stops_play": failure["target_ordinal"] == 36
            and failure["failed"]
            and not failure["transitioned"],
        },
    }


def run(source: Path, train_dir: Path, decompile_dir: Path) -> dict:
    source_raw = source.read_bytes()
    data = json.loads(source_raw.decode("utf-8"))
    case = data["cases"][0]
    if case["chart"] != CHART or case["chart_sha256"] != CHART_SHA256:
        raise ValueError("fixed Phase 16 chart identity changed")
    if data.get("diagnostic_window_s") != [0.0, 6.0]:
        raise ValueError("fixed 0-6 second diagnostic window is required")

    chart_path = train_dir / CHART
    if _hash(chart_path) != CHART_SHA256:
        raise ValueError("fixed Train chart is absent or changed")
    for name, expected in SOURCE_HASHES.items():
        if _hash(decompile_dir / name).upper() != expected:
            raise ValueError(f"audited decompiled source changed: {name}")

    chart = compile_adofai(load_adofai(chart_path))
    segment = build_playable_segment(chart, end_s=6.0)
    targets = targets_from_playable_segment(segment)
    arm = case["arms"][ARM]
    window = arm["diagnostic_window_posthoc"]
    keydowns = _logical_keydowns(window, end_s=WINDOW_END_S)
    repository = _repository_rows(window)
    local_ids = _local_ids(arm, window)

    no_fail = _run_no_fail(targets, keydowns, repository, local_ids)
    fail = _run_fail_anchor(targets, keydowns)
    checks = {
        "no_fail": all(no_fail["checks"].values()),
        "fail": all(fail["checks"].values()),
        "phase16_window_is_ordinary_supported": all(
            targets[index].supported for index in range(35, 93)
        ),
    }
    verdict = (
        "STOCK_MODEL_BACKLOG_REPRODUCED"
        if all(checks.values())
        else "STOCK_STATE_MODEL_INCOMPLETE"
    )
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "scope": {
            "window_s": [WINDOW_START_S, WINDOW_END_S],
            "chart": CHART,
            "chart_sha256": CHART_SHA256,
            "arm": ARM,
            "production_code_changed": False,
            "truth_scope": "posthoc evaluator/reference-model only",
        },
        "evidence": {
            "phase16_source": str(source),
            "phase16_source_sha256": sha256(source_raw).hexdigest(),
            "audited_dll_sha256_from_phase19": AUDITED_DLL_SHA256,
            "decompiled_source_dir": str(decompile_dir),
            "decompiled_source_sha256": SOURCE_HASHES,
            "runtime_build_identity": "UNVERIFIED",
        },
        "checks": checks,
        "no_fail_comparison": no_fail,
        "fail_mode_snapshot": fail,
        "conclusion": {
            "phase16_repository_analogue": (
                "The continuing repository scorer matches the audited no-fail "
                "branch for every target and category in the fixed window."
            ),
            "backlog": (
                "After Event 36 is force-missed, ID38 is consumed by Event37; "
                "the one-event backlog persists through ID89 and ID90 resynchronizes "
                "only after Event89 is force-missed."
            ),
            "fail_mode": (
                "From the Event35 landing snapshot, default fail mode keeps Event36 "
                "current, classifies the next two inputs TooLate, then dies in "
                "CheckPostHoldFail before the following input."
            ),
            "stock_runtime_limit": (
                "These are audited-DLL reference results. The installed/live Stock "
                "build and its exact frame scheduling were not verified."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("reports/phase16-firsttown12-event89-causal-20261021-r2.json"),
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
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
    print(json.dumps({"verdict": result["verdict"], "output": str(args.output)}))
    return 0 if result["verdict"] == "STOCK_MODEL_BACKLOG_REPRODUCED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
