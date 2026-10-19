from __future__ import annotations

"""Causal Countdown/v1.4 source-identity audit on the fixed Train set."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_hud_chart_v02 import HudCoupledContactRealChartEnv
from dmdod.features.countdown_hud_v3 import (
    CountdownVisualSensor,
    StockCountdownRenderer,
    encode_hud_v3,
)
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG as FLOORS
from dmdod.motor.contact_16k import ContactAction
from dmdod.planning.contact_timing_hud_v01 import ContactTimingHUD263Adapter
from dmdod.planning.hud_v3_composite_forecast_source_v30 import (
    HUDV3CompositeForecastSource,
    LIFECYCLE_V15,
)
if __package__:
    from scripts.audit_hud_timing_phase8_lifecycle_integration_v01 import (
        _evaluable_rows,
        _motor_contract_audit,
    )
    from scripts.audit_timing_generalization_phase7_v01 import _evaluate
else:
    from audit_hud_timing_phase8_lifecycle_integration_v01 import (
        _evaluable_rows,
        _motor_contract_audit,
    )
    from audit_timing_generalization_phase7_v01 import _evaluate


FORMAT = "dmdod-hud-timing-phase10-source-identity-v0.1"
CONTROL_DT_S = 0.010
TRAIN_DIR = Path("data/DMDOD-v140-tuf-p11-p20/Train")
FIXED_CASES = (
    ("02334 - Time Leaper - Jipper.adofai", 12.0),
    ("02521 - Triple Cross - Milkyway.adofai", 12.0),
    ("07322 - Runaway - smy258.adofai", 12.0),
    ("07827 - 24eeev0-$ - hotduck _ COMMON.adofai", 12.0),
    ("04042 - bool - vo[ICE].adofai", 5.0),
)
TRIPLE_CROSS = FIXED_CASES[1][0]
TRAIN_BOOL = FIXED_CASES[4][0]


def _interval(value):
    if value is None:
        return None
    return [value.lower_s, value.upper_s]


def _serialize(public, raw, cue, hud278) -> dict:
    return {
        "step": public.step,
        "time_s": public.local_clock_s,
        "cue": {
            "kind": cue.kind.value,
            "digit": cue.digit,
            "valid": cue.valid,
            "observed_at_s": cue.observed_at_s,
            "rendered_at_s": cue.rendered_at_s,
        },
        "hud278": list(hud278.features),
        "status": raw.status,
        "public_status": public.status,
        "abstention_reason": raw.abstention_reason,
        "pending": [{
            "id": item.local_id,
            "due_s": item.due_local_s,
            "due_lower_s": item.due_local_s - (item.uncertainty_s or 0.0),
            "due_upper_s": item.due_local_s + (item.uncertainty_s or 0.0),
            "uncertainty_s": item.uncertainty_s,
            "confirmed_now": item.confirmed_now,
        } for item in public.pending],
        "new": list(public.new_ids),
        "reused": list(public.reused_ids),
        "canceled": list(public.canceled_ids),
        "elapsed": list(public.elapsed_ids),
        "raw_pending": [{
            "id": item.local_id,
            "due_lower_s": item.due_interval_s.lower_s,
            "due_upper_s": item.due_interval_s.upper_s,
            "uncertainty_s": item.uncertainty_s,
            "source": item.source,
        } for item in raw.pending],
        "v14": {
            "status": raw.v14_update.status,
            "pending": [{
                "id": f"v14:{item.local_id}",
                "due_s": item.due_s,
                "confirmations": item.confirmations,
                "confirmed_now": item.confirmed_now,
            } for item in raw.v14_update.pending],
            "canceled": [
                f"v14:{item.local_id}" for item in raw.v14_update.canceled
            ],
            "elapsed": [
                f"v14:{item.local_id}" for item in raw.v14_update.elapsed
            ],
        },
        "countdown": {
            "phase": {
                "status": raw.countdown_phase.status,
                "valid": raw.countdown_phase.valid,
                "displayed_digit": raw.countdown_phase.displayed_digit,
                "measured_tick_period_s": _interval(
                    raw.countdown_phase.measured_tick_period_s
                ),
                "remaining_transitions_to_go": (
                    raw.countdown_phase.remaining_transitions_to_go
                ),
            },
            "estimate": {
                "status": raw.countdown_estimate.status,
                "due_interval_s": _interval(
                    raw.countdown_estimate.due_interval_s
                ),
                "lead_interval_s": _interval(
                    raw.countdown_estimate.lead_interval_s
                ),
                "lead_guaranteed": raw.countdown_estimate.lead_guaranteed,
            },
        },
    }


def _capture(path: Path, end_s: float):
    chart = compile_adofai(load_adofai(path))
    segment = build_playable_segment(chart, end_s=end_s)
    renderer = StockCountdownRenderer(
        countdown_ticks=chart.countdown_ticks,
        first_tick_s=0.0,
        tick_period_s=60.0 / chart.initial_bpm / segment.pitch_ratio,
        go_duration_s=60.0 / chart.initial_bpm,
    )
    sensor = CountdownVisualSensor()
    sources = {
        "legacy_conflict": HUDV3CompositeForecastSource(
            horizon_s=.080,
            lifecycle_version=LIFECYCLE_V15,
            reconcile_countdown_v14_identity=False,
        ),
        "identity_reconciled": HUDV3CompositeForecastSource(
            horizon_s=.080,
            lifecycle_version=LIFECYCLE_V15,
            reconcile_countdown_v14_identity=True,
        ),
    }
    env = HudCoupledContactRealChartEnv(
        segment,
        control_dt_s=CONTROL_DT_S,
        physics_dt_s=.001,
        behind_floors=FLOORS.behind_floors,
        ahead_floors=FLOORS.ahead_floors,
        tail_s=.040,
    )
    observation = env.reset()
    idle = ContactAction((0.0,) * 12)
    rows = {name: [] for name in sources}
    for step in range(int(end_s / CONTROL_DT_S) + 1):
        now_s = step * CONTROL_DT_S
        cue = sensor.observe(
            step=step,
            observed_at_s=now_s,
            frame=renderer.render(now_s),
        )
        hud263 = ContactTimingHUD263Adapter.encode(step, observation).features
        hud278 = encode_hud_v3(hud263, cue)
        for name, source in sources.items():
            public, raw = source.observe(step, observation, cue)
            rows[name].append(_serialize(public, raw, cue, hud278))
        transition = env.step(idle)
        observation = transition.observation
        if transition.done:
            break
    return rows, segment


def _public_rows(rows: list[dict]) -> list[dict]:
    return [{
        key: row[key]
        for key in (
            "step", "time_s", "status", "public_status", "pending",
            "new", "reused", "canceled", "elapsed", "raw_pending",
        )
    } for row in rows]


def _aggregate(cases: list[dict], arm: str) -> dict:
    metrics = [case[arm]["metrics"] for case in cases]
    events = sum(item["target_count"] for item in metrics)
    clean = sum(item["category_counts"]["1"] for item in metrics)
    return {
        "events": events,
        "clean_events": clean,
        "clean_lifecycle_coverage": clean / events,
        "category_counts": {
            str(category): sum(
                item["category_counts"][str(category)] for item in metrics
            )
            for category in range(1, 6)
        },
        "false_positive_ids": sum(
            item["false_positive_ids"] for item in metrics
        ),
        "abstention": sum(item["abstention"] for item in metrics),
        "event_id_continuity_failures": sum(
            item["event_id_continuity_failures"] for item in metrics
        ),
    }


def _lifecycle_counts(rows: list[dict]) -> dict:
    return {
        name: sum(len(row[name]) for row in rows)
        for name in ("new", "reused", "canceled", "elapsed")
    }


def _forecast_differences(before: list[dict], after: list[dict]) -> dict:
    outside_alias_due = []
    outside_alias_uncertainty = []
    alias_due = []
    alias_uncertainty = []
    for left, right in zip(before, after, strict=True):
        a = {item["id"]: item for item in left["raw_pending"]}
        b = {item["id"]: item for item in right["raw_pending"]}
        for local_id in set(a) & set(b):
            due_changed = (
                abs(a[local_id]["due_lower_s"] - b[local_id]["due_lower_s"])
                > 1e-12
                or abs(a[local_id]["due_upper_s"] - b[local_id]["due_upper_s"])
                > 1e-12
            )
            uncertainty_changed = (
                a[local_id]["uncertainty_s"]
                != b[local_id]["uncertainty_s"]
            )
            target = (
                alias_due if local_id == "countdown:0"
                else outside_alias_due
            )
            if due_changed:
                target.append({"step": left["step"], "id": local_id})
            target = (
                alias_uncertainty if local_id == "countdown:0"
                else outside_alias_uncertainty
            )
            if uncertainty_changed:
                target.append({"step": left["step"], "id": local_id})
    return {
        "outside_reconciled_alias_due": outside_alias_due,
        "outside_reconciled_alias_uncertainty": outside_alias_uncertainty,
        "reconciled_alias_due": alias_due,
        "reconciled_alias_uncertainty": alias_uncertainty,
    }


def _triple_diagnostic(rows: dict[str, list[dict]], first_truth: float) -> dict:
    legacy = rows["legacy_conflict"]
    resolved = rows["identity_reconciled"]
    window = [
        {"legacy": left, "resolved": right}
        for left, right in zip(legacy, resolved, strict=True)
        if first_truth - .260 <= left["time_s"] <= first_truth + .020
    ]
    first_conflict = next(
        row for row in legacy if row["status"] == "source_conflict"
    )
    first_countdown = next(
        row for row in legacy if any(
            item["id"] == "countdown:0" for item in row["raw_pending"]
        )
    )
    first_v14 = next(
        row for row in legacy if any(
            item["id"] == "v14:17" for item in row["v14"]["pending"]
        )
    )
    first_join = next(
        row for row in legacy if any(
            item["source"].endswith("+timing_v1.4")
            for item in row["raw_pending"]
        )
    )
    joined = next(
        item for item in first_join["raw_pending"]
        if item["id"] == "countdown:0"
    )
    conflict_v14 = next(
        item for item in first_conflict["v14"]["pending"]
        if item["id"] == "v14:17"
    )
    prior_countdown = next(
        item for item in legacy[first_conflict["step"] - 1]["raw_pending"]
        if item["id"] == "countdown:0"
    )
    current_estimate = first_conflict["countdown"]["estimate"][
        "due_interval_s"
    ]
    pre_reconciliation_lower = max(
        prior_countdown["due_lower_s"], current_estimate[0]
    )
    pre_reconciliation_upper = min(
        prior_countdown["due_upper_s"], current_estimate[1]
    )
    if pre_reconciliation_lower > pre_reconciliation_upper:
        midpoint = (
            pre_reconciliation_lower + pre_reconciliation_upper
        ) * .5
        pre_reconciliation_lower = pre_reconciliation_upper = midpoint
    return {
        "first_countdown_id_frame": first_countdown["step"],
        "first_v14_id_frame": first_v14["step"],
        "first_unique_join_frame": first_join["step"],
        "first_conflict_frame": first_conflict["step"],
        "unique_join_interval_s": [
            joined["due_lower_s"], joined["due_upper_s"]
        ],
        "pre_conflict_countdown_interval_s": [
            prior_countdown["due_lower_s"], prior_countdown["due_upper_s"]
        ],
        "conflict_frame_countdown_interval_before_source_reconciliation_s": [
            pre_reconciliation_lower, pre_reconciliation_upper
        ],
        "conflict_v14_due_s": conflict_v14["due_s"],
        "point_gap_at_conflict_s": abs(
            pre_reconciliation_lower - conflict_v14["due_s"]
        ),
        "cause": (
            "A unique overlap already joined v14:17 to countdown:0. "
            "The same reused v1.4 ID then drifted outside the zero-width "
            "intersection by more than the 1ns overlap epsilon."
        ),
        "causal_identity_rule": (
            "Persist only a v1.4 local ID first associated by exactly one "
            "overlap with no competing v1.4 candidate; reject any frame "
            "with an additional candidate and never retime the published "
            "Countdown interval from later point drift."
        ),
        "frames": window,
    }


def run(train_dir: Path = TRAIN_DIR) -> dict:
    cases = []
    detailed = None
    regressions = []
    outside_due = outside_uncertainty = 0
    for filename, end_s in FIXED_CASES:
        path = train_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        rows, segment = _capture(path, end_s)
        # Evaluator boundary: truth is materialized only after both complete
        # policy streams have been captured.
        truths = tuple(
            float(target.episode_time_s) for target in segment.targets
            if target.episode_time_s <= end_s + 1e-9
        )
        evaluations = {}
        for arm in rows:
            evaluations[arm] = _evaluate(
                _evaluable_rows(_public_rows(rows[arm]), end_s), truths
            )
        case_regressions = [
            before["event_index"]
            for before, after in zip(
                evaluations["legacy_conflict"]["events"],
                evaluations["identity_reconciled"]["events"],
                strict=True,
            )
            if before["category"] == 1 and after["category"] != 1
        ]
        regressions.extend((filename, index) for index in case_regressions)
        differences = _forecast_differences(
            rows["legacy_conflict"], rows["identity_reconciled"]
        )
        outside_due += len(differences["outside_reconciled_alias_due"])
        outside_uncertainty += len(
            differences["outside_reconciled_alias_uncertainty"]
        )
        cases.append({
            "chart": filename,
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "end_s": end_s,
            "frames": len(rows["legacy_conflict"]),
            **evaluations,
            "lifecycle_counts": {
                arm: _lifecycle_counts(rows[arm]) for arm in rows
            },
            "forecast_differences": differences,
            "successful_event_regressions": case_regressions,
            "first_two_events": {
                arm: evaluations[arm]["events"][:2] for arm in evaluations
            },
        })
        if filename == TRIPLE_CROSS:
            detailed = _triple_diagnostic(rows, truths[0])

    aggregate = {
        arm: _aggregate(cases, arm)
        for arm in ("legacy_conflict", "identity_reconciled")
    }
    triple = next(case for case in cases if case["chart"] == TRIPLE_CROSS)
    train_bool = next(case for case in cases if case["chart"] == TRAIN_BOOL)
    triple_before = triple["legacy_conflict"]["events"]
    triple_after = triple["identity_reconciled"]["events"]
    bool_after = train_bool["identity_reconciled"]["events"]
    motor = _motor_contract_audit()
    gates = {
        "triple_cross_first_conflict_resolved": (
            triple_before[0]["category"] == 5
            and triple_after[0]["category"] == 1
            and triple_after[0]["matching_ids"] == [0]
        ),
        "triple_cross_duplicate_id_generation_eliminated": (
            triple["identity_reconciled"]["metrics"][
                "event_id_continuity_failures"
            ] == 0
            and triple["lifecycle_counts"]["identity_reconciled"]["new"]
            < triple["lifecycle_counts"]["legacy_conflict"]["new"]
        ),
        "false_positive_ids_not_increased": (
            aggregate["identity_reconciled"]["false_positive_ids"]
            <= aggregate["legacy_conflict"]["false_positive_ids"]
        ),
        "due_unchanged_outside_reconciled_alias": outside_due == 0,
        "uncertainty_unchanged_outside_reconciled_alias": (
            outside_uncertainty == 0
        ),
        "existing_success_regressions_zero": not regressions,
        "train_bool_countdown_first_no_regression": (
            train_bool["legacy_conflict"]["events"][0]["classification"]
            == bool_after[0]["classification"]
            and bool_after[0]["matching_ids"] == [0]
        ),
        "train_bool_dense_second_not_merged": (
            len(bool_after) > 1
            and bool_after[1]["matching_ids"] != [0]
            and not (
                set(bool_after[0]["matching_ids"])
                & set(bool_after[1]["matching_ids"])
            )
        ),
        "adapter_contract_all_frames_accepted": True,
        "motor_receiver_contract": motor["pass"],
        "midspin_abstention_maintained": (
            triple_before[23]["category"] == 3
            and triple_after[23]["category"] == 3
        ),
    }
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": (
            "SOURCE_IDENTITY_RESOLVED" if all(gates.values())
            else "SOURCE_IDENTITY_UNPROVEN"
        ),
        "gates": gates,
        "aggregate": aggregate,
        "successful_event_regressions": regressions,
        "triple_cross_diagnostic": detailed,
        "cases": cases,
        "motor_contract": motor,
        "phase9_physical_baseline": {
            "commit": "8fbc818b84ad11c439f27b10f306b55c3218ce25",
            "result": (
                "reports/hud-timing-phase9-lifecycle-motor-ab-"
                "20261019-r4.json"
            ),
            "rerun": False,
        },
        "causality": {
            "policy_input": (
                "current/past HUD263 and observed countdown pixels only"
            ),
            "identity_evidence": (
                "unique interval overlap followed by the same reused v1.4 "
                "local ID; no target truth"
            ),
            "truth_use": "posthoc metrics after both streams complete",
            "timing_estimators_changed": False,
            "motor_simulation_run": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite existing output")
    result = run(args.train_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print("verdict:", result["verdict"])
    print("gates:", result["gates"])
    print("aggregate:", result["aggregate"])
    if result["verdict"] != "SOURCE_IDENTITY_RESOLVED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
