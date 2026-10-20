from __future__ import annotations

"""Phase 15: HUD-only Checkpoint arrival recovery, fixed Train-only audit.

This evaluator captures both policy streams BEFORE reading target truth. Only
the posthoc evaluator uses floor indices or compiled event target times.
"""

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
    CountdownVisualSensor, StockCountdownRenderer,
)
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG as FLOORS
from dmdod.motor.contact_16k import ContactAction
from dmdod.planning.hud_v3_composite_forecast_source_v30 import (
    HUDV3CompositeForecastSource,
)

if __package__:
    from scripts.audit_hud_timing_phase8_lifecycle_integration_v01 import (
        _evaluable_rows, _lifecycle_counts, _serialize,
    )
    from scripts.audit_hud_timing_phase11_generalization_v01 import (
        REFERENCE, UNSEEN_END_S, verify_selection,
    )
    from scripts.audit_hud_timing_phase14_twirl_v01 import (
        MOTOR_LEAD_S, PHASE12_TARGETS,
    )
    from scripts.audit_timing_generalization_phase7_v01 import _evaluate
else:
    from audit_hud_timing_phase8_lifecycle_integration_v01 import (
        _evaluable_rows, _lifecycle_counts, _serialize,
    )
    from audit_hud_timing_phase11_generalization_v01 import (
        REFERENCE, UNSEEN_END_S, verify_selection,
    )
    from audit_hud_timing_phase14_twirl_v01 import (
        MOTOR_LEAD_S, PHASE12_TARGETS,
    )
    from audit_timing_generalization_phase7_v01 import _evaluate


FORMAT = "dmdod-hud-timing-phase15-checkpoint-audit-v0.1"
TRAIN_DIR = Path("data/DMDOD-v140-tuf-p11-p20/Train")
TARGET_CHART = "04349 - ヒアソビ (feat. 初音ミク) - Zedwards.adofai"
TARGET_SHA256 = "baaf6bff3cd012d8b0e4f123f58868c1cd4f262a35394e32eb2ebcba898caab9"
TARGET_INDEX = 54
CONTROL_DT_S = .010
BASELINE = "baseline-v15-midspin-twirl"
CANDIDATE = "candidate-v15-midspin-twirl-checkpoint"
ARMS = (BASELINE, CANDIDATE)


def _source(checkpoint: bool) -> HUDV3CompositeForecastSource:
    return HUDV3CompositeForecastSource(
        horizon_s=.080,
        lifecycle_version="v1.5",
        reconcile_countdown_v14_identity=True,
        midspin_recovery=True,
        twirl_recovery=True,
        checkpoint_recovery=checkpoint,
    )


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


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
        BASELINE: _source(False),
        CANDIDATE: _source(True),
    }
    env = HudCoupledContactRealChartEnv(
        segment, control_dt_s=CONTROL_DT_S, physics_dt_s=.001,
        behind_floors=FLOORS.behind_floors,
        ahead_floors=FLOORS.ahead_floors,
        tail_s=.040,
    )
    observation = env.reset()
    idle = ContactAction((0.0,) * 12)
    rows = {arm: [] for arm in ARMS}
    for step in range(int(end_s / CONTROL_DT_S) + 1):
        now_s = step * CONTROL_DT_S
        cue = sensor.observe(
            step=step, observed_at_s=now_s,
            frame=renderer.render(now_s),
        )
        for arm, source in sources.items():
            public, raw = source.observe(step, observation, cue)
            rows[arm].append(_serialize(public, raw))
        transition = env.step(idle)
        observation = transition.observation
        if transition.done:
            break
    return rows, segment


def _case(path: Path, end_s: float):
    rows, segment = _capture(path, end_s)
    # Evaluator-only after both policies have been fully captured.
    targets = tuple(
        target for target in segment.targets
        if target.episode_time_s <= end_s + 1e-9
    )
    truths = tuple(float(target.episode_time_s) for target in targets)
    evaluations = {
        arm: _evaluate(_evaluable_rows(rows[arm], end_s), truths)
        for arm in ARMS
    }
    regressions = [
        index for index, (old, new) in enumerate(zip(
            evaluations[BASELINE]["events"],
            evaluations[CANDIDATE]["events"], strict=True,
        )) if old["category"] == 1 and new["category"] != 1
    ]
    return {
        "chart": path.name,
        "sha256": _hash(path),
        "end_s": end_s,
        "event_count": len(targets),
        "evaluations": evaluations,
        "lifecycle": {arm: _lifecycle_counts(rows[arm]) for arm in ARMS},
        "existing_success_regressions": regressions,
    }


def _aggregate(cases: list[dict], arm: str) -> dict:
    events = [
        event for case in cases
        for event in case["evaluations"][arm]["events"]
    ]
    clean = [event for event in events if event["category"] == 1]
    return {
        "charts": len(cases),
        "events": len(events),
        "clean": len(clean),
        "categories": {
            str(i): sum(event["category"] == i for event in events)
            for i in range(1, 6)
        },
        "false_positive_ids": sum(
            case["evaluations"][arm]["metrics"]["false_positive_ids"]
            for case in cases
        ),
        "timing_mae_ms_conditional_on_clean": (
            sum(abs(event["first_forecast_error_ms"]) for event in clean)
            / len(clean) if clean else None
        ),
        "lifecycle": {
            key: sum(case["lifecycle"][arm][key] for case in cases)
            for key in ("new", "reused", "canceled", "elapsed")
        },
    }


def run(train_dir: Path, manifest: dict, *, target_only: bool = False):
    unseen_paths = verify_selection(manifest, train_dir)
    target_path = train_dir / TARGET_CHART
    if not target_path.is_file() or _hash(target_path) != TARGET_SHA256:
        raise ValueError("Phase 15 pinned ヒアソビ chart is absent or changed")
    if target_path not in unseen_paths:
        raise ValueError("Phase 15 chart not present in frozen Train16 selection")
    target_case = _case(target_path, UNSEEN_END_S)
    target_events = target_case["evaluations"]
    before = target_events[BASELINE]["events"][TARGET_INDEX]
    after = target_events[CANDIDATE]["events"][TARGET_INDEX]
    target_info = {
        "event_index": TARGET_INDEX,
        "baseline": before,
        "candidate": after,
        "motor_ready_44ms": (
            after["category"] == 1
            and after["first_forecast_lead_ms"] >= MOTOR_LEAD_S * 1000 - 1e-6
        ),
    }
    cases = [target_case] if target_only else [
        target_case if path == target_path else _case(path, UNSEEN_END_S)
        for path in unseen_paths
    ]
    refs = [] if target_only else [
        _case(train_dir / name, end_s)
        for name, (_, end_s) in REFERENCE.items()
    ]
    checks = {
        "target_old_abstained": before["category"] != 1,
        "target_new_clean": after["category"] == 1,
        "target_motor_ready": target_info["motor_ready_44ms"],
        "false_positive_ids_zero": all(
            case["evaluations"][CANDIDATE]["metrics"]["false_positive_ids"] == 0
            for case in cases + refs
        ),
        "existing_clean_regression_zero": not any(
            case["existing_success_regressions"] for case in cases + refs
        ),
    }
    unseen = {arm: _aggregate(cases, arm) for arm in ARMS}
    reference = {arm: _aggregate(refs, arm) for arm in ARMS}
    phase12_stats = {}
    if not target_only:
        by_name = {case["chart"]: case for case in cases}
        phase12_events = [
            by_name[name]["evaluations"][CANDIDATE]["events"][i]
            for name, indices in PHASE12_TARGETS.items()
            for i in indices
        ]
        phase12_stats = {
            "clean": sum(event["category"] == 1 for event in phase12_events),
            "motor_ready_44ms": sum(
                event["category"] == 1
                and event["first_forecast_lead_ms"]
                >= MOTOR_LEAD_S * 1000 - 1e-6
                for event in phase12_events
            ),
        }
        checks.update({
            "phase14_baseline_unseen_1067": unseen[BASELINE]["clean"] == 1067,
            "phase14_baseline_reference_200": reference[BASELINE]["clean"] == 200,
            "candidate_unseen_1068": unseen[CANDIDATE]["clean"] == 1068,
            "candidate_reference_200": reference[CANDIDATE]["clean"] == 200,
            "phase12_fixed13_preserved": (
                phase12_stats["clean"] == 13
                and phase12_stats["motor_ready_44ms"] >= 9
            ),
        })
    result = {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "target-only" if target_only else "train16-plus-reference5",
        "verdict": (
            "CHECKPOINT_CAUSAL_RECOVERY_PASS" if all(checks.values())
            and not target_only else
            "CHECKPOINT_TARGET_PROBE_PASS" if all(checks.values()) else
            "CHECKPOINT_INTEGRATION_BLOCKED"
        ),
        "checks": checks,
        "target_54": target_info,
        "train16": unseen,
        "reference5": reference,
        "phase12_fixed13": phase12_stats,
        "no_student_truth": True,
        "checkpoint_only_affects_incoming_arrival": True,
        "future_setspeed_gate_relaxed": False,
        "cases": cases + refs,
    }
    return result


def _write_new(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(data, output, indent=2, ensure_ascii=False)
        output.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = run(args.train_dir, manifest, target_only=args.target_only)
    _write_new(args.output, result)
    print("verdict:", result["verdict"])
    print("target_54:", result["target_54"])
    print("checks:", result["checks"])
    if not args.target_only:
        print("train16:", result["train16"])
        print("reference5:", result["reference5"])


if __name__ == "__main__":
    main()
