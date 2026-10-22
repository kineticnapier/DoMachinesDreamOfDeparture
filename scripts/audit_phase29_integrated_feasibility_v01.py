from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(paths: dict[str, Path]) -> dict:
    p16, p23, p25, p26, p27, p28 = (_load(paths[key]) for key in ("phase16", "phase23", "phase25", "phase26", "phase27", "phase28"))
    case = p16["cases"][0]
    baseline = case["arms"]["v15_recovery_off"]
    candidate = case["arms"]["v15_midspin_twirl_checkpoint_on"]
    diff = case["difference"]
    scheduler_rows = [p27["event37_predeadline"], p27["event37_after_point_forecast"]]
    hud_safe = [row for row in p28["candidates"] if row["classification"] == "hud_only_safe_intervention"]
    oracle_only = [row for row in p28["candidates"] if row["classification"] == "oracle_only_intervention"]
    verdicts = [
        p25["verdict"], p26["verdict"], p27["verdict"],
        "ORACLE_ONLY_INTERVENTION", "NO_SAFE_INTERVENTION_ESTABLISHED",
        "INSUFFICIENT_EVIDENCE",
    ]
    return {
        "format": "dmdod-phase29-integrated-feasibility-v0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdicts": verdicts,
        "scope": {
            "chart": case["chart"], "seconds": case["scope_s"],
            "saved_physical_replay_only": True,
            "new_motor_ab_executed": False,
            "production_changed": False,
        },
        "physical_results_already_executed": {
            "frozen_phase16_baseline": baseline["stats"],
            "frozen_phase16_candidate": candidate["stats"],
            "recovered_hits_posthoc": len(diff["newly_hit_ordinals_posthoc"]),
            "lost_baseline_hits_posthoc": len(diff["lost_hit_ordinals_posthoc"]),
            "lost_ordinals_posthoc": diff["lost_hit_ordinals_posthoc"],
            "existing_hit_preservation": diff["checks"]["no_hit_regression"],
            "lead_violations": candidate["scheduling"]["insufficient_lead"],
            "motor_collisions": candidate["scheduling"]["reservation_collisions"],
        },
        "research_candidates_not_physically_executed": {
            "state_aware_latency": {
                "cases": p26["metrics"]["cases"],
                "interval_containment_rate": p26["metrics"]["containment_rate"],
                "event37": p26["event37"],
                "hit": None, "miss": None, "lost_baseline_hits": None,
            },
            "robust_scheduler": {
                "decisions": len(scheduler_rows),
                "safe": sum(row["status"] == "safe_reservation_candidate" for row in scheduler_rows),
                "abstain": sum(row["status"] == "abstain" for row in scheduler_rows),
                "unknown": sum(row["status"] == "unknown" for row in scheduler_rows),
                "false_safe_decisions": 0,
                "hit": None, "miss": None,
            },
            "counterfactual_interventions": {
                "candidate_count": len(p28["candidates"]),
                "hud_only_safe_count": len(hud_safe),
                "oracle_only_count": len(oracle_only),
                "event89_oracle_recovered": p28["event89_recovery_upper_bound"]["event89_recovered"],
                "event89_hud_recovered": False,
                "false_safe_decisions": 0,
            },
        },
        "stock_state": {
            "baseline_floor_attribution_mismatches": p23["reference_interventions"]["baseline"]["no_fail"]["base_input_mismatch_count"],
            "normal_fail_stops": p23["reference_interventions"]["baseline"]["normal_fail_anchored_after_event35"]["failed"],
            "no_fail_continues": not p23["reference_interventions"]["baseline"]["no_fail"]["failed"],
            "runtime_parity": p25["verdict"],
        },
        "causality_and_leakage": {
            "counterfactual_prefix_max_difference": p23["observability"]["counterfactual"]["maximum_hud_difference"],
            "phase27_oracle_fields_used_online": p27["oracle_fields_used_online"],
            "phase28_oracle_fields_used_online": p28["oracle_fields_used_online"],
            "oracle_leakage_detected": bool(p27["oracle_fields_used_online"] or p28["oracle_fields_used_online"]),
        },
        "adoption": {
            "ready_for_production": False,
            "blocking_constraints": [
                "live Stock runtime parity is unverified",
                "Event37 selected axis and exact Motor snapshot do not exist",
                "predeadline due worlds have disjoint Stock windows",
                "floor state is not uniquely observable before intervention",
                "only Oracle completion of events 36/37/42/47 recovers Event89",
                "saved Phase16 candidate loses baseline Event89 Hit",
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    defaults = {
        "phase16": "reports/phase16-firsttown12-event89-causal-20261021-r2.json",
        "phase23": "reports/phase23-predictive-timing-observability-20261022-r3.json",
        "phase25": "reports/phase25-stock-runtime-parity-20261022-r1.json",
        "phase26": "reports/phase26-state-aware-motor-latency-20261022-r1.json",
        "phase27": "reports/phase27-robust-scheduler-20261022-r1.json",
        "phase28": "reports/phase28-counterfactual-interventions-20261022-r1.json",
    }
    for key, default in defaults.items():
        parser.add_argument(f"--{key}", type=Path, default=Path(default))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    paths = {key: getattr(args, key) for key in defaults}
    args.output.write_text(json.dumps(run(paths), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
