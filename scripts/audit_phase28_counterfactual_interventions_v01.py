from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from dmdod.research.counterfactual_intervention_v01 import InterventionEvidence, classify_intervention


def _stable_hash(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run(phase16_path: Path, phase23_path: Path, phase25_path: Path, phase27_path: Path) -> dict:
    p16 = json.loads(phase16_path.read_text(encoding="utf-8"))
    p23 = json.loads(phase23_path.read_text(encoding="utf-8"))
    p25 = json.loads(phase25_path.read_text(encoding="utf-8"))
    p27 = json.loads(phase27_path.read_text(encoding="utf-8"))
    diagnostic = p16["cases"][0]["arms"]["v15_midspin_twirl_checkpoint_on"]["diagnostic_window_posthoc"]
    baseline_inputs = diagnostic["logical_keydowns"]
    reference = p23["reference_interventions"]
    candidates = []
    for name, arms in reference.items():
        nofail = arms["no_fail"]
        oracle_success = bool(nofail["event89_input_hit"])
        is_oracle_all = name == "oracle_all_pre89_lead_abstentions"
        evidence = InterventionEvidence(
            name=name,
            oracle_event89_recovered=oracle_success,
            action_information_visible_by_deadline=False if is_oracle_all else None,
            robust_all_hypotheses_safe=False if is_oracle_all else None,
            motor_envelope_valid=False if is_oracle_all else None,
            baseline_hits_preserved=None,
            runtime_parity_confirmed=p25["verdict"] == "RUNTIME_PARITY_CONFIRMED",
        )
        assessment = classify_intervention(evidence)
        candidates.append({
            "name": name,
            "reference_nofail": {
                "mismatch_count": nofail["base_input_mismatch_count"],
                "event89_recovered": nofail["event89_input_hit"],
                "failed": nofail["failed"],
                "intervention_rows": nofail["selected_intervention_rows"],
            },
            "reference_normal_fail": {
                "failed": arms["normal_fail_anchored_after_event35"]["failed"],
                "event89_recovered": arms["normal_fail_anchored_after_event35"]["event89_input_hit"],
            },
            "evidence": asdict(evidence),
            "classification": assessment.classification.value,
            "reasons": assessment.reasons,
        })
    oracle_all = next(row for row in candidates if row["name"] == "oracle_all_pre89_lead_abstentions")
    return {
        "format": "dmdod-phase28-counterfactual-interventions-v0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdicts": ["ORACLE_ONLY_INTERVENTION", "NO_SAFE_INTERVENTION_ESTABLISHED"],
        "baseline": {
            "logical_keydown_count": len(baseline_inputs),
            "logical_keydowns_sha256": _stable_hash(baseline_inputs),
            "phase23_baseline_mismatch": reference["baseline"]["no_fail"]["base_input_mismatch_count"],
            "event89_recovered": reference["baseline"]["no_fail"]["event89_input_hit"],
        },
        "candidates": candidates,
        "event89_recovery_upper_bound": {
            "candidate": oracle_all["name"],
            "required_evaluator_only_events": [36, 37, 42, 47],
            "first_reference_processed_frame_s": 3.19,
            "last_reference_processed_frame_s": 3.58,
            "mismatch_count_after": oracle_all["reference_nofail"]["mismatch_count"],
            "event89_recovered": oracle_all["reference_nofail"]["event89_recovered"],
            "causal_status": "not selectable: all four exact dues/floors are evaluator truth and include known lead-abstention events",
            "baseline_hit_effect": "not established by limited Stock attribution replay",
        },
        "other_candidate_families": {
            "state_aware_lead": "Event37 actual axis/snapshot absent -> UNKNOWN",
            "reservation_cancel_or_reorder": "counterfactual physical trace absent -> INCONCLUSIVE",
            "axis_reassignment": "fixed Network abstained before allocation -> INCONCLUSIVE",
            "late_window_keydown": "3.251 NoFail helps Event37 only; Event89 remains missed",
        },
        "robust_scheduler_status": p27["event37_predeadline"]["status"],
        "oracle_fields_used_online": [],
        "production_changed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase16", type=Path, default=Path("reports/phase16-firsttown12-event89-causal-20261021-r2.json"))
    parser.add_argument("--phase23", type=Path, default=Path("reports/phase23-predictive-timing-observability-20261022-r3.json"))
    parser.add_argument("--phase25", type=Path, default=Path("reports/phase25-stock-runtime-parity-20261022-r1.json"))
    parser.add_argument("--phase27", type=Path, default=Path("reports/phase27-robust-scheduler-20261022-r1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = run(args.phase16, args.phase23, args.phase25, args.phase27)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
