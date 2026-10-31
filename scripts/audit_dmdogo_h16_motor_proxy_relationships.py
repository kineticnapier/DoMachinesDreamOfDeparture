from __future__ import annotations

"""H1.6: read-only, paired descriptive analysis of completed H1.5 Motor runs.

This consumes the FULL 22/22 H1.5 census and the 22 independently checked
H1.4 physical results. Nothing is simulated, trained, reviewed or edited.
Repeated counterfactual runs on one chart are NOT independent chart samples.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import run_dmdogo_h15_motor_census as h15
from audit_dmdogo_h13_automated_fingering_quality import PRESSURE_KEYS
from dmdogo_v08_provenance import sha256_file

FORMAT = "dmdogo-h16-motor-vs-abstract-proxy-pairs-v1"
MAE_NOISE_BAND_MS = 0.100  # Descriptive comparison band; NOT a clinical or motor threshold.
AXES = tuple(f"{hand}_{i}" for hand in ("left", "right") for i in range(1, 5))


def strict_number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    return value


def signed_bucket(delta, band=0):
    """Compare observed differences without treating tiny MAE jitter as improvement."""
    delta = strict_number(delta, "delta")
    if delta < -band:
        return "decreased"
    if delta > band:
        return "increased"
    return "unchanged_within_band"


def require_verified_h15(saved, recomputed):
    """A saved H1.5 JSON is not its own proof; recreate it from episode bytes."""
    if not isinstance(saved, dict) or not isinstance(recomputed, dict):
        raise ValueError("missing H1.5 census report")
    if (saved.get("format") != h15.FORMAT
            or saved != recomputed
            or saved.get("full_census_verified") is not True
            or saved.get("verified_episode_count") != 22
            or saved.get("all_hit_exact_plan_count") != 22
            or saved.get("pending_episode_count") != 0
            or saved.get("incomplete_episode_count") != 0
            or saved.get("matching_runtime_fingerprint_count") != 22
            or saved.get("reused_prior_h14_pilots") != 2
            or saved.get("human_likeness_verified") is not False
            or saved.get("validation20_or_final_accessed") is not False):
        raise ValueError("H1.5 census incomplete, drifted or unaudited")
    if (len(saved.get("records", [])) != 22
            or any(row.get("status") != "verified_episode"
                   for row in saved["records"])):
        raise ValueError("H1.5 case records incomplete")


def compare_pair(physical, proxy):
    """Keep objective, abstract pressure and Motor effects on separate axes."""
    cid = physical["candidate_id"]
    if (cid != proxy.get("candidate_id")
            or physical.get("status") != "verified_episode"
            or physical.get("all_hit_and_exact_plan") is not True
            or physical.get("same_recorded_runtime_fingerprint_as_original") is not True
            or physical.get("selected_events") != physical.get("event_count")
            or physical.get("planned_axis_adoption_count") != physical.get("event_count")
            or physical.get("local_replans") != 0
            or physical.get("hits") != physical.get("event_count")
            or physical.get("misses") != 0
            or physical.get("overloaded") is not False
            or proxy.get("changed_event_count") != physical.get("changed_events")
            or proxy.get("triage_tier") != physical.get("triage_tier")):
        raise ValueError(f"{cid}: physical/abstract evidence identity differs")
    heuristic = proxy["heuristic_objective"]
    delta_cost = strict_number(heuristic["alternative_minus_original"], "heuristic delta")
    if (abs(physical["heuristic_cost_delta"]-delta_cost) > 2e-6
            or (delta_cost < -1e-7) != physical["lower_heuristic_cost"]
            or heuristic["alternative_has_lower_frozen_heuristic_cost"]
               != physical["lower_heuristic_cost"]):
        raise ValueError(f"{cid}: cost or improvement flag mismatched")
    abstract = proxy["abstract_metrics"]
    if not isinstance(abstract,dict) or set(abstract) != {"original","alternative"}:
        raise ValueError(f"{cid}: missing paired abstract axis metrics")
    a, b = abstract["original"], abstract["alternative"]
    if a["total_contacts"] != b["total_contacts"] or a["total_contacts"] <= 0:
        raise ValueError(f"{cid}: abstract full-chart contact total differs")
    contact_diff = {}
    for axis in AXES:
        original = a["axis_contacts"].get(axis, 0)
        alternative = b["axis_contacts"].get(axis, 0)
        if (type(original) is not int or original < 0
                or type(alternative) is not int or alternative < 0):
            raise ValueError(f"{cid}: invalid axis contacts")
        contact_diff[axis] = alternative-original
    if sum(contact_diff.values()) != 0:
        raise ValueError(f"{cid}: contact mass is not conserved")
    hand_diff = {}
    for hand in ("left","right"):
        old, new = a["hand_contacts"][hand], b["hand_contacts"][hand]
        if type(old) is not int or type(new) is not int:
            raise ValueError(f"{cid}: invalid hand contacts")
        hand_diff[hand] = new-old
    if (sum(hand_diff.values()) != 0
            or sum(a["hand_contacts"].values()) != a["total_contacts"]
            or sum(b["hand_contacts"].values()) != b["total_contacts"]):
        raise ValueError(f"{cid}: hand contact conservation failed")
    pressure = proxy["pressure_proxy_deltas"]
    if set(pressure) != set(PRESSURE_KEYS):
        raise ValueError(f"{cid}: pressure fields differ from H1.3")
    changes = {}
    for key in PRESSURE_KEYS:
        old = strict_number(a[key], key+" old")
        new = strict_number(b[key], key+" new")
        delta = strict_number(pressure[key], key+" delta")
        if abs((new-old)-delta) > 1e-8:
            raise ValueError(f"{cid}: {key} source delta differs")
        changes[key] = delta
    if (sorted(k for k,v in changes.items() if v>0)
            != proxy["increased_pressure_proxy_names"]
            or sorted(k for k,v in changes.items() if v<0)
            != proxy["decreased_pressure_proxy_names"]):
        raise ValueError(f"{cid}: pressure sign classification drifted")
    original_mae = strict_number(physical["original_timing_mae_ms"], "old MAE")
    alternative_mae = strict_number(physical["timing_mae_ms"], "new MAE")
    diff = alternative_mae-original_mae
    if (physical["timing_mae_delta_ms"] is None
            or abs(physical["timing_mae_delta_ms"]-round(diff,6))>1e-7):
        raise ValueError(f"{cid}: historic and new MAE mismatch")
    original_deferrals = physical.get("original_deferred_events")
    # H1.5 records only new deferrals; original verified baseline lives
    # in its already attested per-chart input row. Pair this in compiler.
    if original_deferrals is None:
        raise ValueError(f"{cid}: historical deferred-events reference missing")
    if type(original_deferrals) is not int or original_deferrals < 0:
        raise ValueError(f"{cid}: invalid original deferral count")
    new_deferrals = physical["deferred_events"]
    if type(new_deferrals) is not int or new_deferrals < 0:
        raise ValueError(f"{cid}: invalid new deferral count")
    return {
        "candidate_id":cid,
        "chart":physical["chart"],
        "chart_sha256":physical["chart_sha256"],
        "chart_event_count":physical["event_count"],
        "changed_events":physical["changed_events"],
        "changed_event_fraction":round(physical["changed_events"]/physical["event_count"],6),
        "triage_tier":proxy["triage_tier"],
        "objective":{
            "alternative_minus_original":delta_cost,
            "lower_frozen_six_term_cost":physical["lower_heuristic_cost"],
            "six_term_deltas":heuristic["six_term_deltas"],
        },
        "physical_motor":{
            "original_timing_mae_ms":original_mae,
            "alternative_timing_mae_ms":alternative_mae,
            "alternative_minus_original_mae_ms":round(diff,6),
            "mae_direction_with_0p1ms_band":signed_bucket(diff, MAE_NOISE_BAND_MS),
            "original_deferred_events":original_deferrals,
            "alternative_deferred_events":new_deferrals,
            "alternative_minus_original_deferred_events":new_deferrals-original_deferrals,
            "deferral_direction":signed_bucket(new_deferrals-original_deferrals),
            "hits":physical["hits"],
            "misses":physical["misses"],
            "local_replans":physical["local_replans"],
            "planned_axis_adoption_count":physical["planned_axis_adoption_count"],
            "same_recorded_runtime_fingerprint":True,
        },
        "abstract_fingering":{
            "original":a, "alternative":b,
            "pressure_proxy_deltas":changes,
            "increased_proxy_count":sum(v>0 for v in changes.values()),
            "decreased_proxy_count":sum(v<0 for v in changes.values()),
            "hand_contacts_alternative_minus_original":hand_diff,
            "axis_contacts_alternative_minus_original":contact_diff,
            "total_contacts":a["total_contacts"],
        },
        "physical_motor_success":True,
        "human_likeness_verified":False,
    }


def group_summary(pairs):
    """Directional counts are descriptive. Do not report p-values/correlations."""
    return {
        "candidate_count":len(pairs),
        "lower_heuristic_cost_count":sum(x["objective"]["lower_frozen_six_term_cost"] for x in pairs),
        "all_hit_exact_plan_count":sum(x["physical_motor_success"] for x in pairs),
        "mae_direction_counts":dict(sorted(Counter(
            x["physical_motor"]["mae_direction_with_0p1ms_band"] for x in pairs).items())),
        "deferral_direction_counts":dict(sorted(Counter(
            x["physical_motor"]["deferral_direction"] for x in pairs).items())),
        "proxy_increased_in_any_dimension_count":sum(
            x["abstract_fingering"]["increased_proxy_count"]>0 for x in pairs),
        "proxy_decreased_in_any_dimension_count":sum(
            x["abstract_fingering"]["decreased_proxy_count"]>0 for x in pairs),
        "pressure_proxy_direction_counts":{key:{
            "increased":sum(x["abstract_fingering"]["pressure_proxy_deltas"][key]>0 for x in pairs),
            "decreased":sum(x["abstract_fingering"]["pressure_proxy_deltas"][key]<0 for x in pairs),
            "equal":sum(x["abstract_fingering"]["pressure_proxy_deltas"][key]==0 for x in pairs),
        } for key in PRESSURE_KEYS},
    }


def compile_analysis(h15_report, h13, h15_rows, original_by_chart, *, input_sha):
    """Compare EXACT source-specific pairs; never substitute independent charts."""
    q={x["candidate_id"]:x for chart in h13["charts"] for x in chart["candidates"]}
    if len(q)!=22 or len(h15_rows)!=22 or len(original_by_chart)!=3:
        raise ValueError("H1.6 requires 22 H1.3 candidates and 3 original Train baselines")
    pairs=[]
    seen=set()
    for physical in h15_rows:
        cid=physical["candidate_id"]
        if cid not in q or cid in seen:
            raise ValueError("missing/duplicate candidate in H1.3–H1.5 join")
        seen.add(cid)
        sha=physical["chart_sha256"]
        old=original_by_chart.get(sha)
        if old is None or old.get("status")!="chart_complete":
            raise ValueError(f"{cid}: no verified original chart baseline")
        if (old.get("hits")!=physical["original_hits"]
                or old.get("timing_mae_ms")!=physical["original_timing_mae_ms"]
                or old.get("local_replans")!=0):
            raise ValueError(f"{cid}: original chart baseline differs")
        row={**physical,"original_deferred_events":old["deferred_events"]}
        pairs.append(compare_pair(row,q[cid]))
    if len(seen)!=22 or seen!=set(q):
        raise ValueError("H1.6 incomplete candidate join")
    lower=[x for x in pairs if x["objective"]["lower_frozen_six_term_cost"]]
    other=[x for x in pairs if not x["objective"]["lower_frozen_six_term_cost"]]
    if len(lower)!=7 or len(other)!=15:
        raise ValueError("H1.1 seven objective gains census changed")
    charts=[]
    for chart in h13["charts"]:
        sha=chart["chart_sha256"]
        subset=[x for x in pairs if x["chart_sha256"]==sha]
        if len(subset)!=chart["candidate_count"]:
            raise ValueError("per-chart counterfactual census differs")
        charts.append({
            "chart":chart["chart"],"chart_sha256":sha,
            "original_source_events":chart["event_count"],
            "summary":group_summary(subset),
        })
    return {
        "format":FORMAT,
        "sources":input_sha,
        "scope":{
            "distinct_train_charts":3,"distinct_original_events":10231,
            "independently_executed_alternative_episodes":22,
            "alternative_seams_per_chart":[x["candidate_count"] for x in h13["charts"]],
            "new_motor_runs":0,"source_v08_unchanged":True,
        },
        "overall":group_summary(pairs),
        "lower_cost_group":group_summary(lower),
        "not_lower_cost_group":group_summary(other),
        "per_chart":charts,
        "pairs":pairs,
        "interpretation":{
            "physical_22_of_22_success":"Saturated in this selected Train Oracle-timed sample; not discriminative.",
            "mae_band_ms":MAE_NOISE_BAND_MS,
            "mae_band_is_human_threshold":False,
            "cost_and_abstract_proxies_are_not_independent":"Both derive from the same chosen axis schedule; signs do not establish preference.",
            "correlation_or_significance_test_performed":False,
            "correlated_counterfactuals":"22 episodes share only three charts and repeated original baselines.",
        },
        "human_likeness_verified":False,
        "human_preference_observed":False,
        "fatigue_or_comfort_measured":False,
        "hud_only_student_verified":False,
        "validation20_or_final_accessed":False,
        "frozen_planner_motor_timing_changed":False,
        "limitations":[
            "Original physical v0.8 baseline is a previously saved episode, not rerun now.",
            "Same recorded runtime fingerprint cannot eliminate wall-time variance between episodes.",
            "Timing MAE differences can reflect execution variability, not just fingering.",
            "Deferral counts show control-loop pressed-state conflicts, not human strain.",
            "H1.3 abstract hand/axis metrics are computed from plans, not measured human fingers.",
            "No independent chart sample: 22 counterfactuals from 3 selected Train charts.",
            "All alternative plans full-Hit with exact adoption here; Motor success alone cannot rank them.",
            "Directional counts are descriptions, not causal effect estimates or statistical significance.",
        ],
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","verified-report","h11-report","h13-report",
                 "results-root","h15-report","output"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--run-root",type=Path,
                   default=Path("artifacts/dmdogo-h15/motor-r1"))
    p.add_argument("--status-root",type=Path,
                   default=Path("logs/dmdogo-h15/motor-r1"))
    args=p.parse_args()
    if args.output.exists():
        p.error("refusing to overwrite an existing H1.6 analysis")
    # H1.5 verifies source v0.8, H1.1/H1.3, status sidecars and all 22
    # complete physical decision and KeyDown traces again before analysis.
    hashes,digest,evidence,order=h15.load_context(args)
    repo=Path(__file__).resolve().parents[1]
    rows=[h15.inspect(cid,repo=repo,result_root=args.run_root,
                      status_root=args.status_root,hashes=hashes,
                      frozen_digest=digest,evidence=evidence) for cid in order]
    recomputed=h15.summary(rows,hashes=hashes,
                           frozen_digest=digest,evidence=evidence)
    saved=h15.load(args.h15_report)
    require_verified_h15(saved,recomputed)
    h13=h15.load(args.h13_report)
    verified=h15.load(args.verified_report)
    baselines={x["chart_sha256"]:x for x in verified["records"]}
    report=compile_analysis(
        saved,h13,rows,baselines,
        input_sha={**hashes,"h15":sha256_file(args.h15_report)})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as target:
        json.dump(report,target,ensure_ascii=False,indent=2)
        target.write("\n")
    print(json.dumps({
        "candidate_count":report["overall"]["candidate_count"],
        "all_hit_exact_plan_count":report["overall"]["all_hit_exact_plan_count"],
        "lower_heuristic_cost_group":report["lower_cost_group"],
        "not_lower_cost_group":report["not_lower_cost_group"],
        "per_chart":report["per_chart"],
        "saved":str(args.output),
        "new_motor_runs":0,
        "human_likeness_verified":False,
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
