from __future__ import annotations

"""Read-only explanation of projective Geometry timing error on Train20.

The original projective estimator is replayed on ALL HUD frames before
consulting the oracle's masks/target IDs/remaining times. This module never
changes the predictor, its thresholds, checkpoints, Geometry, or Stage A.
"""

from collections import Counter, defaultdict
import json
from math import atan2, ceil, isfinite, pi
from statistics import mean

from . import prestige1 as p
from . import prestige1_autoplay_countdown as cd
from . import prestige1_autoplay_diagnostics as distance
from . import prestige1_autoplay_first_frame_geometry_audit as first
from . import prestige1_autoplay_halfturn_audit as halfturn
from . import prestige1_autoplay_hybrid_audit as hybrid
from . import prestige1_autoplay_near_ablation as ablation
from . import prestige1_autoplay_near_countdown_audit as near
from . import prestige1_autoplay_projective_orbit_audit as projective
from . import prestige1_autoplay_special_motion_audit as special
from . import prestige1_autoplay_switch_hud_audit as switch
from . import prestige1_transfer as transfer

FORMAT = "prestige1-stage-t-projective-error-provenance-train20-v1"
REPORT = "stage_timing_autoplay_projective_error_train20.json"
ERROR_CUTOFFS_MS = (0.01, 0.1, 1.0, 5.0, 20.0, 100.0)
EXAMPLES = 16
CANDIDATES = projective.CANDIDATES


def _statistics(values: list[float], *, unit: str) -> dict:
    if not values:
        return {"count": 0, "unit": unit, "mean": None,
                "mean_abs": None, "median_abs": None, "p95_abs": None,
                "min": None, "max": None}
    asc=sorted(abs(v) for v in values)
    return {
        "count":len(values), "unit":unit,
        "mean":mean(values), "mean_abs":mean(asc),
        "median_abs":asc[len(asc)//2] if len(asc)%2 else
          (asc[len(asc)//2-1]+asc[len(asc)//2])/2,
        "p95_abs":asc[ceil(.95*len(asc))-1],
        "min":min(values), "max":max(values),
    }


def _breakdown(rows: list[dict]) -> dict:
    signed=[r["error_ms"] for r in rows]
    by_anchor=defaultdict(list)
    by_tile_change=defaultdict(list)
    by_real_change=defaultdict(list)
    by_sign=defaultdict(list)
    for r in rows:
        by_anchor[str(r["anchor"])].append(r)
        by_tile_change[str(r["tile_bpm_changed"])].append(r["error_ms"])
        by_real_change[str(r["real_bpm_changed"])].append(r["error_ms"])
        by_sign["positive" if r["error_ms"]>0 else "negative" if r["error_ms"]<0 else "zero"].append(r["error_ms"])
    grouped=[]
    for key,items in sorted(by_anchor.items(),key=lambda x:int(x[0])):
        errs=[r["error_ms"] for r in items]
        grouped.append({
            "anchor":int(key),"count":len(errs),"error":_statistics(errs,unit="ms"),
            "total_absolute_error_ms":sum(abs(x) for x in errs),
            "largest_abs_error_ms":max(abs(x) for x in errs),
            "tile_bpm_changed_count":sum(r["tile_bpm_changed"] for r in items),
            "real_bpm_changed_count":sum(r["real_bpm_changed"] for r in items),
        })
    return {
        "eligible":len(rows),
        "error":_statistics(signed,unit="ms"),
        "absolute_error_count_above_ms":{
            str(k):sum(abs(x)>k for x in signed)
            for k in ERROR_CUTOFFS_MS
        },
        "by_anchor":grouped,
        "by_tile_bpm_changed":{
            key:_statistics(v,unit="ms")
            for key,v in sorted(by_tile_change.items())
        },
        "by_real_bpm_changed":{
            key:_statistics(v,unit="ms")
            for key,v in sorted(by_real_change.items())
        },
        "by_error_sign":{
            key:_statistics(v,unit="ms")
            for key,v in sorted(by_sign.items())
        },
        "largest_errors":sorted(rows,key=lambda x:abs(x["error_ms"]),reverse=True)[:EXAMPLES],
        "total_absolute_error_ms":sum(abs(x) for x in signed),
    }


def _frame_properties(
    old:tuple[float,...], current:tuple[float,...], snapshot:dict | None,
) -> dict:
    tile=cd.HUD_BPM_REFERENCE * 2.**current[cd.HUD_OFFSET]
    old_tile=cd.HUD_BPM_REFERENCE * 2.**old[cd.HUD_OFFSET]
    real=cd.HUD_BPM_REFERENCE * 2.**current[cd.HUD_OFFSET+1]
    old_real=cd.HUD_BPM_REFERENCE * 2.**old[cd.HUD_OFFSET+1]
    current_floor=cd._visible_slot(current,cd.CURRENT_SLOT)
    previous_floor=cd._visible_slot(old,cd.CURRENT_SLOT)
    entry=atan2(current_floor[2],current_floor[3])
    exit_angle=atan2(current_floor[4],current_floor[5])
    angle=atan2(current[cd.MOTOR_FEATURES],current[cd.MOTOR_FEATURES+1])
    marker=special._marker_snapshot(current)
    return {
        "tile_bpm":tile,"old_tile_bpm":old_tile,
        "real_bpm":real,"old_real_bpm":old_real,
        "tile_bpm_changed":abs(current[cd.HUD_OFFSET]-old[cd.HUD_OFFSET])>1e-6,
        "real_bpm_changed":abs(current[cd.HUD_OFFSET+1]-old[cd.HUD_OFFSET+1])>1e-6,
        "current_floor_entry_deg":entry*180/pi,
        "current_floor_exit_deg":exit_angle*180/pi,
        "current_orbit_deg":angle*180/pi,
        "current_floor_xy":list(current_floor[:2]),
        "previous_floor_xy":list(previous_floor[:2]),
        "current_markers":[x for x,v in marker["current"].items() if v],
        "next_markers":[x for x,v in marker["next"].items() if v],
        "midspin_or_multiplanet_any":any(marker[x] for x in (
            "current_midspin","next_midspin",
            "current_multiplanet","next_multiplanet")),
        "projective_angular_step_deg":(
            snapshot["current_step"]*180/pi if snapshot and
            snapshot["current_step"] is not None else None),
        "projective_prior_step_deg":(
            snapshot["prior_step"]*180/pi if snapshot and
            snapshot["prior_step"] is not None else None),
        "projective_angular_consistency_deg":(
            snapshot["projective_step_difference_abs_deg"]
            if snapshot else None),
        "projective_speed_ratio":snapshot["current_speed_ratio"] if snapshot else None,
    }


def _replay_split(group: list, traces:dict) -> dict:
    """No oracle field is used until all HUD predictions are computed."""
    all_errors={name:[] for name in CANDIDATES}
    floor_errors={name:[] for name in CANDIDATES}
    floor_rows={name:[] for name in CANDIDATES}
    all_rows={name:[] for name in CANDIDATES}
    unadopted=Counter()
    by_status=Counter()
    by_original=Counter()
    same_event=0
    switch=0
    for seq in group:
        hud=tuple(tuple(float(v) for v in frame)
                  for frame in seq.visible.detach().cpu().tolist())
        candidates,original,snapshots=projective.causal_projective_candidates(hud)
        trace=traces[seq.anchor]
        if len(hud)!=len(trace.fallback_s):
            raise RuntimeError("Frozen GRU trace and HUD length mismatch")
        # CAUSAL precomputation ends here. Oracle fields used only for scoring.
        for k in range(1,len(hud)):
            if not(bool(seq.mask[k-1]) and bool(seq.mask[k])):
                continue
            a,b=seq.event_indices[k-1:k+1]
            if a<0 or b<0:
                continue
            if a==b:
                same_event+=1
                continue
            switch+=1
            true_s=float(seq.remaining_s[k])
            for name in CANDIDATES:
                candidate=candidates[name][k]
                if candidate.seconds is None:
                    if name=="projective_speed":
                        by_status[candidate.status]+=1
                        by_original[(original[k],candidate.status)]+=1
                    continue
                detail={
                    "anchor":seq.anchor,
                    "chart_sha256":seq.chart_sha256,
                    "frame":k,
                    "original_status":original[k],
                    "predicted_remaining_ms":candidate.seconds*1000,
                    "oracle_remaining_ms":true_s*1000,
                    "error_ms":(candidate.seconds-true_s)*1000,
                    "matched_gru_error_ms":(trace.fallback_s[k]-true_s)*1000,
                    "matched_hybrid_error_ms":(trace.chosen_s[k]-true_s)*1000,
                    **_frame_properties(hud[k-1],hud[k],snapshots[k]),
                }
                all_errors[name].append(detail["error_ms"])
                all_rows[name].append(detail)
                if original[k]=="floor_changed":
                    floor_errors[name].append(detail["error_ms"])
                    floor_rows[name].append(detail)
                else:
                    if original[k]!="inconsistent_orbit_motion":
                        raise RuntimeError("Unexpected eligible projective cohort")
    return {
        "switch_count":switch,"same_event_pair_count":same_event,
        "eligible_all_by_candidate":{
            name:len(rows) for name,rows in all_rows.items()
        },
        "candidate_error_check":{
            name:{
                "all":_statistics(all_errors[name],unit="ms"),
                "floor":_statistics(floor_errors[name],unit="ms"),
            } for name in CANDIDATES
        },
        "floor_details_by_candidate":{
            name:_breakdown(floor_rows[name]) for name in CANDIDATES
        },
        "noneligible_projective_speed_by_status":dict(sorted(by_status.items())),
        "noneligible_projective_speed_by_original":[
            {"original":orig,"candidate_status":status,"count":cnt}
            for (orig,status),cnt in sorted(by_original.items())
        ],
        "noneligible_original_reasons":dict(sorted(Counter(
            row["original_status"] for row in [] # empty placeholder below
        ).items())),
    }


def run_projective_error_audit(config:p.PrestigeConfig) -> dict:
    root=transfer.output_dir(config)
    output=root/REPORT
    source=distance.prepare(config,output_paths=(output,))
    prior_path=root/projective.REPORT
    if not prior_path.is_file():
        raise FileNotFoundError("Run projective orbit audit first")
    prior_sha=p.sha256(prior_path)
    saved=json.loads(prior_path.read_text(encoding="utf-8"))
    frozen,near_report,fingerprints=near._restore_near_only(config,source)
    near_sha=hybrid._validate_prior(root,source,near_report,fingerprints,frozen)
    initial={**fingerprints,"near_only_countdown":near_sha}
    scores,switching,results={}, {}, {}
    for split,group in (("train_fit",source.fit),("train_heldout",source.heldout)):
        traces={seq.anchor:hybrid.infer_hybrid(seq.visible,frozen) for seq in group}
        scores[split]=hybrid._aggregate_scores(group,traces,source.reference)
        switching[split]=hybrid._switching_scores(group,traces)
        old_now=projective._score_split(group,traces)
        old=saved["results"][split]
        if (old_now["pair_totals"]!=old["pair_totals"]
            or old_now["original_geometry_reason_at_switch"]!=old["original_geometry_reason_at_switch"]):
            raise RuntimeError(f"Projective counterfactual reason/count changed: {split}")
        for name in CANDIDATES:
            for cohort in ("all_switch","original_motion","original_floor",
                           "floor_halfturn_unexplained","floor_halfturn_explained",
                           "same_event"):
                before=old["double_angle_candidates"][name][cohort]
                now=old_now["double_angle_candidates"][name][cohort]
                for key in ("eligible","total_pairs","statuses"):
                    if before[key]!=now[key]:
                        raise RuntimeError(
                            f"Frozen RP1 {split}/{name}/{cohort}/{key} changed")
                for key in ("candidate_timing","matched_frozen_gru","matched_frozen_hybrid"):
                    prior_result=before[key]
                    now_result=now[key]
                    if prior_result["count"]!=now_result["count"]:
                        raise RuntimeError("Projective score count drift")
                    if prior_result["mae_ms"] is not None and (
                        now_result["mae_ms"] is None
                        or abs(prior_result["mae_ms"]-now_result["mae_ms"])>.05):
                        raise RuntimeError("Frozen RP1 MAE drift")
        results[split]=_replay_split(group,traces)
        if (results[split]["switch_count"]!=old["pair_totals"]["all_switch"]
            or results[split]["same_event_pair_count"]!=old["pair_totals"]["same_event"]):
            raise RuntimeError("RP1 detailed oracle cohort count drift")
        for name in CANDIDATES:
            metrics=results[split]["candidate_error_check"][name]
            for section,old_label in (("all","all_switch"),("floor","original_floor")):
                prev=old["double_angle_candidates"][name][old_label]
                current=metrics[section]
                if (current["count"]!=prev["eligible"]
                    or abs(current["mean_abs"]-prev["candidate_timing"]["mae_ms"])>.01):
                    raise RuntimeError(f"Detailed RP1 MAE drift: {split}/{name}/{section}")
    hybrid_sha=switch._check_saved_hybrid(
        root/hybrid.REPORT,config,source,initial,scores,switching)
    switch_sha=first._verify_prior_report(
        source,config,initial,scores,switching)
    expected={
        **source.source_sha256,
        "near_only_report":p.sha256(root/ablation.REPORT),
        "near_only_checkpoint":p.sha256(root/ablation.CHECKPOINT),
        "distance_audit":p.sha256(root/distance.REPORT),
        "old_countdown":p.sha256(root/cd.REPORT),
        "near_only_countdown":p.sha256(root/near.REPORT),
        "stateless_hybrid":hybrid_sha,
        "hud_switch_audit":switch_sha,
        "first_frame_geometry":p.sha256(root/first.REPORT),
        "special_motion":p.sha256(root/special.REPORT),
        "halfturn_angle_audit":p.sha256(root/halfturn.REPORT),
    }
    if (saved.get("format")!=projective.FORMAT
        or saved.get("config_sha256")!=config.signature()
        or saved.get("train_partition")!=source.partition
        or saved.get("source_sha256")!=expected
        or saved.get("no_training") is not True
        or saved.get("validation_evaluated") is not False
        or saved.get("final_evaluated") is not False
        or saved.get("motor_student_training_permitted") is not False
        or saved.get("original_stage_a_gate_overridden") is not False
        or saved.get("adoption_permitted") is not False
        or p._protect_old_best(config)!=source.original_best_sha
        or p.sha256(prior_path)!=prior_sha):
        raise RuntimeError("Frozen projective report/Stage A provenance mismatch")
    report={
        "format":FORMAT,
        "role":"Read-only Train20 fit/heldout explanatory error audit",
        "purpose":"Explain Train-fit projective floor-change MAE with unsmoothed signed residuals",
        "hud_causal_predictions_before_oracle_scoring":True,
        "no_training":True,
        "validation_evaluated":False,"final_evaluated":False,
        "original_stage_a_gate_overridden":False,
        "motor_student_training_permitted":False,
        "automatically_adopted":False,
        "geometry_unchanged":True,"projective_estimator_unchanged":True,
        "config_sha256":config.signature(),
        "train_partition":source.partition,
        "source_sha256":{**expected,"projective_orbit_report":prior_sha},
        "error_cutoffs_ms":ERROR_CUTOFFS_MS,
        "results":results,
        "limitations":[
            "Prior Train-heldout was repeatedly inspected; no clean independent test",
            "All conclusions conditional on current HUD renderer and frozen counterfactual",
            "No new predictor/guard threshold selected from observed errors",
            "No Motor/Validation/FINAL; Stage A FAIL unchanged",
        ],
    }
    root.mkdir(parents=True,exist_ok=True)
    p._no_replace_json(output,report)
    return report
