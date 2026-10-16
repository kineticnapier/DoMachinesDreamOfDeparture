from __future__ import annotations

"""Read-only RP1 (double-angle) orbit-motion and countdown diagnostic.

All candidate outputs use causal [T,263] visible HUD only. Floor/current
event identities, oracle timestamps, and near masks are used AFTER inference
to classify scores. Original Geometry, GRU, Hybrid and Stage A are unchanged.
"""

from collections import Counter
from dataclasses import dataclass
import json
from math import atan2, ceil, cos, isfinite, pi, radians, sin
from statistics import mean, median

from . import prestige1 as p
from . import prestige1_autoplay_countdown as cd
from . import prestige1_autoplay_diagnostics as distance
from . import prestige1_autoplay_first_frame_geometry_audit as first
from . import prestige1_autoplay_halfturn_audit as halfturn
from . import prestige1_autoplay_hybrid_audit as hybrid
from . import prestige1_autoplay_near_ablation as ablation
from . import prestige1_autoplay_near_countdown_audit as near
from . import prestige1_autoplay_special_motion_audit as special
from . import prestige1_autoplay_switch_hud_audit as switch
from . import prestige1_autoplay_readout as stage_t
from . import prestige1_transfer as transfer

FORMAT = "prestige1-stage-t-projective-orbit-counterfactual-train20-v1"
REPORT = "stage_timing_autoplay_projective_orbit_train20.json"
CANDIDATES = ("projective_speed", "projective_prior_agreement")
TARGET_REASONS = ("inconsistent_orbit_motion", "floor_changed")
PRIOR_DEG = 1.0
EXAMPLES = 8


@dataclass(frozen=True)
class Candidate:
    seconds: float | None
    status: str


def _angle(frame: tuple[float, ...]) -> float:
    return atan2(frame[cd.MOTOR_FEATURES], frame[cd.MOTOR_FEATURES+1])


def _normalized_orbit(frame: tuple[float, ...]) -> tuple[float, float] | None:
    """Return (cos theta, sin theta) from the HUD's (sin theta, cos theta)."""
    x = frame[cd.MOTOR_FEATURES]
    y = frame[cd.MOTOR_FEATURES+1]
    length = (x*x + y*y)**0.5
    if not isfinite(length) or length < .5:
        return None
    return y/length, x/length


def projective_step(a: tuple[float, ...], b: tuple[float, ...]) -> float | None:
    """Signed angle modulo pi in [-pi/2,pi/2]; invariant to either planet swap.

    delta = (cos(theta_b)+i sin(theta_b)) * conj(z_a).
    Use delta**2 and halve its principal argument. This is the RP1 step,
    not a detection of whether a planet swap occurred.
    """
    start, end = _normalized_orbit(a), _normalized_orbit(b)
    if start is None or end is None:
        return None
    c0, s0 = start
    c1, s1 = end
    real = c1*c0+s1*s0
    imag = s1*c0-c1*s0
    return .5*atan2(2.*real*imag, real*real-imag*imag)


def _snapshot(
    hud: tuple[tuple[float, ...], ...], k: int,
) -> dict:
    old2, old, now = hud[k-2:k+1]
    prior = projective_step(old2,old)
    current = projective_step(old,now)
    ordinary_previous_step = cd._wrap(_angle(old)-_angle(old2))
    ordinary_current_step = cd._wrap(_angle(now)-_angle(old))
    doubled_consistency = (
        abs(cd._wrap(2.*(current-prior)))*90./pi
        if prior is not None and current is not None else None
    )
    bpm = cd.HUD_BPM_REFERENCE * (2.**float(now[cd.HUD_OFFSET]))
    expected = pi*bpm/60.*p.CONTROL_DT
    if not isfinite(bpm) or bpm <= 0. or not isfinite(expected) or expected <= 0.:
        raise ValueError("Invalid visible Tile BPM")
    return {
        "prior_step":prior,
        "current_step":current,
        "ordinary_prior_step":ordinary_previous_step,
        "ordinary_current_step":ordinary_current_step,
        "projective_step_difference_abs_deg":doubled_consistency,
        "projective_prior_abs_deg":abs(prior)*180./pi if prior is not None else None,
        "projective_current_abs_deg":abs(current)*180./pi if current is not None else None,
        "current_speed_ratio":abs(current)/expected if current is not None else None,
        "prior_speed_ratio":abs(prior)/expected if prior is not None else None,
        "tile_bpm":bpm,
        "alias_risk":1.5*expected >= pi/2,
    }


def causal_projective_candidates(
    hud: tuple[tuple[float, ...], ...],
) -> tuple[dict[str, tuple[Candidate, ...]], tuple[str, ...],
           tuple[dict | None, ...]]:
    """Make causal proposals on every HUD frame, before any oracle scoring."""
    if not hud or any(len(row) != p.INPUT_DIM for row in hud):
        raise ValueError("Require nonempty [T,263] HUD")
    if any(not all(isfinite(v) for v in row) for row in hud):
        raise ValueError("Nonfinite HUD input")
    candidates = {name: [] for name in CANDIDATES}
    statuses = []
    snapshots = []
    for k,now in enumerate(hud):
        original = (
            cd.geometry_one(hud[k-1],now)[1]
            if k else "no_previous_hud_frame"
        )
        statuses.append(original)
        snap = _snapshot(hud,k) if k >= 2 else None
        snapshots.append(snap)
        proposals = {
            name: Candidate(None,"not_target_guard")
            for name in CANDIDATES
        }
        if original in TARGET_REASONS:
            if snap is None:
                proposals = {
                    name: Candidate(None,"insufficient_history")
                    for name in CANDIDATES
                }
            elif snap["alias_risk"]:
                proposals = {
                    name: Candidate(None,"alias_risk")
                    for name in CANDIDATES
                }
            elif snap["current_step"] is None or snap["prior_step"] is None:
                proposals = {
                    name: Candidate(None,"invalid_orbit")
                    for name in CANDIDATES
                }
            elif not .5 <= snap["current_speed_ratio"] <= 1.5:
                proposals = {
                    name: Candidate(None,"projective_speed_inconsistent")
                    for name in CANDIDATES
                }
            else:
                previous = list(hud[k-1])
                if original == "floor_changed":
                    # Lift ONLY floor-changed guard in the synthesized previous.
                    start = cd.SLOT_START+cd.CURRENT_SLOT*cd.FLOOR_DIM
                    previous[start:start+cd.FLOOR_DIM] = (
                        now[start:start+cd.FLOOR_DIM]
                    )
                # Replicate orbit motion at theta[k] with projective speed,
                # using the original frozen Geometry implementation.
                before_angle = _angle(now)-snap["current_step"]
                previous[cd.MOTOR_FEATURES] = sin(before_angle)
                previous[cd.MOTOR_FEATURES+1] = cos(before_angle)
                value,reason = cd.geometry_one(tuple(previous),now)
                if value is not None and (not isfinite(value) or value < 0.):
                    raise RuntimeError("Nonfinite/negative Geometry prediction")
                proposals["projective_speed"] = Candidate(
                    value,"eligible" if value is not None else reason)
                if not .5 <= snap["prior_speed_ratio"] <= 1.5:
                    proposals["projective_prior_agreement"] = Candidate(
                        None,"prior_speed_inconsistent")
                elif snap["projective_step_difference_abs_deg"] > PRIOR_DEG:
                    proposals["projective_prior_agreement"] = Candidate(
                        None,"prior_projective_step_mismatch")
                else:
                    proposals["projective_prior_agreement"] = Candidate(
                        value,"eligible" if value is not None else reason)
        for name in CANDIDATES:
            candidates[name].append(proposals[name])
    return ({name:tuple(rows) for name,rows in candidates.items()},
            tuple(statuses),tuple(snapshots))


def _angles(values: list[float]) -> dict:
    if not values:
        return {
            "count":0,"mean_abs_deg":None,"p95_abs_deg":None,
            "median_abs_deg":None,"within_1deg":0,
        }
    ordered=sorted(abs(x) for x in values)
    return {
        "count":len(values),
        "mean_abs_deg":mean(ordered),
        "median_abs_deg":median(ordered),
        "p95_abs_deg":ordered[ceil(.95*len(ordered))-1],
        "within_1deg":sum(x <= 1. for x in ordered),
    }


def _empty_bucket() -> dict:
    return {
        "proposed":Counter(),
        "timing":[],
        "matched_gru":[],
        "matched_hybrid":[],
        "near_timing":[],
        "angular_residual_deg":[],
        "angular_speed_ratio":[],
        "examples":[],
        "total":0,
    }


def _finish_bucket(bucket:dict) -> dict:
    examples=sorted(
        bucket["examples"],key=lambda x:abs(x["error_ms"]),reverse=True
    )
    return {
        "total_pairs":bucket["total"],
        "eligible":len(bucket["timing"]),
        "statuses":dict(sorted(bucket["proposed"].items())),
        "candidate_timing":cd._summary(bucket["timing"]),
        "matched_frozen_gru":cd._summary(bucket["matched_gru"]),
        "matched_frozen_hybrid":cd._summary(bucket["matched_hybrid"]),
        "near_timing":cd._summary(bucket["near_timing"]),
        "double_angle_step_consistency":_angles(bucket["angular_residual_deg"]),
        "projective_speed_ratio":{
            "count":len(bucket["angular_speed_ratio"]),
            "mean":mean(bucket["angular_speed_ratio"]) if bucket["angular_speed_ratio"] else None,
        },
        "largest_errors":examples[:EXAMPLES],
    }


def _score_split(
    group: list[stage_t.TimingSequence],
    traces: dict[int, hybrid.HybridTrace],
) -> dict:
    """Oracle after full-sequence HUD proposals. Never choose via oracle labels."""
    labels = (
        "all_switch", "same_event",
        "original_motion", "original_floor",
        "floor_halfturn_explained", "floor_halfturn_unexplained",
    )
    metrics={name:{label:_empty_bucket() for label in labels}
             for name in CANDIDATES}
    carried_baseline={label:[] for label in labels}
    status_counts={"switch":Counter(),"same_event":Counter()}
    baseline_counts=Counter()
    totals=Counter()
    per_anchor=[]
    for seq in group:
        frames=tuple(tuple(float(x) for x in row)
                     for row in seq.visible.detach().cpu().tolist())
        proposals,original,snapshots=causal_projective_candidates(frames)
        old_carried,_=first.first_frame_candidates(frames)
        trace=traces[seq.anchor]
        if len(trace.fallback_s)!=len(frames):
            raise RuntimeError("Frozen Hybrid frame count mismatch")
        anchor_total=Counter()
        for k in range(1,len(frames)):
            if not(bool(seq.mask[k-1]) and bool(seq.mask[k])):
                continue
            a,b=seq.event_indices[k-1:k+1]
            if a<0 or b<0:
                continue
            is_switch=a!=b
            if is_switch and k<2:
                raise RuntimeError("Prior halfturn cohort cannot label frame 1 switch")
            primary="all_switch" if is_switch else "same_event"
            if is_switch:
                status_counts["switch"][original[k]]+=1
                buckets=["all_switch"]
                if original[k]=="floor_changed":
                    buckets.append("original_floor")
                    # These two diagnostic subsets are ORACLE-labelled only.
                    # The predictor has ALREADY run on the entire HUD stream.
                    half=halfturn.hud_halfturn_snapshot(frames,k)
                    if half["halfturn_minus_prior_absolute_deg"]>1.:
                        buckets.append("floor_halfturn_unexplained")
                    else:
                        buckets.append("floor_halfturn_explained")
                elif original[k]=="inconsistent_orbit_motion":
                    buckets.append("original_motion")
            else:
                buckets=["same_event"]
                status_counts["same_event"][original[k]]+=1
            totals[primary]+=1
            anchor_total[primary]+=1
            true_s=float(seq.remaining_s[k])
            carried=old_carried["carried_previous_step"][k]
            for label in buckets:
                if carried.estimate_s is not None:
                    carried_baseline[label].append(
                        (carried.estimate_s-true_s)*1000.)
                if carried.estimate_s is not None:
                    baseline_counts[label]+=1
            for name in CANDIDATES:
                c=proposals[name][k]
                for label in buckets:
                    bucket=metrics[name][label]
                    bucket["total"]+=1
                    bucket["proposed"][c.status]+=1
                    snap = snapshots[k]
                    if snap is not None and snap["projective_step_difference_abs_deg"] is not None:
                        bucket["angular_residual_deg"].append(
                            snap["projective_step_difference_abs_deg"])
                    if snap is not None and snap["current_speed_ratio"] is not None:
                        bucket["angular_speed_ratio"].append(
                            snap["current_speed_ratio"])
                    if c.seconds is None:
                        continue
                    error=(c.seconds-true_s)*1000.
                    bucket["timing"].append(error)
                    bucket["matched_gru"].append(
                        (trace.fallback_s[k]-true_s)*1000.)
                    bucket["matched_hybrid"].append(
                        (trace.chosen_s[k]-true_s)*1000.)
                    if bool(seq.near[k]):
                        bucket["near_timing"].append(error)
                    bucket["examples"].append({
                        "anchor":seq.anchor,"frame":k,"original_status":original[k],
                        "error_ms":error,"oracle_remaining_ms":true_s*1000.,
                        "predicted_remaining_ms":c.seconds*1000.,
                        "gru_error_ms":(trace.fallback_s[k]-true_s)*1000.,
                        "angle_residual_deg":snapshots[k]["projective_step_difference_abs_deg"],
                    })
        per_anchor.append({
            "anchor":seq.anchor,"chart_sha256":seq.chart_sha256,
            "true_switches":anchor_total["all_switch"],
            "same_event_pairs":anchor_total["same_event"],
        })
    if not totals["all_switch"] or not totals["same_event"]:
        raise RuntimeError("No valid switch/same-event evaluation pairs")
    scored={name:{label:_finish_bucket(item) for label,item in buckets.items()}
            for name,buckets in metrics.items()}
    for name,buckets in scored.items():
        buckets["false_positive_rate"] = (
            buckets["same_event"]["eligible"]/totals["same_event"]
        )
        positive=buckets["all_switch"]["eligible"]
        negative=buckets["same_event"]["eligible"]
        buckets["descriptive_precision"] = (
            positive/(positive+negative) if positive+negative else None
        )
    return {
        "pair_totals":dict(totals),
        "original_geometry_reason_at_switch":dict(sorted(status_counts["switch"].items())),
        "original_geometry_reason_at_same_event":dict(sorted(status_counts["same_event"].items())),
        "classified_floor_halfturn_explained":metrics["projective_speed"]["floor_halfturn_explained"]["total"],
        "classified_floor_halfturn_unexplained":metrics["projective_speed"]["floor_halfturn_unexplained"]["total"],
        "prior_carried_previous_step":{
            label:{"eligible":baseline_counts[label],
                   "error":cd._summary(carried_baseline[label])}
            for label in labels
        },
        "double_angle_candidates":scored,
        "per_anchor":per_anchor,
        "caveat":"Floor subgroups are oracle-posthoc classification; never a feature or candidate selector.",
    }


def run_projective_orbit_audit(config:p.PrestigeConfig) -> dict:
    root=transfer.output_dir(config)
    output=root/REPORT
    source=distance.prepare(config,output_paths=(output,))
    predecessor=root/halfturn.REPORT
    if not predecessor.is_file():
        raise FileNotFoundError("Run 180-degree angular audit first")
    previous_sha=p.sha256(predecessor)
    previous=json.loads(predecessor.read_text(encoding="utf-8"))
    frozen, near_report, fingerprints=near._restore_near_only(config,source)
    near_sha=hybrid._validate_prior(root,source,near_report,fingerprints,frozen)
    initial={**fingerprints,"near_only_countdown":near_sha}
    scores,switches,results={},{},{}
    for split,group in (("train_fit",source.fit),("train_heldout",source.heldout)):
        traces={seq.anchor:hybrid.infer_hybrid(seq.visible,frozen) for seq in group}
        scores[split]=hybrid._aggregate_scores(group,traces,source.reference)
        switches[split]=hybrid._switching_scores(group,traces)
        results[split]=_score_split(group,traces)
        original=previous["results"][split]
        result=results[split]
        if (result["pair_totals"]["all_switch"]!=original["total_labelled_switches_with_prior_step"]
            or result["original_geometry_reason_at_switch"]!=original["original_reason_counts"]
            or result["pair_totals"]["all_switch"]!=switches[split]["event_switches_observed"]
            or result["pair_totals"]["same_event"]!=switches[split]["same_oracle_event_pairs"]
            or result["classified_floor_halfturn_explained"]!=
               original["cohorts"]["original_floor_changed"]["halfturn_residual_threshold_counts"]["1.0"]
            or result["classified_floor_halfturn_unexplained"]!=
               original["cohorts"]["original_floor_changed"]["events"]-
               original["cohorts"]["original_floor_changed"]["halfturn_residual_threshold_counts"]["1.0"]):
            raise RuntimeError("Frozen halfturn or Hybrid cohort counts drifted: "+split)
    hybrid_sha=switch._check_saved_hybrid(
        root/hybrid.REPORT,config,source,initial,scores,switches)
    switch_sha=first._verify_prior_report(
        source,config,initial,scores,switches)
    required={
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
    }
    if (previous.get("format")!=halfturn.FORMAT
        or previous.get("role")!="Train20 fit / chart-disjoint heldout ONLY"
        or previous.get("config_sha256")!=config.signature()
        or previous.get("train_partition")!=source.partition
        or previous.get("source_sha256")!=required
        or previous.get("no_training") is not True
        or previous.get("motor_student_training_permitted") is not False
        or previous.get("validation_evaluated") is not False
        or previous.get("final_evaluated") is not False
        or previous.get("original_stage_a_gate_overridden") is not False
        or previous.get("automatically_adopted") is not False
        or p._protect_old_best(config)!=source.original_best_sha
        or p.sha256(predecessor)!=previous_sha):
        raise RuntimeError("Frozen angular audit, Stage A, or checkpoints changed")
    report={
        "format":FORMAT,
        "role":"Read-only Train20 expert HUD fit / chart-disjoint heldout",
        "principle":"RP1: orbit theta equivalent to theta+pi; estimate modulo-pi signed angular step",
        "model_inputs":"At most three causal 263D visible HUD frames",
        "evaluator_only_labels":True,
        "floor_halfturn_unexplained_threshold_deg":1.0,
        "prior_agreement_threshold_deg":PRIOR_DEG,
        "alias_safety":"Abstain when 1.5 times expected Tile-BPM angular step reaches pi/2",
        "original_geometry_changed":False,
        "hybrid_changed":False,
        "adoption_permitted":False,
        "no_training":True,
        "new_training":False,
        "motor_student_training_permitted":False,
        "validation_evaluated":False,
        "final_evaluated":False,
        "original_stage_a_gate_overridden":False,
        "config_sha256":config.signature(),
        "source_sha256":{**required,"halfturn_angle_audit":previous_sha},
        "train_partition":source.partition,
        "results":results,
        "limitations":[
            "Only Train20 prior-revisited chart-disjoint heldout; not independent confirmation",
            "Original floor_changed must be synthesized for a diagnostic; not a safe deployed guard bypass",
            "Projective representation destroys planet parity and aliases angular travel >=90 degrees per tick",
            "No true event tracker; same-event eligibility measured posthoc as potential false positive",
            "A low conditional MAE does not imply safe adoption or target-identity correctness",
            "Stage A FAIL; original Geometry, Hybrid, checkpoints and Validation/FINAL untouched",
        ],
    }
    root.mkdir(parents=True,exist_ok=True)
    p._no_replace_json(output,report)
    return report
