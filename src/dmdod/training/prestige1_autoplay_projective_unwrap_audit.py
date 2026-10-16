from __future__ import annotations

"""Stage T read-only: BPM-guided RP1 orbit unwrapping on re-captured v2 HUD.

Raw Tile BPM is a human-visible observation encoded with an unclipped log
transform. Chart/target truth is used ONLY after all causal HUD proposals
are produced and v2 frames are independently matched to frozen v1 frames.
The pretrained Stage A/Stage T models, Geometry and Hybrid are NEVER updated.
"""

from collections import Counter
from dataclasses import dataclass, replace
from math import atan2, ceil, isfinite, pi
import json
from statistics import mean

import torch

from dmdod.envs.n_key import encode_n_key_hud_real_chart_observation
from dmdod.features.hud_features import encode_bpm
from dmdod.motor.capacity import CenterFirstNKeyTeacher
from . import prestige1 as p
from . import prestige1_alignment_audit as alignment
from . import prestige1_autoplay_countdown as cd
from . import prestige1_autoplay_diagnostics as frozen_source
from . import prestige1_autoplay_projective_orbit_audit as rp1
from . import prestige1_autoplay_readout as stage_t
from . import prestige1_transfer as transfer
from .n_key import _make_env, _teacher_action

FORMAT = "prestige1-stage-t-projective-unwrapped-v2-hud-train20-readonly-v1"
REPORT = "stage_timing_autoplay_projective_unwrapped_v2_train20.json"
CANDIDATES = ("unwrap_speed", "unwrap_prior_agreement")
MAX_EXAMPLES = 12
# No tuning/sweep. Fixed conservative tolerances for a constant-speed
# renderer and float32 HUD (smaller than original 0.5-1.5 Geometry band).
MAX_SPEED_REL_ERROR = 0.10
MIN_UNIQUENESS_REL_GAP = 0.10
PRIOR_AGREEMENT_DEG = 1.0
COMPARE_HUD_TOL = 1e-5
ERROR_THRESHOLDS_MS = (0.01, 0.1, 1.0, 2.0, 5.0)
BPM_BINS = ("under_1920", "1920_to_3000", "3000_to_6000",
            "6000_to_10000", "at_least_10000")


@dataclass(frozen=True)
class Step:
    radians: float | None
    status: str
    expected_abs_deg: float | None = None
    observed_projective_deg: float | None = None
    competitor_gap_deg: float | None = None


@dataclass(frozen=True)
class Proposal:
    seconds: float | None
    status: str
    step: Step


def _angle(hud: tuple[float, ...]) -> float:
    return atan2(hud[cd.MOTOR_FEATURES], hud[cd.MOTOR_FEATURES+1])


def _bpm(hud: tuple[float, ...]) -> float | None:
    encoded = hud[cd.HUD_OFFSET]
    try:
        bpm = 120.0 * (2.0 ** encoded)
    except OverflowError:
        return None
    return bpm if isfinite(bpm) and bpm > 0. else None


def _bin(bpm:float)->str:
    if bpm < 1920.: return BPM_BINS[0]
    if bpm < 3000.: return BPM_BINS[1]
    if bpm < 6000.: return BPM_BINS[2]
    if bpm < 10000.: return BPM_BINS[3]
    return BPM_BINS[4]


def unwrap_step(
    before: tuple[float, ...], now: tuple[float, ...],
) -> Step:
    """Choose unique signed r+k*pi closest in MAGNITUDE to HUD Tile BPM.

    No event labels, chart floor speeds, oracle times or future observations.
    The floor's is_ccw flag is NOT available in the 263D HUD. Therefore the
    ambiguous +speed/-speed case must abstain, never assume a direction.
    """
    r = rp1.projective_step(before, now)
    bpm = _bpm(now)
    if r is None:
        return Step(None, "invalid_orbit")
    if bpm is None:
        return Step(None, "invalid_bpm")
    # If the visible Tile BPM changed within the observation interval,
    # instantaneous current Tile BPM does not describe its integral.
    if abs(now[cd.HUD_OFFSET] - before[cd.HUD_OFFSET]) > 1e-6:
        return Step(None, "bpm_changed_across_sample")
    expected = pi*bpm/60. * p.CONTROL_DT
    if not isfinite(expected) or expected <= 0.:
        return Step(None, "invalid_expected_step")
    # k need only span +-expected/pi plus a buffer; no artificial BPM cap.
    max_k = ceil(expected/pi) + 3
    candidates = sorted(
        ((abs(abs(r+k*pi)-expected), r+k*pi, k)
         for k in range(-max_k, max_k+1)),
        key=lambda candidate:(candidate[0],abs(candidate[1]),candidate[2])
    )
    first, second = candidates[0], candidates[1]
    gap = second[0]-first[0]
    fields = (expected*180/pi, r*180/pi, gap*180/pi)
    if first[0] > MAX_SPEED_REL_ERROR*expected:
        return Step(None, "speed_not_credible", *fields)
    if gap <= MIN_UNIQUENESS_REL_GAP*expected:
        return Step(None, "sign_or_alias_ambiguous", *fields)
    return Step(first[1], "eligible", *fields)


def _projective_geometry(
    previous:tuple[float,...], current:tuple[float,...],
    delta:float, *, original_status:str,
) -> tuple[float|None,str]:
    """Freeze all Geometry guards except diagnostic floor-switch exemption.

    Unlike frozen geometry_one, the speed is NOT reduced modulo 2pi
    and a >180-degree per-frame step can be represented.
    """
    before=cd._visible_slot(previous,cd.CURRENT_SLOT)
    floor=cd._visible_slot(current,cd.CURRENT_SLOT)
    behind=cd._visible_slot(current,cd.CURRENT_SLOT-1)
    ahead=cd._visible_slot(current,cd.NEXT_SLOT)
    if any(v[-1] < .5 for v in (before,floor,behind,ahead)):
        return None,"unavailable_ordinary_floor"
    changed=any(abs(x-y)>1e-6 for x,y in zip(before,floor))
    if changed and original_status!="floor_changed":
        return None,"floor_changed"
    if any(floor[j]>.5 or ahead[j]>.5 for j in (6,7)):
        return None,"midspin_or_multiplanet"
    if any(marker>.5 for tile in (floor,ahead)
           for marker in tile[cd.SKIP_MARKER_FROM:cd.SKIP_MARKER_TO]):
        return None,"special_event"
    entry=atan2(floor[2],floor[3])
    exit_angle=atan2(floor[4],floor[5])
    if abs(cd._wrap(exit_angle-entry)) < 1e-4:
        return None,"ambiguous_full_turn"
    if max(abs(v) for v in current[cd.MOTOR_FEATURES:cd.MOTOR_FEATURES+2])<.5:
        return None,"invalid_orbit"
    if abs(delta)<=1e-5:
        return None,"stopped_orbit"
    bpm=_bpm(current)
    if bpm is None:
        return None,"invalid_bpm"
    angular_rate=pi*bpm/60.
    if not .5<=abs(delta)/(angular_rate*p.CONTROL_DT)<=1.5:
        return None,"inconsistent_orbit_motion"
    sign=1. if delta>0 else -1.
    # This still assumes one simple target on the current arc. A full-turn,
    # multi-note skip or hidden phase count is NOT solved by unwrapping.
    distance=(sign*(exit_angle-_angle(current)))%(2*pi)
    seconds=distance/angular_rate
    if not isfinite(seconds) or seconds<0.:
        return None,"nonfinite_result"
    return seconds,"eligible_simple_floor"


def causal_candidates(
    frames:tuple[tuple[float,...],...],
) -> tuple[dict[str,tuple[Proposal,...]],tuple[str,...]]:
    if not frames or any(len(h)!=p.INPUT_DIM for h in frames):
        raise ValueError("Require nonempty 263D HUD frame sequence")
    if any(not all(isfinite(x) for x in h) for h in frames):
        raise ValueError("Nonfinite HUD")
    rows={name:[] for name in CANDIDATES}
    original_statuses=[]
    empty_step=Step(None,"insufficient_history")
    for t,hud in enumerate(frames):
        orig=cd.geometry_one(frames[t-1],hud)[1] if t else "no_previous_hud_frame"
        original_statuses.append(orig)
        choices={name:Proposal(None,"not_original_guard",empty_step) for name in CANDIDATES}
        expected_bpm=_bpm(hud)
        high_speed_ordinary = (
            expected_bpm is not None and expected_bpm>=3000. and
            orig in ("eligible_simple_floor","stopped_orbit")
        )
        if orig in ("floor_changed","inconsistent_orbit_motion") or high_speed_ordinary:
            if t<2:
                choices={name:Proposal(None,"insufficient_history",empty_step)
                         for name in CANDIDATES}
            else:
                step=unwrap_step(frames[t-1],hud)
                if step.radians is None:
                    choices={name:Proposal(None,step.status,step) for name in CANDIDATES}
                else:
                    value,status=_projective_geometry(
                        frames[t-1],hud,step.radians,original_status=orig)
                    if status=="eligible_simple_floor" and value is None:
                        raise RuntimeError("Geometry returned invalid eligible state")
                    output=Proposal(value, "eligible" if value is not None else status,step)
                    choices["unwrap_speed"]=output
                    prior=unwrap_step(frames[t-2],frames[t-1])
                    if prior.radians is None:
                        choices["unwrap_prior_agreement"]=Proposal(None,
                            "prior_"+prior.status,step)
                    elif abs(step.radians-prior.radians)> (
                            PRIOR_AGREEMENT_DEG*pi/180.):
                        # Wrap difference only because constant signed speed could
                        # vary by 2pi with a different k; report abstention.
                        choices["unwrap_prior_agreement"]=Proposal(
                            None,"prior_step_mismatch",step)
                    else:
                        choices["unwrap_prior_agreement"]=output
        for name in CANDIDATES:
            rows[name].append(choices[name])
    return {name:tuple(v) for name,v in rows.items()},tuple(original_statuses)


def _capture_v2_visible(named, config:p.PrestigeConfig,
                        expected:stage_t.TimingSequence) -> stage_t.TimingSequence:
    """Re-run exactly the same expert state trajectory, with ONLY BPM v2.

    Do not query target time or oracle when constructing visible features.
    Labels and masks are taken from independently frozen v1 capture AFTER
    full per-frame non-BPM HUD equality checks.
    """
    env=_make_env(named.segment,key_count=p.KEY_COUNT,
                  control_dt_s=p.CONTROL_DT,physics_dt_s=p.PHYSICS_DT)
    actuator=CenterFirstNKeyTeacher(p.KEY_COUNT)
    observation=env.reset()
    saved=[]
    last_clock=-float("inf")
    expected_length=int(expected.visible.shape[0])
    for t in range(expected_length):
        hud=encode_n_key_hud_real_chart_observation(observation)
        if len(hud)!=p.INPUT_DIM or not all(isfinite(x) for x in hud):
            raise RuntimeError("v2 HUD contract changed")
        if abs(hud[cd.HUD_OFFSET]-encode_bpm(observation.tile_bpm))>1e-9 or (
            abs(hud[cd.HUD_OFFSET+1]-encode_bpm(observation.real_bpm))>1e-9
        ):
            raise RuntimeError("v2 encoded BPM mismatches the human-visible HUD")
        old=expected.visible[t].detach().cpu().tolist()
        for j,(x,y) in enumerate(zip(hud,old)):
            if j not in (cd.HUD_OFFSET,cd.HUD_OFFSET+1) and abs(x-y)>COMPARE_HUD_TOL:
                raise RuntimeError(
                    f"v2 replay altered non-BPM HUD anchor={expected.anchor} t={t} j={j}")
        # The old codec may differ ONLY by historical saturation at +-4.
        for j in (cd.HUD_OFFSET,cd.HUD_OFFSET+1):
            new_value=hud[j]
            old_value=old[j]
            if abs(new_value-old_value)>COMPARE_HUD_TOL and not (
                (abs(old_value-4.)<COMPARE_HUD_TOL and new_value>4.)
                or (abs(old_value+4.)<COMPARE_HUD_TOL and new_value< -4.)
            ):
                raise RuntimeError("v2 BPM differs outside frozen v1 clipping")
        now=float(env.privileged_episode_time_s())
        if now<=last_clock:
            raise RuntimeError("Non-monotonic v2 expert capture")
        last_clock=now
        saved.append(hud)
        act=_teacher_action(env,actuator,observation,lead_s=config.lead_s)
        result=env.step(act)
        observation=result.observation
        if result.done and t!=expected_length-1:
            raise RuntimeError("v2 expert capture ended before frozen HUD replay")
        if t==expected_length-1 and not result.done:
            raise RuntimeError("v2 expert capture exceeded frozen HUD length")
    visible=torch.tensor(saved,dtype=torch.float32,device=expected.visible.device)
    return replace(expected,visible=visible)


def _metric(errors:list[float])->dict:
    return cd._summary(errors)


def _score(
    group:list[stage_t.TimingSequence],
)->dict:
    # Future/oracle state consulted only after predictor proposed all HUD frames.
    totals=Counter()
    scored={}
    anchors=[]
    for name in CANDIDATES:
        scored[name]={label:{"status":Counter(),"errors":[],
                             "samples":[],"total":0}
                      for label in ("switch","same_event","above_1920",
                                    "above_3000","at_or_below_1920",
                                    *BPM_BINS)}
    original=Counter()
    clip_counts=Counter()
    for seq in group:
        frames=tuple(tuple(float(x) for x in h)
                     for h in seq.visible.detach().cpu().tolist())
        proposals,original_reasons=causal_candidates(frames)
        anchor=Counter()
        for k in range(1,len(frames)):
            if not (bool(seq.mask[k-1]) and bool(seq.mask[k])):
                continue
            a,b=seq.event_indices[k-1:k+1]
            if a<0 or b<0:
                continue
            kind="switch" if a!=b else "same_event"
            totals[kind]+=1
            anchor[kind]+=1
            bpm=_bpm(frames[k])
            if bpm is None:
                raise RuntimeError("Nonfinite observable BPM")
            if kind=="switch":
                original[original_reasons[k]]+=1
            if bpm>1920.+.01:
                clip_counts[kind]+=1
            categories=[kind]
            if kind=="switch":
                categories.append(_bin(bpm))
                categories.append("above_1920" if bpm>1920.+.01
                                  else "at_or_below_1920")
                if bpm>3000.+.01:
                    categories.append("above_3000")
            for name in CANDIDATES:
                c=proposals[name][k]
                for label in categories:
                    bucket=scored[name][label]
                    bucket["total"]+=1
                    bucket["status"][c.status]+=1
                    if c.seconds is None:continue
                    error=(c.seconds-float(seq.remaining_s[k]))*1000.
                    bucket["errors"].append(error)
                    bucket["samples"].append({
                        "anchor":seq.anchor,"frame":k,
                        "bpm":bpm,"error_ms":error,
                        "predicted_ms":c.seconds*1000.,
                        "oracle_ms":float(seq.remaining_s[k])*1000.,
                        "unwrapped_step_deg":c.step.radians*180/pi,
                        "expected_deg":c.step.expected_abs_deg,
                        "projective_observed_deg":c.step.observed_projective_deg,
                        "original_status":original_reasons[k],
                    })
        anchors.append({"anchor":seq.anchor,"sha256":seq.chart_sha256,**anchor})
    results={}
    for name,by_cohort in scored.items():
        results[name]={}
        for label,row in by_cohort.items():
            result={
                "total":row["total"],"eligible":len(row["errors"]),
                "status":dict(sorted(row["status"].items())),
                "timing":_metric(row["errors"]),
                "above_error_ms":{
                    str(v):sum(abs(err)>v for err in row["errors"])
                    for v in ERROR_THRESHOLDS_MS},
                "worst":sorted(row["samples"],
                               key=lambda v:abs(v["error_ms"]),reverse=True)[:MAX_EXAMPLES],
            }
            results[name][label]=result
        num=results[name]["switch"]["eligible"]
        denominator=num+results[name]["same_event"]["eligible"]
        results[name]["precision_descriptive"]=num/denominator if denominator else None
        results[name]["same_event_false_positive_rate"]=(
            results[name]["same_event"]["eligible"]/totals["same_event"]
            if totals["same_event"] else None)
    return {
        "oracle_pair_counts":dict(totals),
        "original_geometry_switch_reasons":dict(sorted(original.items())),
        "above_1920_pairs":dict(clip_counts),
        "candidate_results":results,
        "per_anchor":anchors,
    }


def run_unwrap_audit(config:p.PrestigeConfig)->dict:
    root=transfer.output_dir(config)
    path=root/REPORT
    source=frozen_source.prepare(config,output_paths=(path,))
    report_path=root/rp1.REPORT
    if not report_path.is_file():
        raise FileNotFoundError("Frozen RP1 report required for baselines")
    old_sha=p.sha256(report_path)
    old=json.loads(report_path.read_text(encoding="utf-8"))
    if (old.get("format")!=rp1.FORMAT or
        old.get("config_sha256")!=config.signature() or
        old.get("train_partition")!=source.partition or
        old.get("no_training") is not True or
        old.get("motor_student_training_permitted") is not False or
        old.get("validation_evaluated") is not False or
        old.get("final_evaluated") is not False or
        old.get("original_stage_a_gate_overridden") is not False):
        raise RuntimeError("Frozen RP1 provenance changed")
    # Independently recapture the same Train20 with the default (unclipped) v2
    # encoder and verify identical trajectories except for the two BPM slots.
    selected,manifest=alignment._train_manifest(config)
    if manifest!=source.stage_report["train_manifest"]:
        raise RuntimeError("Train manifest has changed")
    frozen_by_id={s.anchor:s for s in [*source.fit,*source.heldout]}
    v2=[]
    for i,named in enumerate(selected,1):
        base=frozen_by_id.get(i)
        if base is None or named.chart_sha256!=base.chart_sha256:
            raise RuntimeError("Train20 anchor drift")
        sample=_capture_v2_visible(named,config,base)
        v2.append(sample)
        print(f"RP1 unwrap v2 capture Train {i:02d}/20: {sample.visible.shape[0]} frames",
              flush=True)
    fit,heldout,partition=p.split_stage_a_train(
        v2,{"train":source.stage_report["train_manifest"]},seed=config.seed)
    if partition!=source.partition:
        raise RuntimeError("Frozen Train20 partition drift")
    results={}
    for split,group in (("train_fit",fit),("train_heldout",heldout)):
        result=_score(group)
        previously=old["results"][split]
        if (result["oracle_pair_counts"]["switch"]!=
              previously["pair_totals"]["all_switch"] or
            result["oracle_pair_counts"]["same_event"]!=
              previously["pair_totals"]["same_event"] or
            result["original_geometry_switch_reasons"].get("floor_changed",0)!=
              previously["original_geometry_reason_at_switch"].get("floor_changed",0)):
            raise RuntimeError("Frozen RP1 source/geometry switch count drift")
        results[split]=result
    if (p.sha256(report_path)!=old_sha or
        p._protect_old_best(config)!=source.original_best_sha):
        raise RuntimeError("Protected report/checkpoint changed")
    output={
        "format":FORMAT,
        "role":"HUD v2 Train20 fit / chart-disjoint heldout ONLY",
        "feature":"unclipped v2 log Tile/Real BPM; 263D unchanged",
        "prediction_input":"causal 2-3-frame 263D visible HUD only",
        "oracle_used_in_prediction":False,
        "new_training":False,"no_training":True,
        "motor_student_training_permitted":False,
        "original_geometry_changed":False,
        "hybrid_changed":False,"automatically_adopted":False,
        "original_stage_a_gate_overridden":False,
        "validation_evaluated":False,"final_evaluated":False,
        "known_caveats":[
            "Same-frame switch labels are evaluator-only, never candidate inputs",
            "Stage T Train-heldout has already been repeatedly inspected",
            "An unclipped BPM may resolve aliasing but does not reveal unobserved floor/planet parity",
            "Signed speed ties, speed changes, special events and full turns abstain",
            "At >=3000 BPM unwrapping also audits originally-eligible simple floors (old wrapped direction can be wrong)",
            "Multiple notes per frame / hidden phase counts remain possible",
            "Frozen v1 artifacts are NOT fed v2 inputs and Stage A gate stays FAIL",
        ],
        "config_sha256":config.signature(),
        "train_partition":source.partition,
        "source_sha256":{**source.source_sha256,"frozen_rp1_report":old_sha},
        "constants":{
            "max_speed_relative_error":MAX_SPEED_REL_ERROR,
            "min_uniqueness_relative_gap":MIN_UNIQUENESS_REL_GAP,
            "prior_agreement_deg":PRIOR_AGREEMENT_DEG,
        },
        "results":results,
    }
    root.mkdir(parents=True,exist_ok=True)
    p._no_replace_json(path,output)
    return output
