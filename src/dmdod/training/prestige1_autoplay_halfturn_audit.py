from __future__ import annotations

"""Read-only causal HUD audit: is apparent switch angular speed a 180-degree swap?

Do not reinterpret a modulo-2pi angle discontinuity as a physical acceleration
without checking its residual after a hypothesized half-turn. No estimator,
geometry guard, network, checkpoint or motor policy is changed.
"""

from collections import Counter
import json
from math import atan2, ceil, isfinite, pi
from statistics import mean, median

from . import prestige1 as p
from . import prestige1_autoplay_countdown as cd
from . import prestige1_autoplay_diagnostics as distance
from . import prestige1_autoplay_first_frame_geometry_audit as first
from . import prestige1_autoplay_near_ablation as ablation
from . import prestige1_autoplay_near_countdown_audit as near
from . import prestige1_autoplay_hybrid_audit as hybrid
from . import prestige1_autoplay_special_motion_audit as special
from . import prestige1_autoplay_switch_hud_audit as switch
from . import prestige1_transfer as transfer

FORMAT = "prestige1-stage-t-180-degree-orbit-swap-readonly-train20-v1"
REPORT = "stage_timing_autoplay_halfturn_diagnostic_train20.json"
COHORTS = ("original_inconsistent_orbit_motion", "original_floor_changed")
THRESHOLDS_DEG = (0.1, 0.5, 1.0, 5.0)
EXAMPLES = 8


def _angle(sin_value: float, cos_value: float) -> float:
    return atan2(sin_value, cos_value)


def _rad_to_deg(angle: float) -> float:
    return angle*180.0/pi


def hud_halfturn_snapshot(
    visible: tuple[tuple[float, ...], ...], k: int,
) -> dict:
    """At frame k, use only HUD[k-2], HUD[k-1], HUD[k]. No oracle inputs."""
    if k < 2 or k >= len(visible):
        raise ValueError("Requires three successive causal HUD frames")
    if any(len(frame) != p.INPUT_DIM for frame in visible[k-2:k+1]):
        raise ValueError("Expected 263D HUD frames")
    if any(not all(isfinite(v) for v in frame) for frame in visible[k-2:k+1]):
        raise ValueError("Nonfinite HUD features")

    old2, before, now = visible[k-2:k+1]
    theta2 = _angle(old2[cd.MOTOR_FEATURES], old2[cd.MOTOR_FEATURES+1])
    theta1 = _angle(before[cd.MOTOR_FEATURES], before[cd.MOTOR_FEATURES+1])
    theta0 = _angle(now[cd.MOTOR_FEATURES], now[cd.MOTOR_FEATURES+1])
    prior = cd._wrap(theta1-theta2)
    crossing = cd._wrap(theta0-theta1)
    corrected = cd._wrap(crossing-pi)
    prior_residual = cd._wrap(corrected-prior)

    floor_before = cd._visible_slot(before, cd.CURRENT_SLOT)
    floor_now = cd._visible_slot(now, cd.CURRENT_SLOT)
    old_exit = _angle(floor_before[4],floor_before[5])
    new_entry = _angle(floor_now[2],floor_now[3])
    entry_minus_exit = cd._wrap(new_entry-old_exit)
    entry_exit_swap_residual = cd._wrap(entry_minus_exit-pi)

    # Independent "does the corrected crossing obey expected physical speed?"
    # diagnostic. This does NOT override the original Geometry guard.
    tile_bpm = special.HUD_BPM_REFERENCE * 2.**now[cd.HUD_OFFSET]
    old_tile_bpm = special.HUD_BPM_REFERENCE * 2.**before[cd.HUD_OFFSET]
    expected = pi*tile_bpm/60.0*p.CONTROL_DT
    if not all(isfinite(x) and x > 0. for x in (tile_bpm, old_tile_bpm, expected)):
        raise ValueError("Invalid visible Tile BPM")
    corrected_ratio = abs(corrected)/expected
    prior_ratio = abs(prior)/expected
    crossing_ratio = abs(crossing)/expected
    current_floor_unchanged = all(
        abs(a-b) <= 1e-6 for a,b in zip(floor_before,floor_now)
    )
    return {
        "prior_step_deg": _rad_to_deg(prior),
        "crossing_step_deg": _rad_to_deg(crossing),
        "crossing_minus_prior_deg": _rad_to_deg(cd._wrap(crossing-prior)),
        "halfturn_corrected_step_deg": _rad_to_deg(corrected),
        "halfturn_minus_prior_signed_deg": _rad_to_deg(prior_residual),
        "halfturn_minus_prior_absolute_deg": abs(_rad_to_deg(prior_residual)),
        "old_exit_to_new_entry_deg": _rad_to_deg(entry_minus_exit),
        "entry_exit_halfturn_residual_deg": _rad_to_deg(entry_exit_swap_residual),
        "crossing_speed_ratio": crossing_ratio,
        "halfturn_corrected_speed_ratio": corrected_ratio,
        "prior_speed_ratio": prior_ratio,
        "corrected_within_original_speed_band": .5 <= corrected_ratio <= 1.5,
        "prior_within_original_speed_band": .5 <= prior_ratio <= 1.5,
        "current_floor_unchanged": current_floor_unchanged,
        "tile_bpm": tile_bpm,
        "previous_tile_bpm": old_tile_bpm,
        "tile_bpm_changed": abs(now[cd.HUD_OFFSET]-before[cd.HUD_OFFSET]) > 1e-6,
    }


def _stats(values: list[float], unit: str) -> dict:
    if not values:
        return {"unit":unit,"count":0,"mean":None,"median":None,"min":None,
                "max":None,"p95":None}
    sorted_values=sorted(values)
    return {"unit":unit,"count":len(values),"mean":mean(values),
            "median":median(values),"min":sorted_values[0],
            "max":sorted_values[-1],
            "p95":sorted_values[ceil(.95*len(values))-1]}


def _cohort(rows: list[dict]) -> dict:
    absolute = [x["halfturn_minus_prior_absolute_deg"] for x in rows]
    exit_absolute = [abs(x["entry_exit_halfturn_residual_deg"]) for x in rows]
    corrected = [x["halfturn_corrected_speed_ratio"] for x in rows]
    raw = [x["crossing_speed_ratio"] for x in rows]
    return {
        "events":len(rows),
        "halfturn_minus_prior_abs_degrees":_stats(absolute,"deg"),
        "entry_exit_halfturn_abs_degrees":_stats(exit_absolute,"deg"),
        "halfturn_corrected_speed_ratio":_stats(corrected,"ratio"),
        "uncorrected_crossing_speed_ratio":_stats(raw,"ratio"),
        "halfturn_residual_threshold_counts":{
            str(t):sum(value <= t for value in absolute)
            for t in THRESHOLDS_DEG
        },
        "floor_entry_exit_threshold_counts":{
            str(t):sum(value <= t for value in exit_absolute)
            for t in THRESHOLDS_DEG
        },
        "corrected_within_original_speed_band":sum(
            x["corrected_within_original_speed_band"] for x in rows),
        "prior_within_original_speed_band":sum(
            x["prior_within_original_speed_band"] for x in rows),
        "current_floor_features_unchanged":sum(
            x["current_floor_unchanged"] for x in rows),
        "tile_bpm_changed":sum(x["tile_bpm_changed"] for x in rows),
        "worst_halfturn_residual_examples":sorted(
            rows,key=lambda x:x["halfturn_minus_prior_absolute_deg"],reverse=True
        )[:EXAMPLES],
        "largest_apparent_speed_examples":sorted(
            rows,key=lambda x:x["crossing_speed_ratio"],reverse=True
        )[:EXAMPLES],
    }


def _describe(group) -> dict:
    records=[]
    per_anchor=[]
    for seq in group:
        frames=tuple(
            tuple(float(v) for v in row)
            for row in seq.visible.detach().cpu().tolist()
        )
        # All causal estimates and original frozen statuses are computed BEFORE
        # reading evaluator-only masks / event IDs.
        observations=[hud_halfturn_snapshot(frames,k) if k>=2 else None
                      for k in range(len(frames))]
        original=[cd.geometry_one(frames[k-1],frames[k])[1] if k>=1
                  else "no_previous_hud_frame" for k in range(len(frames))]
        local=Counter()
        for k in range(2,len(frames)):
            if not (bool(seq.mask[k-1]) and bool(seq.mask[k])):
                continue
            if (seq.event_indices[k-1] < 0 or seq.event_indices[k] < 0
                    or seq.event_indices[k-1] == seq.event_indices[k]):
                continue
            local[original[k]] += 1
            records.append({
                "anchor":seq.anchor,"frame":k,"original_status":original[k],
                **observations[k],
            })
        per_anchor.append({
            "anchor":seq.anchor,"chart_sha256":seq.chart_sha256,
            "switches_with_three_frame_history":sum(local.values()),
            "original_status_counts":dict(sorted(local.items())),
        })
    if not records:
        raise RuntimeError("No label-only true switches with 3-frame history")
    cohorts={
        "original_inconsistent_orbit_motion":[
            row for row in records
            if row["original_status"]=="inconsistent_orbit_motion"
        ],
        "original_floor_changed":[
            row for row in records if row["original_status"]=="floor_changed"
        ],
    }
    return {
        "total_labelled_switches_with_prior_step":len(records),
        "original_reason_counts":dict(sorted(Counter(
            row["original_status"] for row in records).items())),
        "cohorts":{key:_cohort(rows) for key,rows in cohorts.items()},
        "per_anchor":per_anchor,
        "limitations":(
            "Posthoc labelled switch cohorts; adding pi is a diagnostic "
            "hypothesis, not a new estimator. Data may include coordinate "
            "resets other than planet swap; prove actual rendering semantics "
            "before any inference changes."
        ),
    }


def run_halfturn_audit(config:p.PrestigeConfig) -> dict:
    root=transfer.output_dir(config)
    path=root/REPORT
    source=distance.prepare(config,output_paths=(path,))
    prior_path=root/special.REPORT
    if not prior_path.is_file():
        raise FileNotFoundError("Run special/motion audit first")
    saved_sha=p.sha256(prior_path)
    saved=json.loads(prior_path.read_text(encoding="utf-8"))
    required={
        **source.source_sha256,
        "near_only_report":p.sha256(root/ablation.REPORT),
        "near_only_checkpoint":p.sha256(root/ablation.CHECKPOINT),
        "distance_audit":p.sha256(root/distance.REPORT),
        "old_countdown":p.sha256(root/cd.REPORT),
        "near_only_countdown":p.sha256(root/near.REPORT),
        "stateless_hybrid":p.sha256(root/hybrid.REPORT),
        "hud_switch_audit":p.sha256(root/switch.REPORT),
        "first_frame_geometry":p.sha256(root/first.REPORT),
    }
    if (saved.get("format")!=special.FORMAT
        or saved.get("role")!="Train20 chart-disjoint fit / heldout only"
        or saved.get("source_sha256")!=required
        or saved.get("config_sha256")!=config.signature()
        or saved.get("train_partition")!=source.partition
        or saved.get("no_training") is not True
        or saved.get("validation_evaluated") is not False
        or saved.get("final_evaluated") is not False
        or saved.get("original_stage_a_gate_overridden") is not False
        or saved.get("motor_student_training_permitted") is not False
        or saved.get("automatically_adopted") is not False):
        raise RuntimeError("Special/motion frozen provenance mismatch")
    results={}
    for key,group in (("train_fit",source.fit),("train_heldout",source.heldout)):
        result=_describe(group)
        earlier=saved["results"][key]
        if result["total_labelled_switches_with_prior_step"] != earlier["true_note_switches"]:
            raise RuntimeError(f"Note-switch count drift: {key}")
        if result["original_reason_counts"] != earlier["original_geometry_first_rejection"]:
            raise RuntimeError(f"Original Geometry reason drift: {key}")
        for cohort,original_count in (
            ("original_inconsistent_orbit_motion",
             earlier["original_inconsistent_orbit_motion"]["count"]),
            ("original_floor_changed",
             earlier["original_geometry_first_rejection"].get("floor_changed",0)),
        ):
            if result["cohorts"][cohort]["events"]!=original_count:
                raise RuntimeError(f"Halfturn cohort count drift: {key}/{cohort}")
        results[key]=result
    if (p._protect_old_best(config)!=source.original_best_sha
        or p.sha256(prior_path)!=saved_sha):
        raise RuntimeError("Protected frozen source changed")
    report={
        "format":FORMAT,
        "role":"Train20 fit / chart-disjoint heldout ONLY",
        "hypothesis":"observed orbit step = rotation step + pi (mod 2pi)",
        "predictor_inputs":"HUD[k-2:k+1] causal 263D only",
        "true_event_ids_and_masks_used_evaluator_only":True,
        "no_training":True,"geometry_guards_unchanged":True,
        "motor_student_training_permitted":False,
        "validation_evaluated":False,"final_evaluated":False,
        "original_stage_a_gate_overridden":False,
        "automatically_adopted":False,
        "config_sha256":config.signature(),
        "train_partition":source.partition,
        "source_sha256":{**required,"special_motion":saved_sha},
        "results":results,
    }
    root.mkdir(parents=True,exist_ok=True)
    p._no_replace_json(path,report)
    return report
