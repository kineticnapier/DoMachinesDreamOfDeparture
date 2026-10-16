from __future__ import annotations

"""Read-only Stage T: causal half-turn-corrected Geometry and false-positive audit.

No oracle event IDs, masks, true timestamps, privileged chart data, future
HUD frames, or trained model outputs enter the two candidate calculations.
Both candidates operate on the SAME 263D stream with frozen Geometry guards;
oracle truth is used only after all candidate estimates are calculated.
"""

from collections import Counter
from dataclasses import dataclass
import json
from math import atan2, cos, isfinite, pi, radians, sin
from pathlib import Path

from . import prestige1 as p
from . import prestige1_autoplay_countdown as cd
from . import prestige1_autoplay_diagnostics as distance
from . import prestige1_autoplay_halfturn_audit as halfturn
from . import prestige1_autoplay_hybrid_audit as hybrid
from . import prestige1_autoplay_near_ablation as ablation
from . import prestige1_autoplay_near_countdown_audit as near
from . import prestige1_autoplay_switch_hud_audit as switch
from . import prestige1_autoplay_first_frame_geometry_audit as first
from . import prestige1_autoplay_special_motion_audit as special
from . import prestige1_autoplay_readout as stage_t
from . import prestige1_transfer as transfer

FORMAT = "prestige1-stage-t-halfturn-corrected-countdown-readonly-train20-v1"
REPORT = "stage_timing_autoplay_halfturn_countdown_train20.json"
CANDIDATES = ("halfturn_speed_only", "halfturn_prior_agreement")
PRIOR_AGREEMENT_DEG = 1.0  # Fixed before reading results; NO selection sweep.
MAX_EXAMPLES = 10


@dataclass(frozen=True)
class HalfturnCandidate:
    seconds: float | None
    reason: str


def _orbit_angle(hud: tuple[float, ...]) -> float:
    return atan2(hud[cd.MOTOR_FEATURES], hud[cd.MOTOR_FEATURES+1])


def _corrected_hud_previous(
    previous: tuple[float, ...],
    current: tuple[float, ...],
    corrected_step: float,
) -> tuple[float, ...]:
    """Counterfactual previous orbit only; current and floor slots untouched."""
    values = list(previous)
    angle = _orbit_angle(current) - corrected_step
    values[cd.MOTOR_FEATURES] = sin(angle)
    values[cd.MOTOR_FEATURES+1] = cos(angle)
    return tuple(values)


def causal_halfturn_candidates(
    visible: tuple[tuple[float, ...], ...],
) -> tuple[dict[str, tuple[HalfturnCandidate, ...]], tuple[str, ...]]:
    """Compute all framewise proposals from causal HUD prefixes, without labels."""
    if not visible or any(len(frame) != p.INPUT_DIM for frame in visible):
        raise ValueError("Expected nonempty [T,263] visible HUD")
    if any(not all(isfinite(v) for v in frame) for frame in visible):
        raise ValueError("Nonfinite HUD")
    proposals = {name: [] for name in CANDIDATES}
    original = []
    for k, current in enumerate(visible):
        status = (
            cd.geometry_one(visible[k-1], current)[1]
            if k >= 1 else "no_previous_hud_frame"
        )
        original.append(status)
        result = {
            name: HalfturnCandidate(None, "not_original_motion_abstention")
            for name in CANDIDATES
        }
        # EXACT original reason, not a privileged oracle event-switch label.
        if status == "inconsistent_orbit_motion":
            if k < 2:
                result = {
                    name: HalfturnCandidate(None, "insufficient_causal_history")
                    for name in CANDIDATES
                }
            else:
                previous, previous2 = visible[k-1], visible[k-2]
                raw = cd._wrap(_orbit_angle(current) - _orbit_angle(previous))
                corrected = cd._wrap(raw - pi)
                prior = cd._wrap(_orbit_angle(previous) - _orbit_angle(previous2))
                bpm = cd.HUD_BPM_REFERENCE * (2. ** float(current[cd.HUD_OFFSET]))
                expected_step = pi*bpm/60. * p.CONTROL_DT
                if not isfinite(expected_step) or expected_step <= 0.:
                    raise RuntimeError("Invalid visible Tile BPM")
                corrected_ratio = abs(corrected)/expected_step
                prior_ratio = abs(prior)/expected_step
                if not (0.5 <= corrected_ratio <= 1.5):
                    result = {
                        name: HalfturnCandidate(None, "corrected_speed_not_credible")
                        for name in CANDIDATES
                    }
                else:
                    synthetic = _corrected_hud_previous(
                        previous, current, corrected)
                    estimate, reason = cd.geometry_one(synthetic, current)
                    if estimate is not None and (not isfinite(estimate) or estimate < 0.):
                        raise RuntimeError("Invalid halfturn Geometry estimate")
                    result["halfturn_speed_only"] = HalfturnCandidate(
                        estimate, "eligible" if estimate is not None else reason)
                    residual = abs(cd._wrap(corrected-prior))
                    if not (0.5 <= prior_ratio <= 1.5):
                        result["halfturn_prior_agreement"] = HalfturnCandidate(
                            None, "prior_speed_not_credible")
                    elif residual > radians(PRIOR_AGREEMENT_DEG):
                        result["halfturn_prior_agreement"] = HalfturnCandidate(
                            None, "prior_angle_mismatch")
                    else:
                        result["halfturn_prior_agreement"] = HalfturnCandidate(
                            estimate, "eligible" if estimate is not None else reason)
        for name in CANDIDATES:
            proposals[name].append(result[name])
    return ({key: tuple(rows) for key, rows in proposals.items()},
            tuple(original))


def _summary(errors_ms: list[float]) -> dict:
    return cd._summary(errors_ms)


def _score(group: list[stage_t.TimingSequence],
           traces: dict[int, hybrid.HybridTrace]) -> dict:
    # Candidate inference happens for every frame BEFORE any oracle field read.
    group_rows = {
        name: {
            "switch": {"candidate": [], "gru": [], "hybrid": [], "near": []},
            "same": {"candidate": [], "gru": [], "hybrid": [], "near": []},
            "statuses": {"switch": Counter(), "same": Counter()},
            "predictions_at_switch": [], "predictions_at_same": [],
        }
        for name in CANDIDATES
    }
    baseline_statuses = {"switch": Counter(), "same": Counter()}
    total = Counter()
    local_anchors = []
    for seq in group:
        frames = tuple(tuple(float(x) for x in row)
                       for row in seq.visible.detach().cpu().tolist())
        estimates, original = causal_halfturn_candidates(frames)
        frozen = traces[seq.anchor]
        if len(frozen.chosen_s) != len(frames):
            raise RuntimeError("Frozen Hybrid trace and HUD lengths mismatch")
        local = Counter()
        for k in range(1, len(frames)):
            if not (bool(seq.mask[k-1]) and bool(seq.mask[k])):
                continue
            before, after = seq.event_indices[k-1:k+1]
            if before < 0 or after < 0:
                continue
            kind = "same" if before == after else "switch"
            total[kind] += 1
            local[kind] += 1
            baseline_statuses[kind][original[k]] += 1
            target_s = float(seq.remaining_s[k])
            for name, predictions in estimates.items():
                candidate = predictions[k]
                row = group_rows[name]
                row["statuses"][kind][candidate.reason] += 1
                if candidate.seconds is None:
                    continue
                e = (candidate.seconds-target_s)*1000.
                row[kind]["candidate"].append(e)
                row[kind]["gru"].append((frozen.fallback_s[k]-target_s)*1000.)
                row[kind]["hybrid"].append((frozen.chosen_s[k]-target_s)*1000.)
                if bool(seq.near[k]):
                    row[kind]["near"].append(e)
                example = {
                    "anchor":seq.anchor, "frame":k,
                    "candidate_error_ms":e,
                    "gru_error_ms":(frozen.fallback_s[k]-target_s)*1000.,
                    "existing_hybrid_error_ms":(frozen.chosen_s[k]-target_s)*1000.,
                    "oracle_remaining_ms":target_s*1000.,
                    "candidate_remaining_ms":candidate.seconds*1000.,
                }
                row["predictions_at_switch" if kind=="switch" else
                    "predictions_at_same"].append(example)
        local_anchors.append({
            "anchor":seq.anchor,"chart_sha256":seq.chart_sha256,
            "oracle_switch_pairs":local["switch"],
            "oracle_same_event_pairs":local["same"],
        })
    if not total["switch"] or not total["same"]:
        raise RuntimeError("Need oracle switch and non-switch pairs")
    outcome = {}
    for name, rows in group_rows.items():
        switch_count = len(rows["switch"]["candidate"])
        same_count = len(rows["same"]["candidate"])
        outcome[name] = {
            "switch": {
                "eligible":switch_count,
                "coverage":switch_count/total["switch"],
                "errors":{
                    key:_summary(rows["switch"][key])
                    for key in ("candidate","gru","hybrid","near")
                },
                "reasons":dict(sorted(rows["statuses"]["switch"].items())),
                "largest_candidate_errors":sorted(
                    rows["predictions_at_switch"],
                    key=lambda x:abs(x["candidate_error_ms"]),reverse=True
                )[:MAX_EXAMPLES],
            },
            "same_event": {
                "eligible":same_count,
                "false_positive_rate":same_count/total["same"],
                "errors":{
                    key:_summary(rows["same"][key])
                    for key in ("candidate","gru","hybrid","near")
                },
                "reasons":dict(sorted(rows["statuses"]["same"].items())),
                "largest_false_positives":sorted(
                    rows["predictions_at_same"],
                    key=lambda x:abs(x["candidate_error_ms"]),reverse=True
                )[:MAX_EXAMPLES],
            },
            "descriptive_precision_on_scored_pairs":(
                switch_count/(switch_count+same_count)
                if switch_count+same_count else None
            ),
        }
    return {
        "label_only_pair_counts":dict(total),
        "original_frozen_geometry_abstention_reasons":{
            key:dict(sorted(val.items())) for key,val in baseline_statuses.items()
        },
        "candidate_results":outcome,
        "per_anchor":local_anchors,
        "limitations":[
            "All scoring uses oracle labels AFTER candidate predictions",
            "The prior 51 failures are not automatically valid next-target predictions",
            "Candidate MAE is conditional on HUD eligibility; GRU/Hybrid matched exactly",
            "A false positive is candidate eligibility on an oracle same-event pair",
            "This is NOT an adopted switch detector, Geometry change, or trained model",
        ],
    }


def run_halfturn_countdown_audit(config: p.PrestigeConfig) -> dict:
    root = transfer.output_dir(config)
    path = root / REPORT
    source = distance.prepare(config, output_paths=(path,))
    old_path = root / halfturn.REPORT
    if not old_path.is_file():
        raise FileNotFoundError("Run frozen halfturn angular audit first")
    old_sha = p.sha256(old_path)
    old = json.loads(old_path.read_text(encoding="utf-8"))
    frozen, near_report, fingerprints = near._restore_near_only(config,source)
    countdown_sha = hybrid._validate_prior(root,source,near_report,fingerprints,frozen)
    initial = {**fingerprints, "near_only_countdown":countdown_sha}
    scores, switching, results = {}, {}, {}
    for split, group in (("train_fit",source.fit),("train_heldout",source.heldout)):
        trace = {
            seq.anchor:hybrid.infer_hybrid(seq.visible,frozen)
            for seq in group
        }
        scores[split] = hybrid._aggregate_scores(group,trace,source.reference)
        switching[split] = hybrid._switching_scores(group,trace)
        results[split] = _score(group,trace)
        saved_halfturn = old["results"][split]
        counts = results[split]["label_only_pair_counts"]
        if counts["switch"] != saved_halfturn["total_labelled_switches_with_prior_step"]:
            raise RuntimeError("Previous halfturn switch count mismatch")
        if results[split]["original_frozen_geometry_abstention_reasons"]["switch"] != saved_halfturn["original_reason_counts"]:
            raise RuntimeError("Previous halfturn frozen Geometry reason mismatch")
        if counts["switch"] != switching[split]["event_switches_observed"] or (
            counts["same"] != switching[split]["same_oracle_event_pairs"]
        ):
            raise RuntimeError("Frozen Hybrid pair counts differ")
    hybrid_sha = switch._check_saved_hybrid(
        root/hybrid.REPORT, config, source, initial, scores, switching)
    switch_sha = first._verify_prior_report(source,config,initial,scores,switching)
    required = {
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
    if (
        old.get("format") != halfturn.FORMAT
        or old.get("role") != "Train20 fit / chart-disjoint heldout ONLY"
        or old.get("config_sha256") != config.signature()
        or old.get("train_partition") != source.partition
        or old.get("source_sha256") != required
        or old.get("no_training") is not True
        or old.get("validation_evaluated") is not False
        or old.get("final_evaluated") is not False
        or old.get("motor_student_training_permitted") is not False
        or old.get("original_stage_a_gate_overridden") is not False
        or old.get("automatically_adopted") is not False
    ):
        raise RuntimeError("Frozen halfturn audit provenance changed")
    if p._protect_old_best(config)!=source.original_best_sha or p.sha256(old_path)!=old_sha:
        raise RuntimeError("Protected Stage A/Stage T artifact changed during evaluation")
    report = {
        "format":FORMAT,
        "role":"Train20 chart-disjoint fit/heldout read-only counterfactual",
        "hypothesis":"Correct 180-degree coordinate discontinuity for original motion guard",
        "prediction_inputs":"causal previous 3 frames of 263D HUD only",
        "guard":"original Geometry must report inconsistent_orbit_motion; corrected speed fixed [0.5,1.5], optional prior agreement <=1deg",
        "prior_agreement_degrees":PRIOR_AGREEMENT_DEG,
        "new_training":False,
        "no_training":True,
        "original_geometry_changed":False,
        "existing_hybrid_changed":False,
        "model_adoption":False,
        "motor_student_training_permitted":False,
        "validation_evaluated":False,
        "final_evaluated":False,
        "original_stage_a_gate_overridden":False,
        "config_sha256":config.signature(),
        "train_partition":source.partition,
        "source_sha256":{**required,"halfturn_angle_audit":old_sha},
        "results":results,
    }
    root.mkdir(parents=True,exist_ok=True)
    p._no_replace_json(path,report)
    return report
