from __future__ import annotations

"""Read-only, evaluator-labelled analysis of what visible HUD changes at note switches.

The 263D feature diffs and Hybrid predictions are computed without oracle data.
Only AFTER those predictions exist are Auto event IDs/timestamps/masks used to
stratify observed switch vs non-switch pairs and post-switch frames.
This is NOT a trained or deployed event detector. No new model is fit.
"""

from collections import Counter
import json
from math import atan2, degrees
from pathlib import Path

import torch

from . import prestige1 as p
from . import prestige1_autoplay_countdown as cd
from . import prestige1_autoplay_diagnostics as distance
from . import prestige1_autoplay_hybrid_audit as hybrid
from . import prestige1_autoplay_near_ablation as ablation
from . import prestige1_autoplay_near_countdown_audit as previous_countdown
from . import prestige1_autoplay_readout as stage_t
from . import prestige1_transfer as transfer

FORMAT = "prestige1-stage-t-hud-note-transition-diagnostic-train20-v1"
REPORT = "stage_timing_autoplay_hud_note_transition_train20.json"
CHANGE_EPS = 1e-6  # Fixed floor equality epsilon, same as geometry_one
OFFSETS = (0, 1, 2, 3, 4)  # fixed descriptive first 50ms; never tuned
EXAMPLE_COUNT = 5
FLAGS = (
    "motor_changed", "orbit_xy_changed", "behind_floor_changed",
    "current_floor_changed", "next_floor_changed", "other_floor_changed",
    "any_floor_changed", "current_position_changed",
    "current_angles_changed", "current_event_flags_changed",
    "current_visibility_changed", "next_angles_changed",
    "next_event_flags_changed", "tile_bpm_changed", "real_bpm_changed",
    "feedback_changed",
)


def _changed(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return any(abs(x-y) > CHANGE_EPS for x, y in zip(a, b, strict=True))


def hud_delta(previous: tuple[float, ...], current: tuple[float, ...]) -> dict:
    """HUD-only deterministic pair features. No oracle time or event identity."""
    if len(previous) != p.INPUT_DIM or len(current) != p.INPUT_DIM:
        raise ValueError("HUD delta needs two visible 263D frames")
    if not all(__import__("math").isfinite(x) for x in previous + current):
        raise ValueError("Nonfinite HUD delta input")
    off, n = cd.SLOT_START, cd.FLOOR_DIM
    floor_slots = cd.FLOORS.floor_slots
    def sl(v: tuple[float, ...], start: int, end: int) -> tuple[float, ...]:
        return v[start:end]
    def floor(v: tuple[float, ...], slot: int) -> tuple[float, ...]:
        return sl(v, off + slot*n, off + (slot+1)*n)
    pbehind, nbehind = floor(previous, cd.CURRENT_SLOT-1), floor(current, cd.CURRENT_SLOT-1)
    pcurrent, ncurrent = floor(previous, cd.CURRENT_SLOT), floor(current, cd.CURRENT_SLOT)
    pnext, nnext = floor(previous, cd.NEXT_SLOT), floor(current, cd.NEXT_SLOT)
    changed_slots = [
        i for i in range(floor_slots)
        if _changed(floor(previous, i), floor(current, i))
    ]
    before = atan2(previous[cd.MOTOR_FEATURES], previous[cd.MOTOR_FEATURES+1])
    after = atan2(current[cd.MOTOR_FEATURES], current[cd.MOTOR_FEATURES+1])
    return {
        "motor_changed": _changed(sl(previous, 0, cd.MOTOR_FEATURES),
                                  sl(current, 0, cd.MOTOR_FEATURES)),
        "orbit_xy_changed": _changed(sl(previous, cd.MOTOR_FEATURES, off),
                                    sl(current, cd.MOTOR_FEATURES, off)),
        "behind_floor_changed": _changed(pbehind, nbehind),
        "current_floor_changed": _changed(pcurrent, ncurrent),
        "next_floor_changed": _changed(pnext, nnext),
        "other_floor_changed": any(
            slot not in (cd.CURRENT_SLOT-1, cd.CURRENT_SLOT, cd.NEXT_SLOT)
            for slot in changed_slots),
        "any_floor_changed": bool(changed_slots),
        "current_position_changed": _changed(pcurrent[0:2], ncurrent[0:2]),
        "current_angles_changed": _changed(pcurrent[2:6], ncurrent[2:6]),
        "current_event_flags_changed": _changed(pcurrent[6:14], ncurrent[6:14]),
        "current_visibility_changed": _changed(pcurrent[14:15], ncurrent[14:15]),
        "next_angles_changed": _changed(pnext[2:6], nnext[2:6]),
        "next_event_flags_changed": _changed(pnext[6:14], nnext[6:14]),
        "tile_bpm_changed": _changed(previous[cd.HUD_OFFSET:cd.HUD_OFFSET+1],
                                     current[cd.HUD_OFFSET:cd.HUD_OFFSET+1]),
        "real_bpm_changed": _changed(previous[cd.HUD_OFFSET+1:cd.HUD_OFFSET+2],
                                     current[cd.HUD_OFFSET+1:cd.HUD_OFFSET+2]),
        "feedback_changed": _changed(previous[cd.HUD_OFFSET+2:],
                                     current[cd.HUD_OFFSET+2:]),
        "changed_floor_slots": tuple(changed_slots),
        "orbit_delta_degrees": degrees(cd._wrap(after-before)),
        "current_floor_linf": max(abs(a-b) for a, b in zip(pcurrent, ncurrent)),
        "next_floor_linf": max(abs(a-b) for a, b in zip(pnext, nnext)),
    }


def _pair_summary(group: list[stage_t.TimingSequence]) -> dict:
    positives = Counter()
    negatives = Counter()
    switch_samples: list[dict] = []
    same_samples: list[dict] = []
    anchors = []
    for seq in group:
        features = [tuple(map(float, row)) for row in seq.visible.detach().cpu().tolist()]
        switches = nonswitches = 0
        for index in range(1, len(features)):
            if not (bool(seq.mask[index-1]) and bool(seq.mask[index])):
                continue
            a, b = seq.event_indices[index-1], seq.event_indices[index]
            if a < 0 or b < 0:
                continue
            props = hud_delta(features[index-1], features[index])
            if a == b:
                nonswitches += 1
                same_samples.append(props)
                for flag in FLAGS:
                    negatives[flag] += int(props[flag])
            else:
                switches += 1
                switch_samples.append(props)
                for flag in FLAGS:
                    positives[flag] += int(props[flag])
        anchors.append({
            "anchor": seq.anchor, "chart_sha256": seq.chart_sha256,
            "true_event_switch_pairs": switches, "same_event_pairs": nonswitches,
        })
    positive_n, negative_n = len(switch_samples), len(same_samples)
    if not positive_n or not negative_n:
        raise RuntimeError("Switch HUD audit requires switch and non-switch pairs")
    cues = {}
    for flag in FLAGS:
        tp = positives[flag]
        fp = negatives[flag]
        cues[flag] = {
            "switch_cue_true": tp, "same_event_cue_true": fp,
            "true_switch_count": positive_n, "true_same_event_count": negative_n,
            "switch_change_rate": tp/positive_n,
            "same_event_change_rate": fp/negative_n,
            "descriptive_precision": tp/(tp+fp) if tp+fp else None,
            "descriptive_recall": tp/positive_n,
            "descriptive_specificity": (negative_n-fp)/negative_n,
        }
    def number_summary(samples: list[dict], field: str) -> dict:
        vals = [float(s[field]) for s in samples]
        vals.sort()
        return {
            "count": len(vals),
            "mean": sum(vals)/len(vals),
            "median": (vals[(len(vals)-1)//2]+vals[len(vals)//2])/2,
            "p95_abs": sorted(abs(v) for v in vals)[max(0, (95*len(vals)+99)//100-1)],
        }
    return {
        "pair_counts": {"true_switch": positive_n, "same_event": negative_n},
        "cues": cues,
        "orbit_angle_step_degrees": {
            "true_switch": number_summary(switch_samples, "orbit_delta_degrees"),
            "same_event": number_summary(same_samples, "orbit_delta_degrees"),
        },
        "current_floor_linf": {
            "true_switch": number_summary(switch_samples, "current_floor_linf"),
            "same_event": number_summary(same_samples, "current_floor_linf"),
        },
        "per_anchor": anchors,
        "warning": (
            "All cue rates and descriptive precision use oracle event identity ONLY "
            "as an AFTER-inference label; this is NOT an independently validated "
            "HUD-only switch detector. All flags use fixed 1e-6 change epsilon."
        ),
    }


def _post_switch(group: list[stage_t.TimingSequence],
                 traces: dict[int, hybrid.HybridTrace]) -> dict:
    offset_errors = {
        i: {"hybrid": [], "near_only_gru": [], "geometry": []}
        for i in OFFSETS
    }
    offset_counts = {i: Counter() for i in OFFSETS}
    first_geom_delays: list[int] = []
    first_geom_errors: list[float] = []
    first_frame_examples: list[dict] = []
    terminal_reasons = Counter()
    examined = 0
    for seq in group:
        t = traces[seq.anchor]
        frames = [tuple(map(float, row)) for row in seq.visible.detach().cpu().tolist()]
        n = len(seq.event_indices)
        for k in range(1, n):
            if not (bool(seq.mask[k-1]) and bool(seq.mask[k])):
                continue
            old_ev, new_ev = seq.event_indices[k-1:k+1]
            if old_ev < 0 or new_ev < 0 or old_ev == new_ev:
                continue
            examined += 1
            start = k
            # Oracle event grouping is evaluator-only: stop measuring the
            # recovery of THIS ground-truth target at the next target change.
            end = k
            while end < n and bool(seq.mask[end]) and seq.event_indices[end] == new_ev:
                end += 1
            first_geo = next((j for j in range(start, end) if t.mode[j] == "geometry"), None)
            if first_geo is None:
                if end == n:
                    terminal_reasons["sequence_end_before_geometry"] += 1
                elif not bool(seq.mask[end]):
                    terminal_reasons["observability_end_before_geometry"] += 1
                else:
                    terminal_reasons["next_event_before_geometry"] += 1
            else:
                delay = first_geo-start
                first_geom_delays.append(delay)
                first_geom_errors.append(
                    (t.chosen_s[first_geo]-float(seq.remaining_s[first_geo]))*1000.)
            for offset in OFFSETS:
                j = k+offset
                if j >= end:
                    continue
                oracle_s = float(seq.remaining_s[j])
                offset_counts[offset][t.mode[j]] += 1
                offset_errors[offset]["hybrid"].append((t.chosen_s[j]-oracle_s)*1000.)
                offset_errors[offset]["near_only_gru"].append((t.fallback_s[j]-oracle_s)*1000.)
                if t.geometry_s[j] is not None:
                    offset_errors[offset]["geometry"].append(
                        (t.geometry_s[j]-oracle_s)*1000.)
            changed = hud_delta(frames[k-1], frames[k])
            first_frame_examples.append({
                "anchor": seq.anchor, "chart_sha256": seq.chart_sha256,
                "frame_index": k, "oracle_switch_id_before": old_ev,
                "oracle_switch_id_after": new_ev,
                "first_mode": t.mode[k],
                "first_abstain_reason": t.abstain_reasons[k],
                "true_new_remaining_ms": float(seq.remaining_s[k])*1000.,
                "first_hybrid_error_ms": (t.chosen_s[k]-float(seq.remaining_s[k]))*1000.,
                "first_gru_error_ms": (t.fallback_s[k]-float(seq.remaining_s[k]))*1000.,
                "first_geometry_delay_ms": (
                    (first_geo-k)*p.CONTROL_DT*1000 if first_geo is not None else None),
                "observed_floor_change": {
                    flag: changed[flag] for flag in (
                        "current_floor_changed", "next_floor_changed",
                        "current_angles_changed", "current_position_changed",
                        "current_event_flags_changed", "tile_bpm_changed",
                        "feedback_changed",
                    )
                },
            })
    first_frame_examples.sort(
        key=lambda item: abs(item["first_hybrid_error_ms"]), reverse=True)
    return {
        "scored_true_event_switches": examined,
        "first_eligible_geometry_same_event": {
            "reached": len(first_geom_delays),
            "not_reached": examined-len(first_geom_delays),
            "reason_if_not_reached": dict(sorted(terminal_reasons.items())),
            "delay_ms": cd._summary([d*p.CONTROL_DT*1000. for d in first_geom_delays]),
            "delay_frames_distribution": dict(sorted(Counter(first_geom_delays).items())),
            "absolute_timing_error_on_first_geometry": cd._summary(first_geom_errors),
        },
        "offset_from_switch": {
            str(i): {
                "frames_with_same_oracle_event_and_observable": sum(offset_counts[i].values()),
                "modes": dict(offset_counts[i]),
                "errors": {
                    key: cd._summary(value) for key, value in offset_errors[i].items()
                },
            }
            for i in OFFSETS
        },
        "largest_first_frame_errors": first_frame_examples[:EXAMPLE_COUNT],
        "warning": (
            "The first Geometry-eligible delay is evaluated within the same ORACLE "
            "event, posthoc, never fed to inference. It is NOT a stateful "
            "tracker's resynchronization latency. Offset buckets may be censored "
            "by the next event, observability, or segment end."
        ),
    }


def _check_saved_hybrid(
    path: Path, config: p.PrestigeConfig,
    source: distance.FrozenTimingInputs, snapshots: dict,
    scores: dict, switching: dict,
) -> str:
    if not path.is_file():
        raise FileNotFoundError("Run the frozen stateless Hybrid switching audit first")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if (
        saved.get("format") != hybrid.FORMAT
        or saved.get("role") != "Train20 fit / chart-disjoint heldout only"
        or saved.get("no_training") is not True
        or saved.get("validation_evaluated") is not False
        or saved.get("final_evaluated") is not False
        or saved.get("original_stage_a_gate_overridden") is not False
        or saved.get("motor_student_training_permitted") is not False
        or saved.get("automatically_adopted") is not False
        or saved.get("true_event_tracker_implemented") is not False
        or saved.get("config_sha256") != config.signature()
        or saved.get("train_partition") != source.partition
        or saved.get("source_sha256") != snapshots
    ):
        raise RuntimeError("Prior stateless Hybrid provenance changed")
    for split in ("train_fit", "train_heldout"):
        before = saved["scores"][split]
        after = scores[split]
        for key in ("visible_frames", "near_frames", "geometry_near_frames"):
            if before[key] != after[key]:
                raise RuntimeError(f"Prior Hybrid coverage drift: {split}/{key}")
        for name in ("near_only_gru", "geometry", "hybrid"):
            for bucket in ("all_visible", "near"):
                old = before["matched_baselines"][name][bucket]
                new = after["matched_baselines"][name][bucket]
                if old["count"] != new["count"] or abs(old["mae_ms"]-new["mae_ms"]) > .05:
                    raise RuntimeError(f"Prior Hybrid MAE drift: {split}/{name}/{bucket}")
        for key in ("event_switches_observed", "same_oracle_event_pairs"):
            if saved["switching"][split][key] != switching[split][key]:
                raise RuntimeError(f"Prior Hybrid event alignment drift: {split}/{key}")
    return p.sha256(path)


def run_switch_hud_audit(config: p.PrestigeConfig) -> dict:
    root = transfer.output_dir(config)
    report_path = root / REPORT
    source = distance.prepare(config, output_paths=(report_path,))
    model, old_near_report, old_fingerprints = previous_countdown._restore_near_only(config, source)
    previous_sha = hybrid._validate_prior(
        root, source, old_near_report, old_fingerprints, model)
    snapshots = {
        **old_fingerprints,
        "near_only_countdown": previous_sha,
    }
    score_replays, switch_replays, results = {}, {}, {}
    for split, group in (("train_fit", source.fit), ("train_heldout", source.heldout)):
        traces = {seq.anchor: hybrid.infer_hybrid(seq.visible, model) for seq in group}
        score_replays[split] = hybrid._aggregate_scores(group, traces, source.reference)
        switch_replays[split] = hybrid._switching_scores(group, traces)
        pair = _pair_summary(group)
        post = _post_switch(group, traces)
        if pair["pair_counts"]["true_switch"] != switch_replays[split]["event_switches_observed"]:
            raise RuntimeError("Switch pair counts drifted from frozen Hybrid")
        if post["scored_true_event_switches"] != pair["pair_counts"]["true_switch"]:
            raise RuntimeError("Switch post-window count drifted")
        results[split] = {"visible_cue_differences": pair, "post_switch": post}
    hybrid_sha = _check_saved_hybrid(
        root / hybrid.REPORT, config, source,
        snapshots, score_replays, switch_replays,
    )
    protected_now = {
        **source.source_sha256,
        "near_only_report": p.sha256(root / ablation.REPORT),
        "near_only_checkpoint": p.sha256(root / ablation.CHECKPOINT),
        "distance_audit": p.sha256(root / distance.REPORT),
        "old_countdown": p.sha256(root / cd.REPORT),
        "near_only_countdown": p.sha256(root / previous_countdown.REPORT),
    }
    if (
        protected_now != snapshots
        or p.sha256(root / hybrid.REPORT) != hybrid_sha
        or p._protect_old_best(config) != source.original_best_sha
    ):
        raise RuntimeError("Frozen Timing artifacts changed during switch HUD audit")
    report = {
        "format": FORMAT, "role": "Train20 chart-disjoint fit / heldout only",
        "purpose": "Observe visible HUD transitions before/after true next-note changes",
        "model_inputs": "Frozen near-only GRU and previous/current visible 263D HUD only",
        "labels": "Auto event IDs/timestamps evaluator-only AFTER all predictions",
        "switch_detection_model_trained": False,
        "new_hybrid_model_trained": False,
        "no_training": True,
        "validation_evaluated": False, "final_evaluated": False,
        "original_stage_a_gate_overridden": False,
        "motor_student_training_permitted": False, "automatically_adopted": False,
        "config_sha256": config.signature(),
        "train_partition": source.partition,
        "source_sha256": {**snapshots, "stateless_hybrid": hybrid_sha},
        "hud_change_epsilon": CHANGE_EPS,
        "post_switch_offsets_10ms": OFFSETS,
        "results": results,
        "limitations": [
            "This is a posthoc ORACLE-labelled diagnosis, NOT a deployed HUD-only event detector.",
            "Feature groups are descriptive and threshold 1e-6 is fixed, never fitted to heldout.",
            "Only the stored expert-driven Train20 HUD streams are evaluated.",
            "First available Geometry does not imply correct event-tracking recovery.",
            "No Validation, FINAL, retraining, Motor or Fingering training or Stage A override.",
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    p._no_replace_json(report_path, report)
    return report
