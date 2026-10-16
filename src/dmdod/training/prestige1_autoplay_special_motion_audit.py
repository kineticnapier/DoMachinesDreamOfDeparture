from __future__ import annotations

"""Immutable Stage T analysis of remaining HUD-only Geometry abstentions.

Two independently defined populations, with no new estimator:
  A) existing *carried_previous_step* candidate has special_event status
     at an oracle-labelled target switch;
  B) the original frozen geometry_one() reports inconsistent_orbit_motion
     at an oracle-labelled target switch.

Every HUD feature, original abstention, and frozen counterfactual status is
computed for the entire causal 263D stream FIRST. Oracle event IDs, masks and
exact times only segment/score the resulting diagnostics AFTER inference.
"""

from collections import Counter
import json
from math import atan2, ceil, isfinite, pi
from pathlib import Path
import statistics

from dmdod.features.hud_features import HUD_BPM_REFERENCE
from dmdod.features.real_chart import REAL_CHART_EVENT_MARKERS

from . import prestige1 as p
from . import prestige1_autoplay_countdown as cd
from . import prestige1_autoplay_diagnostics as distance
from . import prestige1_autoplay_first_frame_geometry_audit as previous
from . import prestige1_autoplay_hybrid_audit as hybrid
from . import prestige1_autoplay_near_ablation as ablation
from . import prestige1_autoplay_near_countdown_audit as near_countdown
from . import prestige1_autoplay_switch_hud_audit as switch_hud
from . import prestige1_autoplay_readout as stage_t
from . import prestige1_transfer as transfer

FORMAT = "prestige1-stage-t-special-and-motion-failure-classification-train20-v1"
REPORT = "stage_timing_autoplay_special_motion_train20.json"
MARKERS = tuple(REAL_CHART_EVENT_MARKERS)
SPEED_BAND = (0.5, 1.5)  # EXACT existing frozen geometry_one guard
EXAMPLE_COUNT = 8


def _marker_snapshot(current: tuple[float, ...]) -> dict:
    here = cd._visible_slot(current, cd.CURRENT_SLOT)
    ahead = cd._visible_slot(current, cd.NEXT_SLOT)
    def flags(tile: tuple[float, ...]) -> dict:
        return {
            marker: tile[cd.SKIP_MARKER_FROM + i] > .5
            for i, marker in enumerate(MARKERS)
        }
    current_flags, next_flags = flags(here), flags(ahead)
    return {
        "current": current_flags,
        "next": next_flags,
        "current_midspin": here[6] > .5,
        "next_midspin": ahead[6] > .5,
        "current_multiplanet": here[7] > .5,
        "next_multiplanet": ahead[7] > .5,
        "current_visible": here[-1] > .5,
        "next_visible": ahead[-1] > .5,
    }


def hud_motion_snapshot(
    visible: tuple[tuple[float, ...], ...], k: int,
) -> dict:
    """Causal HUD-only physics diagnostics; NO evaluator labels or oracle."""
    if not 1 <= k < len(visible) or any(len(row) != p.INPUT_DIM for row in visible):
        raise ValueError("Motion requires index>=1 and 263D HUD")
    frames = visible[max(0, k-2):k+1]
    if not all(isfinite(v) for row in frames for v in row):
        raise ValueError("Nonfinite HUD")
    prev, cur = visible[k-1], visible[k]
    old = visible[k-2] if k >= 2 else None
    cross = cd._wrap(atan2(cur[cd.MOTOR_FEATURES],cur[cd.MOTOR_FEATURES+1]) -
                      atan2(prev[cd.MOTOR_FEATURES],prev[cd.MOTOR_FEATURES+1]))
    prior = (
        cd._wrap(atan2(prev[cd.MOTOR_FEATURES],prev[cd.MOTOR_FEATURES+1]) -
                 atan2(old[cd.MOTOR_FEATURES],old[cd.MOTOR_FEATURES+1]))
        if old is not None else None
    )
    bpm_current = HUD_BPM_REFERENCE * (2. ** cur[cd.HUD_OFFSET])
    bpm_previous = HUD_BPM_REFERENCE * (2. ** prev[cd.HUD_OFFSET])
    real_current = HUD_BPM_REFERENCE * (2. ** cur[cd.HUD_OFFSET+1])
    real_previous = HUD_BPM_REFERENCE * (2. ** prev[cd.HUD_OFFSET+1])
    expected_current = pi*bpm_current/60. * p.CONTROL_DT
    expected_previous = pi*bpm_previous/60. * p.CONTROL_DT
    cross_ratio = abs(cross)/expected_current
    prior_ratio_current = abs(prior)/expected_current if prior is not None else None
    prior_ratio_previous = abs(prior)/expected_previous if prior is not None else None
    if not all(isfinite(v) and v > 0. for v in (
        bpm_current,bpm_previous,real_current,real_previous,
        expected_current,expected_previous,
    )):
        raise RuntimeError("Invalid encoded HUD BPM")
    def speed_class(ratio: float | None) -> str:
        if ratio is None:
            return "no_prior_frame"
        if ratio < SPEED_BAND[0]:
            return "below_half_expected"
        if ratio > SPEED_BAND[1]:
            return "above_one_and_half_expected"
        return "within_original_guard"
    return {
        "marker_flags": _marker_snapshot(cur),
        "crossing_step_deg": cross*180./pi,
        "prior_step_deg": prior*180./pi if prior is not None else None,
        "crossing_minus_prior_deg": cd._wrap(cross-prior)*180./pi
        if prior is not None else None,
        "direction_flip": prior*cross < 0 if prior is not None else None,
        "tile_bpm": bpm_current, "previous_tile_bpm": bpm_previous,
        "real_bpm": real_current, "previous_real_bpm": real_previous,
        "tile_bpm_changed": abs(cur[cd.HUD_OFFSET]-prev[cd.HUD_OFFSET]) > 1e-6,
        "real_bpm_changed": abs(cur[cd.HUD_OFFSET+1]-prev[cd.HUD_OFFSET+1]) > 1e-6,
        "crossing_speed_ratio": cross_ratio,
        "prior_to_current_speed_ratio": prior_ratio_current,
        "prior_to_previous_speed_ratio": prior_ratio_previous,
        "crossing_speed_class": speed_class(cross_ratio),
        "prior_current_speed_class": speed_class(prior_ratio_current),
        "prior_previous_speed_class": speed_class(prior_ratio_previous),
        "current_floor_changed": any(
            abs(a-b)>1e-6 for a,b in zip(
                cd._visible_slot(prev,cd.CURRENT_SLOT),
                cd._visible_slot(cur,cd.CURRENT_SLOT)
            )
        ),
    }


def _number_summary(values: list[float], *, unit: str) -> dict:
    """Dimensionally honest summary: angles are degrees; speeds are ratios."""
    if not values:
        return {
            "unit":unit, "count":0, "mean":None, "mean_abs":None,
            "median":None, "p95_abs":None, "min":None, "max":None,
        }
    ascending = sorted(values)
    absolute = sorted(abs(v) for v in values)
    return {
        "unit": unit,
        "count": len(values),
        "mean": statistics.mean(values),
        "mean_abs": statistics.mean(absolute),
        "median": statistics.median(ascending),
        "p95_abs": absolute[ceil(.95*len(values))-1],
        "min": ascending[0],
        "max": ascending[-1],
    }


def _marker_report(rows: list[dict]) -> dict:
    by_slot = {"current": Counter(), "next": Counter()}
    both_slots = Counter()
    arrangements = Counter()
    flags_present = Counter()
    examples: list[dict] = []
    for row in rows:
        flags = row["snapshot"]["marker_flags"]
        labels = []
        for marker in MARKERS:
            on_current = flags["current"][marker]
            on_next = flags["next"][marker]
            if on_current:
                by_slot["current"][marker] += 1
                labels.append("current:"+marker)
            if on_next:
                by_slot["next"][marker] += 1
                labels.append("next:"+marker)
            if on_current and on_next:
                both_slots[marker] += 1
        arrangement = ",".join(labels) if labels else "(no markers)"
        arrangements[arrangement] += 1
        for indicator in (
            "current_midspin","next_midspin",
            "current_multiplanet","next_multiplanet",
        ):
            flags_present[indicator] += int(flags[indicator])
        examples.append({
            "anchor": row["anchor"], "frame": row["frame"],
            "original_status": row["original_status"],
            "carried_status": row["carried_status"],
            "marker_configuration": arrangement,
        })
    return {
        "count": len(rows),
        "marker_occurrences_by_slot": {
            slot: {marker: counter[marker] for marker in MARKERS}
            for slot,counter in by_slot.items()
        },
        "same_marker_on_both_slots": {
            marker: both_slots[marker] for marker in MARKERS
        },
        "marker_patterns_exclusive": dict(sorted(arrangements.items())),
        "structural_flags": dict(sorted(flags_present.items())),
        "examples": examples[:EXAMPLE_COUNT],
        "warning": "Marker counts overlap; marker_patterns_exclusive partitions events.",
    }


def _motion_report(rows: list[dict]) -> dict:
    fields = (
        ("crossing_speed_ratio","ratio"),
        ("prior_to_current_speed_ratio","ratio"),
        ("prior_to_previous_speed_ratio","ratio"),
        ("crossing_step_deg","deg"),
        ("prior_step_deg","deg"),
        ("crossing_minus_prior_deg","deg"),
    )
    classes = (
        "crossing_speed_class","prior_current_speed_class",
        "prior_previous_speed_class",
    )
    by_class = {
        key: dict(sorted(Counter(row["snapshot"][key] for row in rows).items()))
        for key in classes
    }
    numerical = {
        name: _number_summary(
            [float(row["snapshot"][name]) for row in rows
             if row["snapshot"][name] is not None],
            unit=unit,
        )
        for name,unit in fields
    }
    flips = sum(row["snapshot"]["direction_flip"] is True for row in rows)
    tile_changed = sum(row["snapshot"]["tile_bpm_changed"] for row in rows)
    real_changed = sum(row["snapshot"]["real_bpm_changed"] for row in rows)
    speed_patterns = Counter(
        (row["snapshot"]["crossing_speed_class"],
         row["snapshot"]["prior_current_speed_class"])
        for row in rows
    )
    examples = sorted(
        rows,
        key=lambda row: abs(
            row["snapshot"]["crossing_speed_ratio"]-1.),
        reverse=True,
    )[:EXAMPLE_COUNT]
    return {
        "count": len(rows),
        "guard_speed_band_inclusive": SPEED_BAND,
        "speed_classes": by_class,
        "crossing_vs_prior_speed_classes": [
            {"crossing": a, "prior": b, "count": n}
            for (a,b),n in sorted(speed_patterns.items())
        ],
        "quantities": numerical,
        "direction_flips": flips,
        "tile_bpm_changed_count": tile_changed,
        "real_bpm_changed_count": real_changed,
        "joint_markers": _marker_report(rows),
        "most_inconsistent_crossing_samples": [
            {
                "anchor": row["anchor"], "frame": row["frame"],
                "original_status": row["original_status"],
                "carried_status": row["carried_status"],
                "crossing_ratio": row["snapshot"]["crossing_speed_ratio"],
                "prior_current_ratio": row["snapshot"]["prior_to_current_speed_ratio"],
                "tile_bpm": row["snapshot"]["tile_bpm"],
                "previous_tile_bpm": row["snapshot"]["previous_tile_bpm"],
                "direction_flip": row["snapshot"]["direction_flip"],
                "markers": {
                    slot: [
                        name for name,seen in row["snapshot"]["marker_flags"][slot].items()
                        if seen
                    ]
                    for slot in ("current","next")
                },
            }
            for row in examples
        ],
    }


def _describe_split(
    group: list[stage_t.TimingSequence],
) -> dict:
    # Unconditional precomputation over entire HUD stream is the inference/
    # information-boundary check. The evaluator does not control this pass.
    raw_samples = []
    anchor_counts = []
    for seq in group:
        hud = tuple(tuple(float(x) for x in row)
                    for row in seq.visible.detach().cpu().tolist())
        candidates, original = previous.first_frame_candidates(hud)
        observations = [
            hud_motion_snapshot(hud,k) if k else None
            for k in range(len(hud))
        ]
        actual_switches = 0
        for k in range(1,len(hud)):
            # BELOW this boundary, oracle IDs and masks are evaluation-only.
            if not (bool(seq.mask[k-1]) and bool(seq.mask[k])):
                continue
            before,after = seq.event_indices[k-1:k+1]
            if before < 0 or after < 0 or before == after:
                continue
            actual_switches += 1
            snap = observations[k]
            raw_samples.append({
                "anchor": seq.anchor,"frame": k,
                "original_status": original[k],
                "carried_status": candidates["carried_previous_step"][k].status,
                "crossing_status": candidates["current_crossing_step"][k].status,
                "snapshot": snap,
            })
        anchor_counts.append({
            "anchor":seq.anchor,"chart_sha256":seq.chart_sha256,
            "true_note_switches":actual_switches,
        })
    if not raw_samples:
        raise RuntimeError("No oracle-labelled note switches for motion audit")
    original_counts = Counter(row["original_status"] for row in raw_samples)
    carried_counts = Counter(row["carried_status"] for row in raw_samples)
    cross_counts = Counter(row["crossing_status"] for row in raw_samples)
    cross_table = Counter(
        (row["original_status"],row["carried_status"]) for row in raw_samples
    )
    special = [
        row for row in raw_samples
        if row["carried_status"] == "special_event"
    ]
    inconsistent = [
        row for row in raw_samples
        if row["original_status"] == "inconsistent_orbit_motion"
    ]
    floor_changed_inconsistent_after_guard = [
        row for row in raw_samples
        if row["original_status"] == "floor_changed"
        and row["carried_status"] == "inconsistent_orbit_motion"
    ]
    if len(special) != carried_counts["special_event"] or (
        len(inconsistent) != original_counts["inconsistent_orbit_motion"]
    ):
        raise RuntimeError("Disjoint population accounting failed")
    return {
        "true_note_switches": len(raw_samples),
        "original_geometry_first_rejection": dict(sorted(original_counts.items())),
        "counterfactual_carried_first_rejection": dict(sorted(carried_counts.items())),
        "counterfactual_crossing_first_rejection": dict(sorted(cross_counts.items())),
        "original_vs_carried_cross_tab": [
            {"original_status":a,"carried_status":b,"count":count}
            for (a,b),count in sorted(cross_table.items())
        ],
        "carried_special_event": {
            "marker_breakdown": _marker_report(special),
            "motion_context": _motion_report(special),
        },
        "original_inconsistent_orbit_motion": _motion_report(inconsistent),
        "floor_changed_then_carried_motion_inconsistent": _motion_report(
            floor_changed_inconsistent_after_guard
        ),
        "per_anchor":anchor_counts,
        "interpretation_note": (
            "These are two independent overlapping definition domains: "
            "carried_special_event uses carried-candidate status AFTER original "
            "floor_changed; original_inconsistent_orbit_motion uses ORIGINAL first "
            "guard. Do not add their counts to estimate unique failures. "
            "This diagnostic NEVER turns off a Geometry guard."
        ),
    }


def _verify_frozen_first_frame(
    source: distance.FrozenTimingInputs,
    config: p.PrestigeConfig,
    initial_sources: dict,
    scores: dict,
    switches: dict,
    traces: dict,
) -> str:
    path = source.root / previous.REPORT
    if not path.is_file():
        raise FileNotFoundError("Run first-frame Geometry audit before marker breakdown")
    saved = json.loads(path.read_text(encoding="utf-8"))
    required_sources = {
        **initial_sources,
        "stateless_hybrid": p.sha256(source.root / hybrid.REPORT),
        "hud_switch_audit": p.sha256(source.root / switch_hud.REPORT),
    }
    if (
        saved.get("format") != previous.FORMAT
        or saved.get("no_training") is not True
        or saved.get("validation_evaluated") is not False
        or saved.get("final_evaluated") is not False
        or saved.get("original_stage_a_gate_overridden") is not False
        or saved.get("motor_student_training_permitted") is not False
        or saved.get("automatically_adopted") is not False
        or saved.get("config_sha256") != config.signature()
        or saved.get("source_sha256") != required_sources
        or saved.get("train_partition") != source.partition
    ):
        raise RuntimeError("Frozen first-frame Geometry provenance mismatch")
    for split,group in (
        ("train_fit",source.fit),("train_heldout",source.heldout)
    ):
        recorded = saved["results"][split]
        actual = previous._score_split(group, traces[split])
        for field in (
            "oracle_true_switches", "original_floor_changed_at_switch",
            "original_geometry_abstention_at_switch",
        ):
            if recorded[field] != actual[field]:
                raise RuntimeError(f"Frozen first-frame replay drifted: {split}/{field}")
        for candidate in previous.CANDIDATES:
            old = recorded["candidates_on_same_oracle_switch_frames"][candidate]
            new = actual["candidates_on_same_oracle_switch_frames"][candidate]
            if old["eligible_switches"] != new["eligible_switches"]:
                raise RuntimeError("Frozen first-frame candidate eligibility changed")
            for score_key in ("absolute_time_error", "near_time_error",
                              "same_frame_frozen_gru"):
                before, after = old[score_key], new[score_key]
                if before["count"] != after["count"]:
                    raise RuntimeError(f"Frozen first-frame sample changed: {candidate}")
                if before["mae_ms"] is None:
                    if after["mae_ms"] is not None:
                        raise RuntimeError("Frozen empty first-frame metric changed")
                elif after["mae_ms"] is None or abs(
                    before["mae_ms"]-after["mae_ms"]
                ) > .05:
                    raise RuntimeError(f"Frozen first-frame MAE changed: {candidate}")
        if actual["oracle_true_switches"] != switches[split]["event_switches_observed"]:
            raise RuntimeError("First-frame switch count drifted")
        if scores[split]["near_frames"] <= 0:
            raise RuntimeError("Frozen Hybrid near-event coverage missing")
    return p.sha256(path)


def run_special_motion_audit(config: p.PrestigeConfig) -> dict:
    root = transfer.output_dir(config)
    report_path = root / REPORT
    source = distance.prepare(config, output_paths=(report_path,))
    frozen, near_report, fingerprints = near_countdown._restore_near_only(
        config,source)
    previous_sha = hybrid._validate_prior(
        root,source,near_report,fingerprints,frozen)
    inputs = {**fingerprints,"near_only_countdown":previous_sha}
    scores,switches,results,traces_by_split = {},{},{},{}
    for split, group in (
        ("train_fit",source.fit),("train_heldout",source.heldout)
    ):
        traces = {
            seq.anchor:hybrid.infer_hybrid(seq.visible,frozen)
            for seq in group
        }
        traces_by_split[split] = traces
        scores[split] = hybrid._aggregate_scores(group,traces,source.reference)
        switches[split] = hybrid._switching_scores(group,traces)
        results[split] = _describe_split(group)
        if results[split]["true_note_switches"] != switches[split]["event_switches_observed"]:
            raise RuntimeError("HUD special/motion switch sample counts drifted")
    hybrid_sha = switch_hud._check_saved_hybrid(
        root / hybrid.REPORT,config,source,inputs,scores,switches)
    switch_hud_sha = previous._verify_prior_report(
        source,config,inputs,scores,switches)
    first_frame_sha = _verify_frozen_first_frame(
        source,config,inputs,scores,switches,traces_by_split)
    protected = {
        **source.source_sha256,
        "near_only_report":p.sha256(root/ablation.REPORT),
        "near_only_checkpoint":p.sha256(root/ablation.CHECKPOINT),
        "distance_audit":p.sha256(root/distance.REPORT),
        "old_countdown":p.sha256(root/cd.REPORT),
        "near_only_countdown":p.sha256(root/near_countdown.REPORT),
        "stateless_hybrid":p.sha256(root/hybrid.REPORT),
        "hud_switch_audit":p.sha256(root/switch_hud.REPORT),
        "first_frame_geometry":p.sha256(root/previous.REPORT),
    }
    if (
        protected != {
            **inputs,
            "stateless_hybrid":hybrid_sha,
            "hud_switch_audit":switch_hud_sha,
            "first_frame_geometry":first_frame_sha,
        }
        or p._protect_old_best(config) != source.original_best_sha
    ):
        raise RuntimeError("Frozen Timing artifacts changed during special/motion audit")
    report = {
        "format":FORMAT,
        "role":"Train20 chart-disjoint fit / heldout only",
        "purpose":"Separate HUD special marker flags and crossing motion consistency at true note switches",
        "hud_observations_first":True,
        "oracle_ids_and_timestamps_evaluator_only":True,
        "no_training":True,
        "validation_evaluated":False,
        "final_evaluated":False,
        "original_stage_a_gate_overridden":False,
        "motor_student_training_permitted":False,
        "automatically_adopted":False,
        "new_geometry_guards_activated":False,
        "config_sha256":config.signature(),
        "train_partition":source.partition,
        "source_sha256":protected,
        "original_speed_guard_band":SPEED_BAND,
        "marker_names":MARKERS,
        "results":results,
        "limitations":[
            "Posthoc oracle event IDs define scored switch frames ONLY; no learned event detector",
            "Marker flags may overlap or coexist; grouped counts are not additive",
            "A marker with good descriptive scores is NOT safe to ignore for Geometry",
            "Speed ratios are inferred from visible orbit xy and HUD Tile BPM, NOT engine internals",
            "Prior carried-step Geometry is unchanged; no new counterfactual bypass is deployed",
            "Previously inspected Train-heldout data are research diagnostics, not clean final holdout",
            "Train20 only; Stage A FAIL; no Motor, new training, Validation or FINAL",
        ],
    }
    root.mkdir(parents=True,exist_ok=True)
    p._no_replace_json(report_path,report)
    return report
