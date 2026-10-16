from __future__ import annotations

"""Train20-only audit: scheduled event vs physically unresolved target.

Read-only, Expert-controlled; this is NOT a new training recipe. The legacy
Stage A gate and the Stage C v2 checkpoints are not changed. Validation/FINAL
are never captured or evaluated.
"""

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
import json
import math

from dmdod.envs.n_key import (
    DiagnosticHudNKeyRealChartMotorEnv,
    encode_n_key_hud_real_chart_observation,
)
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG
from dmdod.motor.capacity import CenterFirstNKeyTeacher
from . import prestige1 as p
from . import prestige1_transfer as transfer
from .n_key import _teacher_action
from .real_chart import build_anchor_segments, compile_role

FORMAT = "prestige1-c-v22-train-only-target-alignment-audit"
REPORT = "stage_c_v22_train_alignment.json"
SAMPLE_LIMIT = 8
EPS_S = 1e-9


@dataclass(frozen=True)
class TargetAlignment:
    """Evaluator-only target comparison at the beginning of a control frame."""

    kind: str
    legacy_index: int | None
    unresolved_index: int | None
    time_delta_ms: float | None
    legacy_labels: tuple[float, float, float]
    legacy_mask: bool
    unresolved_labels: tuple[float, float, float]
    unresolved_mask: bool
    unresolved_overdue: bool
    legacy_points_to_resolved: bool


def _next_indices(targets, used, missed, now_s: float):
    if len(used) != len(targets) or len(missed) != len(targets):
        raise ValueError("Evaluator target state length mismatch")
    legacy = bisect_left([float(t.episode_time_s) for t in targets], float(now_s) - EPS_S)
    if legacy >= len(targets):
        legacy = None
    unresolved = next(
        (i for i, (u, m) in enumerate(zip(used, missed)) if not u and not m),
        None,
    )
    return legacy, unresolved


def _unresolved_labels(targets, used, missed, index, now_s, current_floor):
    """Hypothetical diagnostic target, not a policy observation or trained label.

    Suppress overdue targets instead of silently clamping negative time to
    zero and treating it as an actionable upcoming note.
    """
    if index is None:
        return (0., 0., 0.), False, False
    target = targets[index]
    delta = float(target.episode_time_s) - float(now_s)
    overdue = delta < -EPS_S
    if overdue:
        return (0., 0., 0.), False, True
    visible = (current_floor - DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors
               <= target.floor_index
               <= current_floor + DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors)
    chord = sum(
        not used[i] and not missed[i]
        and abs(float(t.episode_time_s) - float(target.episode_time_s)) < 1e-7
        for i, t in enumerate(targets)
    )
    if chord > p.KEY_COUNT:
        visible = False
    near = delta <= p.FORECAST_WINDOW_S + EPS_S
    return (min(delta, p.FORECAST_WINDOW_S), float(near),
            float(min(chord, p.KEY_COUNT)-1 if near else 0)), bool(visible), False


def compare_targets(targets, used, missed, now_s: float, current_floor: int):
    """Pure Train/evaluator-only comparison; never feed result into Student."""
    legacy, unresolved = _next_indices(targets, used, missed, now_s)
    old_labels, old_mask = p.target_labels(
        targets, now_s, current_floor,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors)
    next_labels, next_mask, overdue = _unresolved_labels(
        targets, used, missed, unresolved, now_s, current_floor)
    if legacy == unresolved:
        kind = "aligned"
    elif unresolved is None:
        kind = "no_unresolved_target"
    elif legacy is None:
        kind = "legacy_has_no_future"
    elif legacy < unresolved:
        kind = "legacy_points_behind_unresolved"
    else:
        kind = "legacy_skips_unresolved"
    delta_ms = (float(targets[unresolved].episode_time_s)
                - float(targets[legacy].episode_time_s))*1000. if (
                    legacy is not None and unresolved is not None) else None
    return TargetAlignment(
        kind=kind, legacy_index=legacy, unresolved_index=unresolved,
        time_delta_ms=delta_ms,
        legacy_labels=old_labels, legacy_mask=old_mask,
        unresolved_labels=next_labels, unresolved_mask=next_mask,
        unresolved_overdue=overdue,
        legacy_points_to_resolved=bool(
            legacy is not None and (used[legacy] or missed[legacy])),
    )


def _frames_for_train_anchor(named, config, *, anchor):
    env = DiagnosticHudNKeyRealChartMotorEnv(
        named.segment, key_count=p.KEY_COUNT, capture_overload_trace=True,
        control_dt_s=p.CONTROL_DT, physics_dt_s=p.PHYSICS_DT,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors)
    teacher = CenterFirstNKeyTeacher(p.KEY_COUNT)
    obs = env.reset()
    counters = {
        "frames": 0, "mismatch_frames": 0, "legacy_points_to_resolved": 0,
        "legacy_points_behind_unresolved": 0, "legacy_skips_unresolved": 0,
        "legacy_has_no_future": 0, "no_unresolved_target": 0,
        "unresolved_overdue": 0, "both_observable_frames": 0,
        "near_flag_disagreement_frames": 0,
        "near_time_disagreement_frames": 0,
        "time_delta_sum_abs_ms": 0., "time_delta_count": 0,
        "teacher_positive_command_mismatch_frames": 0,
        "teacher_positive_command_match_frames": 0,
    }
    examples = []
    max_steps = int((named.segment.duration_s + 2.0) / p.CONTROL_DT) + 200
    for frame in range(max_steps):
        visible = encode_n_key_hud_real_chart_observation(obs, legacy_bpm_encoding=True)
        if len(visible) != p.INPUT_DIM:
            raise RuntimeError("Visible 263D HUD changed")
        now_s = float(env.privileged_episode_time_s())
        current_floor = env._floor_index_at_chart_time(
            named.segment.chart_time_from_episode(now_s))
        comparison = compare_targets(
            named.segment.targets, env._used, env._missed, now_s, current_floor)
        if comparison.unresolved_index != env._next_target_index():
            raise RuntimeError("Compared target does not match physics next target")
        command = _teacher_action(env, teacher, obs, lead_s=config.lead_s)
        pressing = any(value > 0.0 for value in command.values)
        counters["frames"] += 1
        if comparison.kind != "aligned":
            counters["mismatch_frames"] += 1
            counters[comparison.kind] += 1
            if pressing:
                counters["teacher_positive_command_mismatch_frames"] += 1
            if len(examples) < SAMPLE_LIMIT:
                examples.append({
                    "frame": frame, "time_s": now_s,
                    "kind": comparison.kind,
                    "legacy_index": comparison.legacy_index,
                    "unresolved_index": comparison.unresolved_index,
                    "legacy_points_to_resolved": comparison.legacy_points_to_resolved,
                    "time_delta_ms": comparison.time_delta_ms,
                    "legacy_labels": comparison.legacy_labels,
                    "unresolved_labels": comparison.unresolved_labels,
                    "teacher_action_8k": list(command.values),
                })
        elif pressing:
            counters["teacher_positive_command_match_frames"] += 1
        if comparison.legacy_points_to_resolved:
            counters["legacy_points_to_resolved"] += 1
        if comparison.unresolved_overdue:
            counters["unresolved_overdue"] += 1
        if comparison.legacy_mask and comparison.unresolved_mask:
            counters["both_observable_frames"] += 1
            if comparison.legacy_labels[1] != comparison.unresolved_labels[1]:
                counters["near_flag_disagreement_frames"] += 1
            elif comparison.legacy_labels[1] >= .5:
                if abs(comparison.legacy_labels[0]
                       - comparison.unresolved_labels[0]) > EPS_S:
                    counters["near_time_disagreement_frames"] += 1
        if comparison.kind != "aligned" and comparison.time_delta_ms is not None:
            counters["time_delta_sum_abs_ms"] += abs(comparison.time_delta_ms)
            counters["time_delta_count"] += 1
        step = env.step(command)
        obs = step.observation
        if step.done:
            break
    else:
        raise RuntimeError("Train-only Expert alignment audit exceeded step budget")
    trace = env.overload_trace
    if trace is None or len(trace.action_frames) != counters["frames"]:
        raise RuntimeError("Expert physical trace frame mismatch")
    stats = env.stats
    if len(trace.physical_keydowns) != env.physical_keydowns:
        raise RuntimeError("Expert physical KeyDown trace mismatch")
    if len(trace.too_early_events) != stats.too_early_presses:
        raise RuntimeError("Expert Too Early trace mismatch")
    return {
        "anchor": anchor, "chart": named.chart_name,
        "chart_sha256": named.chart_sha256,
        "counts": counters, "examples": examples,
        "physics": {
            "hits": int(stats.hits), "misses": int(stats.misses),
            "targets": int(stats.targets),
            "too_early": int(stats.too_early_presses),
            "overload": bool(stats.overloaded),
            "keydowns": int(env.physical_keydowns),
            "unjudged": int(stats.targets) - int(stats.hits) - int(stats.misses),
        },
    }


def _train_manifest(config):
    from dmdod.data.multichart import discover_multichart_dataset
    dataset = discover_multichart_dataset(config.dataset)
    if not dataset.train:
        raise RuntimeError("Fail Closed: Train role is empty")
    anchors_per_chart = config.anchors_per_chart or math.ceil(20 / len(dataset.train))
    all_train = build_anchor_segments(
        compile_role(dataset.train), window_s=config.train_window_s,
        anchors_per_chart=anchors_per_chart)
    if len(all_train) < 20:
        raise RuntimeError("Fail Closed: Train20 anchors missing")
    selected = all_train[:20]
    if any(not seg.role.startswith("anchor-") for seg in selected):
        raise RuntimeError("Non-Train anchor supplied")
    manifest = [dict(anchor=i, name=seg.chart_name, sha256=seg.chart_sha256)
                for i, seg in enumerate(selected, 1)]
    return selected, manifest


def _provenance(config):
    root = transfer.output_dir(config)
    path_a = root / p.STAGE_A_REPORT
    path_c = root / transfer.MOTOR_REPORT
    stage_a = json.loads(path_a.read_text(encoding="utf-8"))
    stage_c = json.loads(path_c.read_text(encoding="utf-8"))
    if (stage_a.get("format") != p.VERSION
            or stage_a.get("stage") != "A"
            or stage_a.get("metadata", {}).get("config_sha256") != config.signature()
            or stage_a.get("gate", {}).get("passed") is not False
            or stage_a.get("gate", {}).get("gate_data_role") != "train_heldout"
            or stage_c.get("format") != transfer.EXPERIMENT_VERSION
            or stage_c.get("stage") != "C"
            or stage_c.get("exploratory") is not True
            or stage_c.get("stage_a_original_gate") != stage_a["gate"]
            or stage_c.get("stage_a_gate_overridden") is not False
            or stage_c.get("validation_used_for_selection") is not False
            or stage_c.get("no_final") is not True):
        raise RuntimeError("Existing Stage A/C provenance mismatch")
    return stage_a, stage_c


def _protected_paths(config, stage_c):
    root = transfer.output_dir(config)
    checkpoints = transfer.checkpoint_dir(config)
    files = {
        "stage_a": root / p.STAGE_A_REPORT,
        "stage_b": root / transfer.TRANSFER_REPORT,
        "stage_c": root / transfer.MOTOR_REPORT,
        "gru_v2": root / "gru_v2.pt",
        "core": Path(config.core_path),
        "control": checkpoints / "control_v2.pt",
        "prestige": checkpoints / "prestige_v2.pt",
    }
    for arm in ("control", "prestige"):
        expected = stage_c.get("results", {}).get(arm, {}).get("checkpoint_sha256")
        if not expected or p.sha256(files[arm]) != expected:
            raise RuntimeError(f"{arm} checkpoint digest changed")
    return files


def run_train_alignment_audit(config):
    output = transfer.output_dir(config) / REPORT
    if output.exists():
        raise FileExistsError(f"Train-only audit output already exists: {output}")
    old_best = p._protect_old_best(config)
    stage_a, stage_c = _provenance(config)
    segments, manifest = _train_manifest(config)
    expected = stage_a["metadata"]["dataset_split"]["train"]
    if manifest != expected:
        raise RuntimeError("Train20 manifest differs from Stage A v2")
    files = _protected_paths(config, stage_c)
    before = {name: p.sha256(path) for name, path in files.items()}
    results = []
    for i, named in enumerate(segments, 1):
        result = _frames_for_train_anchor(named, config, anchor=i)
        results.append(result)
        counts = result["counts"]
        print(f"Prestige Train-only alignment {i:02d}/20: "
              f"mismatch={counts['mismatch_frames']}/{counts['frames']} "
              f"resolved-target={counts['legacy_points_to_resolved']} "
              f"near-disagreement={counts['near_flag_disagreement_frames']}",
              flush=True)
    after = {name: p.sha256(path) for name, path in files.items()}
    if before != after or old_best != p._protect_old_best(config):
        raise RuntimeError("Protected artifacts modified during Train-only audit")
    summed = {
        name: sum(item["counts"][name] for item in results)
        for name in results[0]["counts"]
    }
    report = {
        "format": FORMAT, "stage": "C-v2.2-train-audit",
        "role": "Train20 only", "validation_evaluated": False,
        "final_evaluated": False, "no_training": True, "read_only": True,
        "exploratory": True, "validation_used_for_selection": False,
        "stage_a_original_gate": stage_a["gate"], "stage_a_gate_overridden": False,
        "model_checkpoint_loaded": False, "config_sha256": config.signature(),
        "compared_trajectories": "Expert-controlled Train20 only",
        "existing_training_labels_modified": False,
        "audited_hypothesis": "Legacy time-bisect next future note vs actual next unresolved physical target",
        "proposed_future_change": "Consider Train-only unresolved-target label design and explicit actuator-aware policy gating; none are applied in this audit",
        "train_manifest": manifest, "aggregate_counts": summed,
        "per_anchor": results, "protected_input_sha256": before,
        "protected_inputs_unchanged": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    p._no_replace_json(output, report)
    return report
