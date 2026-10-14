from __future__ import annotations

"""Prestige I Stage C v2.5: Train20-only Teacher quality and label audit.

Replay the already-fixed v2.4 Teacher. The physical evaluator supplies the
truth for diagnostics ONLY; no policy, checkpoint or Stage A label is changed.
"""

from collections import Counter
import json
import math
from pathlib import Path

from dmdod.adofai_rules import TimingJudgement

from . import prestige1 as p
from . import prestige1_alignment_audit as alignment
from . import prestige1_teacher_safety as safety
from . import prestige1_actuator_comparison as v24
from . import prestige1_transfer as transfer
from .prestige1_actuator_teacher import ActuatorAwareTeacherSafety

FORMAT = "prestige1-stage-c-v25-train-only-teacher-quality"
REPORT = "stage_c_v25_teacher_quality.json"
VALID_HIT_JUDGEMENTS = frozenset((
    TimingJudgement.PERFECT, TimingJudgement.EARLY_PERFECT,
    TimingJudgement.LATE_PERFECT, TimingJudgement.VERY_EARLY,
    TimingJudgement.VERY_LATE,
))
PERFECT_FAMILY = frozenset((
    TimingJudgement.PERFECT.value,
    TimingJudgement.EARLY_PERFECT.value,
    TimingJudgement.LATE_PERFECT.value,
))


def percentile(samples, fraction: float) -> float | None:
    """Linear interpolation at index (n-1)*fraction, even for tiny samples."""
    if not 0 <= fraction <= 1:
        raise ValueError("percentile fraction must lie within [0, 1]")
    if not samples:
        return None
    ordered = sorted(float(x) for x in samples)
    if any(not math.isfinite(x) for x in ordered):
        raise ValueError("nonfinite timing measurement")
    position = (len(ordered) - 1) * fraction
    low = int(math.floor(position))
    high = int(math.ceil(position))
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_hit_quality(hit_events: list[dict]) -> dict:
    """Errors are signed ms; hit-only statistics exclude TooEarly and Miss."""
    errors = [float(event["error_ms"]) for event in hit_events]
    abs_errors = [abs(value) for value in errors]
    counts = Counter(event["judgement"] for event in hit_events)
    total = len(hit_events)
    if any(event["judgement"] not in {j.value for j in VALID_HIT_JUDGEMENTS}
           for event in hit_events):
        raise ValueError("Non-Hit event passed to timing quality summary")
    return {
        "hits": total,
        "mean_signed_error_ms": sum(errors) / total if total else None,
        "mae_ms": sum(abs_errors) / total if total else None,
        "median_abs_error_ms": percentile(abs_errors, .5),
        "p95_abs_error_ms": percentile(abs_errors, .95),
        "min_signed_error_ms": min(errors) if total else None,
        "max_signed_error_ms": max(errors) if total else None,
        "pure_perfect_hits": counts[TimingJudgement.PERFECT.value],
        "pure_perfect_rate": counts[TimingJudgement.PERFECT.value] / total if total else None,
        "perfect_family_hits": sum(counts[j] for j in PERFECT_FAMILY),
        "perfect_family_rate": (
            sum(counts[j] for j in PERFECT_FAMILY) / total if total else None
        ),
        "judgement_counts": dict(sorted(counts.items())),
    }


def classify_physical_keydowns(events: list[dict], env, expected: dict) -> dict:
    """Verify each recorded physical KeyDown using the actual chart judgement.

    The trace target_index was captured *at scoring*, before the event changed
    the unresolved target. Never attribute an event by nearest chart time.
    """
    hits = []
    too_early = 0
    too_late = 0
    no_target = 0
    if len(events) != int(expected["keydowns"]):
        raise RuntimeError("Physical KeyDown count differs from fixed result")
    for event in events:
        idx = event["target_index"]
        if idx is None:
            no_target += 1
            continue
        if not 0 <= int(idx) < len(env.segment.targets):
            raise RuntimeError("Out-of-range physical target index")
        target = env.segment.targets[idx]
        if not math.isclose(float(event["target_time_s"]),
                            float(target.episode_time_s), abs_tol=1e-8):
            raise RuntimeError("Physical trace target time diverges from chart")
        error_ms = (float(event["time_s"]) - float(target.episode_time_s)) * 1000.
        judgement = env._classify(error_ms / 1000., target)
        if judgement is TimingJudgement.TOO_EARLY:
            too_early += 1
        elif judgement is TimingJudgement.TOO_LATE:
            too_late += 1
        elif judgement in VALID_HIT_JUDGEMENTS:
            hits.append({
                "time_s": float(event["time_s"]),
                "target_index": int(idx),
                "key": event["key"],
                "error_ms": error_ms,
                "judgement": judgement.value,
            })
        else:
            raise RuntimeError("Unsupported physical hit judgement")
    if len(hits) != int(expected["hits"]) or too_early != int(expected["too_early"]):
        raise RuntimeError("Physics Hit/TooEarly reconstruction mismatch")
    return {
        "quality": summarize_hit_quality(hits),
        "hit_events": hits,
        "too_early_keydowns": too_early,
        "too_late_keydowns": too_late,
        "no_target_keydowns": no_target,
    }


def command_sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


class CommandTransitions:
    """Compare unmodified proposals and corrected actions at the SAME states."""

    def __init__(self, width: int):
        self.width = width
        self.previous = None
        self.last_nonzero = [0] * width
        self.counts = Counter()

    def observe(self, values):
        values = tuple(command_sign(x) for x in values)
        if len(values) != self.width:
            raise ValueError("wrong action width")
        if self.previous is not None:
            for key, (prior, current) in enumerate(zip(self.previous, values)):
                if prior != current:
                    self.counts["sign_transitions"] += 1
                if prior * current == -1:
                    self.counts["direct_direction_reversals"] += 1
                if prior > 0 and current <= 0:
                    self.counts["positive_command_endings"] += 1
        for index, sign in enumerate(values):
            if sign and self.last_nonzero[index] and sign != self.last_nonzero[index]:
                self.counts["nonzero_direction_reversals"] += 1
            if sign:
                self.last_nonzero[index] = sign
            if sign > 0:
                self.counts["positive_key_frames"] += 1
            elif sign < 0:
                self.counts["negative_key_frames"] += 1
        self.counts["frames"] += 1
        self.previous = values

    def summary(self) -> dict:
        keys = ("frames", "sign_transitions", "direct_direction_reversals",
                "nonzero_direction_reversals", "positive_command_endings",
                "positive_key_frames", "negative_key_frames")
        return {key: int(self.counts[key]) for key in keys}


class TeacherQualityRecorder:
    """Read original action, evaluator-only alignment, then correct and record."""

    def __init__(self, lead_s: float):
        self.filter = ActuatorAwareTeacherSafety(lead_s=lead_s)
        self.raw_transitions = CommandTransitions(p.KEY_COUNT)
        self.safe_transitions = CommandTransitions(p.KEY_COUNT)
        self.counters = Counter()
        self.examples = []
        self.action_frames = []
        self.env = None

    def __call__(self, env, original):
        if self.env is not None and self.env is not env:
            raise RuntimeError("Recorder reused on a different physical environment")
        self.env = env
        now_s = float(env.privileged_episode_time_s())
        current_floor = env._floor_index_at_chart_time(
            env.segment.chart_time_from_episode(now_s))
        result = alignment.compare_targets(
            env.segment.targets, env._used, env._missed, now_s, current_floor)
        if result.unresolved_index != env._next_target_index():
            raise RuntimeError("Alignment audit differs from physical next target")
        corrected = self.filter(env, original)
        raw = tuple(float(x) for x in original.values)
        safe = tuple(float(x) for x in corrected.values)
        self.raw_transitions.observe(raw)
        self.safe_transitions.observe(safe)
        self.counters["frames"] += 1
        if result.kind != "aligned":
            self.counters["mismatch_frames"] += 1
            self.counters[result.kind] += 1
            if any(x > 0 for x in raw):
                self.counters["original_positive_mismatch_frames"] += 1
            if any(x > 0 for x in safe):
                self.counters["corrected_positive_mismatch_frames"] += 1
            if len(self.examples) < 12:
                self.examples.append({
                    "frame": self.counters["frames"] - 1,
                    "time_s": now_s,
                    "kind": result.kind,
                    "legacy_index": result.legacy_index,
                    "unresolved_index": result.unresolved_index,
                    "legacy_points_to_resolved": result.legacy_points_to_resolved,
                    "legacy_labels": list(result.legacy_labels),
                    "unresolved_labels": list(result.unresolved_labels),
                })
        if result.legacy_points_to_resolved:
            self.counters["legacy_points_to_resolved"] += 1
        if result.unresolved_overdue:
            self.counters["unresolved_overdue_frames"] += 1
        if result.legacy_mask != result.unresolved_mask:
            self.counters["observability_mask_disagreement_frames"] += 1
        if result.legacy_mask and result.unresolved_mask:
            self.counters["both_observable_frames"] += 1
            if result.legacy_labels[1] != result.unresolved_labels[1]:
                self.counters["near_flag_disagreement_frames"] += 1
            elif result.legacy_labels[1] >= .5 and (
                abs(result.legacy_labels[0] - result.unresolved_labels[0])
                > alignment.EPS_S
            ):
                self.counters["near_time_disagreement_frames"] += 1
        changed = [key for key, x, y in zip(env.key_names, raw, safe) if x != y]
        if changed:
            self.counters["changed_frames"] += 1
            self.counters["changed_key_commands"] += len(changed)
            self.counters["positive_to_brake"] += sum(
                x > 0 and y < 0 for x, y in zip(raw, safe))
        # Both *complete* command sequences are kept to study sign flips.
        # Target IDs are evaluator-only audit metadata, not model inputs.
        self.action_frames.append({
            "time_s": now_s,
            "original_action_8k": list(raw),
            "corrected_action_8k": list(safe),
            "modified_keys": changed,
            "alignment_kind": result.kind,
            "legacy_target_index": result.legacy_index,
            "unresolved_target_index": result.unresolved_index,
        })
        return corrected

    def summary(self):
        if self.counters["frames"] != self.filter.frames:
            raise RuntimeError("Recorder and actuator frame counts differ")
        if self.counters["changed_frames"] != self.filter.changed_frames:
            raise RuntimeError("Changed frames disagree with actuator filter")
        return {
            "counts": {
                key: int(self.counters[key])
                for key in (
                    "frames", "mismatch_frames", "legacy_points_to_resolved",
                    "legacy_points_behind_unresolved", "legacy_skips_unresolved",
                    "legacy_has_no_future", "no_unresolved_target",
                    "unresolved_overdue_frames",
                    "observability_mask_disagreement_frames",
                    "both_observable_frames", "near_flag_disagreement_frames",
                    "near_time_disagreement_frames",
                    "original_positive_mismatch_frames",
                    "corrected_positive_mismatch_frames",
                    "changed_frames", "changed_key_commands",
                    "positive_to_brake"
                )
            },
            "original_transitions": self.raw_transitions.summary(),
            "corrected_transitions": self.safe_transitions.summary(),
            "actuator_interventions": self.filter.statistics(),
        }


def _check_v24_report(previous, old_report, stage_a, config, manifest):
    if (previous.get("format") != v24.FORMAT
            or previous.get("stage") != "C-v2.4"
            or previous.get("role") != "Train20 only"
            or previous.get("no_training") is not True
            or previous.get("read_only") is not True
            or previous.get("validation_evaluated") is not False
            or previous.get("final_evaluated") is not False
            or previous.get("validation_used_for_selection") is not False
            or previous.get("checkpoint_modified") is not False
            or previous.get("student_checkpoint_loaded") is not False
            or previous.get("stage_a_gate_overridden") is not False
            or previous.get("original_stage_a_gate") != stage_a["gate"]
            or previous.get("config_sha256") != config.signature()
            or previous.get("train_manifest") != manifest
            or len(previous.get("per_anchor", [])) != 20
            or previous.get("protected_inputs_unchanged") is not True
            or old_report.get("aggregate") != previous.get("baseline")
            or old_report.get("train_manifest") != manifest):
        raise RuntimeError("Frozen v2.4 / v2.3 Train-only provenance mismatch")
    if (stage_a["gate"].get("passed") is not False
            or stage_a["gate"].get("gate_data_role") != "train_heldout"):
        raise RuntimeError("Original Stage A gate must remain FAIL")


def _sum_quality(rows, field):
    events = [event for row in rows for event in row[field]["hit_events"]]
    return summarize_hit_quality(events)


def run_train_teacher_quality(config):
    output = transfer.output_dir(config) / REPORT
    if output.exists():
        raise FileExistsError("Refuse to replace v2.5 Train-only quality report")
    best_before = p._protect_old_best(config)
    stage_a, stage_c = alignment._provenance(config)
    segments, manifest = alignment._train_manifest(config)
    if manifest != stage_a["metadata"]["dataset_split"]["train"]:
        raise RuntimeError("Frozen Train20 manifest mismatch")

    root = transfer.output_dir(config)
    v24_path = root / v24.REPORT
    v23_path = root / safety.REPORT
    previous = json.loads(v24_path.read_text(encoding="utf-8"))
    old = json.loads(v23_path.read_text(encoding="utf-8"))
    _check_v24_report(previous, old, stage_a, config, manifest)
    protected = alignment._protected_paths(config, stage_c)
    protected["stage_c_v22_train_audit"] = root / alignment.REPORT
    protected["stage_c_v23_teacher_safety"] = v23_path
    original_digests = {name: p.sha256(path) for name, path in protected.items()}
    if original_digests != previous["protected_input_sha256"]:
        raise RuntimeError("v2.4 protected input SHA256 mismatch")
    protected["stage_c_v24_actuator_teacher_safety"] = v24_path
    before = {name: p.sha256(path) for name, path in protected.items()}

    rows = []
    for i, named in enumerate(segments, 1):
        baseline = previous["per_anchor"][i - 1]
        old_row = old["per_anchor"][i - 1]
        if (baseline["anchor"] != i or old_row["anchor"] != i
                or baseline["chart_sha256"] != named.chart_sha256
                or old_row["chart_sha256"] != named.chart_sha256
                or baseline["chart"] != named.chart_name
                or old_row["chart"] != named.chart_name
                or baseline["baseline"] != old_row["stats"]):
            raise RuntimeError(f"Frozen Train anchor mismatch #{i:02d}")

        recorder = TeacherQualityRecorder(lead_s=config.lead_s)
        replay = safety._run_teacher(
            named, config, anchor=i, action_filter=recorder)
        if (replay["stats"] != baseline["candidate"]
                or recorder.filter.statistics() != baseline["actuator_interventions"]):
            raise RuntimeError(f"v2.4 Teacher replay diverged at Train #{i:02d}")
        if recorder.env is None or len(recorder.action_frames) != recorder.counters["frames"]:
            raise RuntimeError("Missing full Teacher command trace")
        original = classify_physical_keydowns(
            old_row["all_physical_keydowns"], recorder.env, old_row["stats"])
        candidate = classify_physical_keydowns(
            replay["all_physical_keydowns"], recorder.env, replay["stats"])
        rows.append({
            "anchor": i,
            "chart": named.chart_name, "chart_sha256": named.chart_sha256,
            "baseline": original, "candidate": candidate,
            "physics_stats": replay["stats"],
            "alignment": recorder.summary(),
            "alignment_examples": recorder.examples,
            "action_frames": recorder.action_frames,
        })
        old_mae = original["quality"]["mae_ms"]
        new_mae = candidate["quality"]["mae_ms"]
        print(
            f"Stage C v2.5 Train {i:02d}/20: "
            f"MAE old={f'{old_mae:.2f}' if old_mae is not None else 'n/a'}ms "
            f"new={f'{new_mae:.2f}' if new_mae is not None else 'n/a'}ms "
            f"mismatch={recorder.counters['mismatch_frames']}/"
            f"{recorder.counters['frames']} "
            f"action-changes={recorder.filter.changed_frames}",
            flush=True)

    if ({name: p.sha256(path) for name, path in protected.items()} != before
            or best_before != p._protect_old_best(config)):
        raise RuntimeError("Protected checkpoints/reports changed during v2.5")

    candidate_aggregate = v24._aggregate(
        [{"anchor": row["anchor"], "candidate": row["physics_stats"]}
         for row in rows], "candidate")
    if candidate_aggregate != previous["candidate"]:
        raise RuntimeError("v2.4 aggregate physics reproduction mismatch")
    counts = {
        key: sum(row["alignment"]["counts"][key] for row in rows)
        for key in rows[0]["alignment"]["counts"]
    }
    report = {
        "format": FORMAT, "stage": "C-v2.5", "role": "Train20 only",
        "exploratory": True, "no_training": True, "read_only": True,
        "student_checkpoint_loaded": False, "checkpoint_modified": False,
        "validation_evaluated": False, "final_evaluated": False,
        "validation_used_for_selection": False,
        "original_stage_a_gate": stage_a["gate"],
        "stage_a_gate_overridden": False,
        "config_sha256": config.signature(),
        "train_manifest": manifest,
        "baseline_scope": "v2.3 Hit events on original truncated trajectories",
        "candidate_scope": "v2.4 Hit events on full SAFE Train20 trajectories",
        "alignment_scope": "v2.4 replay at 10ms control frames, v2.2 definitions",
        "action_scope": "Unfiltered and corrected Teacher proposals at identical v2.4 states",
        "timing_quality_baseline": _sum_quality(rows, "baseline"),
        "timing_quality_candidate": _sum_quality(rows, "candidate"),
        "alignment_aggregate": counts,
        "physics_candidate": candidate_aggregate,
        "per_anchor": rows,
        "protected_input_sha256": before,
        "protected_inputs_unchanged": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    p._no_replace_json(output, report)
    return report
