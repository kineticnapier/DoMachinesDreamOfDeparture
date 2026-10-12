from __future__ import annotations

"""Train-only counterfactual motor v1.2: no optimization or teacher adoption.

Existing v1.1 short-pass cases are replayed from a fresh 8-key motor reset.
Privileged targets are read only by the environment/independent expert.
Validation/FINAL are never evaluated. Student receives its existing 263D HUD.
"""

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

EXPECTED_BEST_SHA256 = "476696a2a157c15a00d602845208a21049dcca94479cee9b91bfeb105cac62e0"
EXPECTED_V11_SHORT_PASSERS = 151
HORIZONS = (0.400, 0.800)
FOCUS_ANCHORS = (6, 8, 18, 20)
KEYDOWN_TIME_TOL = 1e-5


@dataclass(frozen=True)
class Snapshot:
    """Scoring at a control boundary, not an extrapolated final score."""
    time_s: float
    hits: int
    misses: int
    too_early: int
    overloaded: bool
    overload_time_s: float | None
    terminated: bool
    termination_reason: str
    hit_ids: tuple[int, ...]
    miss_ids: tuple[int, ...]
    keydowns: tuple[tuple[str, float], ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "time_s": self.time_s, "hits": self.hits, "misses": self.misses,
            "too_early": self.too_early, "overloaded": self.overloaded,
            "overload_time_s": self.overload_time_s, "terminated": self.terminated,
            "termination_reason": self.termination_reason,
            "hit_ids": list(self.hit_ids), "miss_ids": list(self.miss_ids),
            "keydowns": [[key, time] for key, time in self.keydowns],
        }


def source_case(row: dict[str, str], *, key_names: tuple[str, ...]):
    """Strictly decode a saved v1.1 Probe (not regenerate/rerank it)."""
    from dmdod.training.train_counterfactual_motor_v11 import Probe
    names = {name: i for i, name in enumerate(key_names)}
    keys = tuple(names[name] for name in row["keys"].split(","))
    if not keys or len(keys) != len(set(keys)) or len(keys) not in (1, 2, 8):
        raise ValueError("invalid stored candidate key set")
    start = int(row["start_frame"])
    duration = int(row["duration_frames"])
    lead = int(row["lead_frames"])
    delta = float(row["delta"])
    if start < 0 or duration < 1 or lead < 0 or not math.isfinite(delta):
        raise ValueError("invalid stored intervention")
    if abs(delta) not in (0.02, 0.05, 0.1, 0.2):
        # Avoid false equality failures from decimal representations.
        if not any(abs(abs(delta) - x) < 1e-9 for x in (.02, .05, .10, .20)):
            raise ValueError("intervention delta not in original v1.1 grid")
    case = Probe(keys=keys, start=start, duration=duration, delta=delta, lead=lead)
    if case.kind != row["kind"]:
        raise ValueError("stored candidate kind disagrees with key selection")
    return case


def case_identity(anchor: int, case: Any) -> tuple:
    return (anchor, tuple(case.keys), int(case.start), int(case.duration), float(case.delta))


def load_v11_short_passers(directory: Path, key_names: tuple[str, ...]) -> tuple[dict, list[dict]]:
    report_path = directory / "diagnostic.json"
    csv_path = directory / "train_candidate_gates.csv"
    if not report_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError("v1.1 diagnostic.json and train_candidate_gates.csv are required")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("kind") != "train_counterfactual_motor_diagnostic_v11"
            or report.get("source_sha256") != EXPECTED_BEST_SHA256
            or report.get("validation_evaluations") != 0
            or report.get("optimizer_updates") != 0
            or report.get("final_used") is not False
            or abs(float(report.get("followup_horizon_s", 0)) - .8) > 1e-9):
        raise RuntimeError("v1.1 provenance / diagnostic contract mismatch")
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        raw = list(csv.DictReader(stream))
    if len(raw) != int(report.get("train_candidate_evaluations", -1)):
        raise RuntimeError("v1.1 candidate count differs from saved report")
    selected, identities = [], set()
    for row in raw:
        if row["short_status"] == "rejected_short":
            continue
        if row["followup_checked"].strip().lower() not in ("true", "1"):
            raise RuntimeError("v1.1 short-passer without 800ms followup")
        if any(row.get(field) in (None, "") for field in (
                "followup_hits", "followup_early", "followup_misses",
                "followup_keydowns", "followup_overload", "followup_duration_s")):
            raise RuntimeError("v1.1 saved followup is incomplete")
        anchor = int(row["anchor"])
        if not 1 <= anchor <= 20:
            raise RuntimeError("candidate is not one of Train20 anchors")
        case = source_case(row, key_names=key_names)
        identity = case_identity(anchor, case)
        if identity in identities:
            raise RuntimeError("duplicate v1.1 short-passer identity")
        identities.add(identity)
        selected.append({"anchor": anchor, "case": case, "old": row,
                         "source": "v11_short_pass"})
    if len(selected) != EXPECTED_V11_SHORT_PASSERS:
        raise RuntimeError(f"v1.1 shortlist count mismatch: {len(selected)} != 151")
    return report, selected


def _termination_reason(env, *, finished: bool) -> str:
    if env._overload.overloaded:
        return "Overload"
    if env._failed_on_miss:
        return "Miss failure"
    if finished:
        return "All targets resolved" if all(
            u or m for u, m in zip(env._used, env._missed)
        ) else "Time limit"
    return "Horizon cutoff"


def _environment_snapshot(env, *, finished: bool = False) -> Snapshot:
    keydowns = tuple((e.key, float(e.time_s)) for e in env._trace_physical_keydowns)
    time = float(env.privileged_episode_time_s())
    overload_time = getattr(env, "_trace_overload_event_time_s", None)
    if env._overload.overloaded and overload_time is None:
        raise RuntimeError("Overload occurred without a recorded triggering timestamp")
    return Snapshot(
        time_s=time, hits=int(env._hits), misses=int(env._misses),
        too_early=int(env._too_early), overloaded=bool(env._overload.overloaded),
        overload_time_s=float(overload_time) if overload_time is not None else None,
        terminated=finished, termination_reason=_termination_reason(env, finished=finished),
        hit_ids=tuple(i for i, v in enumerate(env._used) if v),
        miss_ids=tuple(i for i, v in enumerate(env._missed) if v),
        keydowns=keydowns,
    )


def _assert_snapshot_consistent(snapshot: Snapshot) -> None:
    if snapshot.hits != len(snapshot.hit_ids):
        raise RuntimeError("Hit count and target-index set disagree")
    if snapshot.misses != len(snapshot.miss_ids):
        raise RuntimeError("Miss count and target-index set disagree")
    if set(snapshot.hit_ids) & set(snapshot.miss_ids):
        raise RuntimeError("target was simultaneously Hit and Miss")


def run_trace(prepared, named, *, case=None, expert=False,
              max_s: float = .800) -> tuple[Snapshot, Snapshot, tuple[Snapshot, ...]]:
    """One 800ms physical rollout, with an actual 400ms snapshot.

    If it terminates before either horizon, later snapshots are the same
    terminal snapshot (never extrapolated or falsely marked as reached).
    """
    if abs(max_s - .800) > 1e-9:
        raise ValueError("v1.2 uses fixed 800ms horizon")
    from dmdod.training.motor_sensitivity_audit import _environment
    from dmdod.envs.n_key import encode_n_key_hud_real_chart_observation
    from dmdod.motor.n_key import NKeyAction
    from dmdod.n_key_capacity import CenterFirstNKeyTeacher
    from dmdod.training.n_key import _teacher_action
    import torch
    env = _environment(prepared, named)
    obs = env.reset()
    model = prepared.model
    if expert:
        teacher = CenterFirstNKeyTeacher(prepared.key_count)
    else:
        state = model.initial_state(prepared.device)
        model.eval()
    frames: list[Snapshot] = []
    at_400 = None
    limit = math.ceil(.800 / prepared.control_dt_s)
    with torch.no_grad():
        for frame in range(limit):
            if expert:
                action = _teacher_action(env, teacher, obs, lead_s=prepared.lead_s)
            else:
                vector = encode_n_key_hud_real_chart_observation(obs)
                if len(vector) != 263:
                    raise RuntimeError("Student visible input changed from 263D")
                means, _, _, state = model.forward_step(
                    torch.tensor(vector, dtype=torch.float32, device=prepared.device), state
                )
                original = tuple(float(v.item()) for v in torch.tanh(means))
                values = list(original)
                if case is not None and case.start <= frame < case.end:
                    for index in case.keys:
                        values[index] = max(-1.0, min(1.0, values[index] + case.delta))
                action = NKeyAction(tuple(values))
            transition = env.step(action)
            obs = transition.observation
            snapshot = _environment_snapshot(env, finished=bool(transition.done))
            _assert_snapshot_consistent(snapshot)
            frames.append(snapshot)
            if frame + 1 == math.ceil(.400 / prepared.control_dt_s):
                at_400 = snapshot
            if transition.done:
                break
    if not frames:
        raise RuntimeError("empty motor rollout")
    end = frames[-1]
    return at_400 or end, end, tuple(frames)


def snapshot_at(frames: tuple[Snapshot, ...], clock: float) -> Snapshot:
    """Observation at shared control boundary, never beyond the recorded run."""
    if clock < 0:
        raise ValueError("comparison cutoff must be non-negative")
    candidates = [s for s in frames if s.time_s <= clock + 1e-7]
    if not candidates:
        raise RuntimeError("no common observed motor control frame")
    return candidates[-1]


def assert_prefix_consistent(short: Snapshot, long: Snapshot) -> None:
    if short.time_s > long.time_s + 1e-7:
        raise RuntimeError("400ms / 800ms trajectory clock reversed")
    if not set(short.hit_ids).issubset(long.hit_ids):
        raise RuntimeError("400ms Hit disappeared in 800ms prefix")
    if not set(short.miss_ids).issubset(long.miss_ids):
        raise RuntimeError("400ms Miss disappeared in 800ms prefix")
    if short.too_early > long.too_early or len(short.keydowns) > len(long.keydowns):
        raise RuntimeError("800ms trace lost 400ms physical events")
    if short.overloaded and (not long.overloaded or long.time_s != short.time_s):
        raise RuntimeError("terminal Overload inconsistently continued")


def verify_v11_baselines(v11: dict, selected: list[dict],
                         baselines: list[dict]) -> None:
    """Verify saved 400ms Student origin before interpreting any 800ms effect."""
    anchors = v11.get("anchors", [])
    if len(anchors) != 20 or len(baselines) != 20:
        raise RuntimeError("Fail Closed: v1.1 Train20 baseline count mismatch")
    for i, (stored, current) in enumerate(zip(anchors, baselines), 1):
        short = current["student_400"]
        if (int(stored["anchor"]) != i or
            stored["chart"] != current["chart"] or
            _saved_bool(str(stored["base_overload_400ms"])) != short["overloaded"] or
            int(stored["base_early_400ms"]) != short["too_early"]):
            raise RuntimeError(f"Fail Closed: v1.1 Train #{i:02d} 400ms baseline mismatch")
    for item in selected:
        row = item["old"]
        short = baselines[item["anchor"] - 1]["student_400"]
        if (int(row["base_hits"]) != short["hits"] or
            int(row["base_early"]) != short["too_early"] or
            _saved_bool(str(row["base_overload"])) != short["overloaded"]):
            raise RuntimeError(
                f"Fail Closed: v1.1 Train #{item['anchor']:02d} candidate origin mismatch"
            )


def _saved_bool(s: str) -> bool:
    if s.lower() in ("true", "1"):
        return True
    if s.lower() in ("false", "0"):
        return False
    raise ValueError("invalid bool in v1.1 saved diagnostic")


def assert_v11_reproduced(row: dict, short: Snapshot, long: Snapshot,
                          *, atol_s: float = 1e-5) -> None:
    comparisons = (
        ("short_hits", short.hits), ("short_early", short.too_early),
        ("short_keydowns", len(short.keydowns)),
        ("followup_hits", long.hits), ("followup_early", long.too_early),
        ("followup_misses", long.misses),
        ("followup_keydowns", len(long.keydowns)),
    )
    for field, measured in comparisons:
        if int(row[field]) != measured:
            raise RuntimeError(f"Fail Closed: v1.1 {field} {row[field]} != {measured}")
    for field, measured in (("short_overload", short.overloaded),
                            ("followup_overload", long.overloaded)):
        if _saved_bool(row[field]) != measured:
            raise RuntimeError(f"Fail Closed: v1.1 {field} disagreement")
    if abs(float(row["followup_duration_s"]) - long.time_s) > atol_s:
        raise RuntimeError("Fail Closed: v1.1 800ms end-time disagreement")
    if (int(row["followup_new_hits"]) != long.hits - short.hits or
            int(row["followup_new_keydowns"]) != len(long.keydowns) - len(short.keydowns) or
            int(row["followup_new_too_early"]) != long.too_early - short.too_early):
        raise RuntimeError("Fail Closed: v1.1 400/800 incremental result disagrees")
    assert_prefix_consistent(short, long)


def common_interval_score(reference: Snapshot, candidate: Snapshot,
                          reference_frames: tuple[Snapshot, ...],
                          candidate_frames: tuple[Snapshot, ...]) -> dict:
    # End of observed overlap. A reference dying at 100ms cannot acquire
    # hypothetical 800ms misses just to compare against surviving candidates.
    common_end = min(reference.time_s, candidate.time_s)
    b = snapshot_at(reference_frames, common_end)
    c = snapshot_at(candidate_frames, common_end)
    lost = sorted(set(b.hit_ids) - set(c.hit_ids))
    gained = sorted(set(c.hit_ids) - set(b.hit_ids))
    return {
        "observed_until_s": common_end,
        "reference_hit_ids": list(b.hit_ids), "candidate_hit_ids": list(c.hit_ids),
        "lost_hit_ids": lost, "gained_hit_ids": gained,
        "hits_preserved": not lost, "common_hit_gain": len(gained),
        "reference_misses": b.misses, "candidate_misses": c.misses,
        "misses_not_increased": c.misses <= b.misses,
        "reference_too_early": b.too_early, "candidate_too_early": c.too_early,
        "too_early_reduced": c.too_early < b.too_early,
        "reference_keydowns": len(b.keydowns),
        "candidate_keydowns": len(c.keydowns),
    }


def classify_student_counterfactual(
        base_400: Snapshot, base_800: Snapshot, base_frames: tuple[Snapshot, ...],
        cand_400: Snapshot, cand_800: Snapshot, cand_frames: tuple[Snapshot, ...],
        *, expert_800: Snapshot, old_row: dict | None = None) -> dict:
    """Student-relative evidence; teacher is diagnostics only, never a gate."""
    assert_prefix_consistent(base_400, base_800)
    assert_prefix_consistent(cand_400, cand_800)
    common = common_interval_score(base_800, cand_800, base_frames, cand_frames)
    known = set(base_800.hit_ids)
    preserved_at_800 = known.issubset(cand_800.hit_ids)
    final_lost_ids = sorted(known - set(cand_800.hit_ids))
    extension = (cand_800.time_s > base_800.time_s + 1e-7)
    after_common = snapshot_at(cand_frames, common["observed_until_s"])
    extension_hits = set(cand_800.hit_ids) - set(after_common.hit_ids)
    extension_misses = set(cand_800.miss_ids) - set(after_common.miss_ids)
    extension_downs = len(cand_800.keydowns) - len(after_common.keydowns)
    reaches_800 = cand_800.time_s >= .800-1e-7 and not cand_800.overloaded
    # A later fatal Overload is a deferment, not a rescue.
    overload_rescued = base_800.overloaded and reaches_800
    slowed_overload = bool(base_800.overloaded and cand_800.overloaded and
                            cand_800.overload_time_s is not None and
                            base_800.overload_time_s is not None and
                            cand_800.overload_time_s > base_800.overload_time_s + .01)
    no_press_suppression = (len(cand_800.keydowns) == 0 or
                            (len(base_800.keydowns) > 0 and
                             len(cand_800.keydowns) < len(base_800.keydowns) and
                             not extension_hits and cand_800.hits <= base_800.hits))
    # At 400ms a favorable measurement must not be nullified at 800ms.
    deferred_failure = bool(
        (not cand_400.overloaded and cand_800.overloaded) or
        (cand_400.too_early < base_400.too_early and
         cand_800.too_early >= base_800.too_early and
         cand_800.time_s >= base_800.time_s - 1e-7) or
        (extension and not extension_hits and extension_downs == 0)
    )
    effective_progress = bool(extension_hits or common["common_hit_gain"] > 0 or
                              (cand_800.hits > base_800.hits and preserved_at_800))
    safe_improvement = (
        common["hits_preserved"] and preserved_at_800 and
        common["misses_not_increased"] and
        not no_press_suppression and not deferred_failure and
        (not cand_800.overloaded) and reaches_800
    )
    booleans = {
        "overload_rescue": overload_rescued,
        "overload_delayed_not_rescued": slowed_overload,
        "too_early_reduced_common": common["too_early_reduced"],
        "baseline_hit_ids_preserved_common": common["hits_preserved"],
        "baseline_hit_ids_preserved_800": preserved_at_800,
        "misses_not_increased_common": common["misses_not_increased"],
        "has_physical_keydowns": len(cand_800.keydowns)>0,
        "no_press_suppression": not no_press_suppression,
        "continuation_has_new_hits": bool(extension_hits),
        "continuation_has_keydowns": extension_downs > 0,
        "reached_800_without_overload": reaches_800,
        "not_deferred_failure": not deferred_failure,
        "student_relative_improvement": False,
    }
    if final_lost_ids or not common["hits_preserved"]:
        category = "hit_suppression"
    elif not common["misses_not_increased"] or (cand_800.overloaded and not base_800.overloaded):
        category = "regression"
    elif no_press_suppression:
        category = "hit_suppression"
    elif deferred_failure:
        category = "deferred_failure"
    elif safe_improvement and overload_rescued and effective_progress:
        category = "overload_rescue"
    elif safe_improvement and common["too_early_reduced"] and cand_800.hits >= base_800.hits:
        category = "early_reduction_preserved_hits"
    elif safe_improvement and cand_800.hits > base_800.hits:
        category = "hit_gain"
    else:
        category = "inconclusive"
    booleans["student_relative_improvement"] = category in (
        "overload_rescue", "early_reduction_preserved_hits", "hit_gain")
    # Diagnostic suitability is separate from categorical classification and
    # never performs an actual teacher/curriculum update.
    eligible = bool(booleans["student_relative_improvement"] and safe_improvement)
    old = old_row or {}
    # v1.1 has teacher-relative gates at BOTH 400ms and 800ms. The
    # shortlist already passed the 400ms teacher test, so the followup
    # teacher gate is the relevant source of post-400ms rejection.
    old_teacher_gate = old.get("followup_expert_relative_hits",
                               old.get("gate_expert_relative_hits"))
    old_qualified = old.get("followup_qualified")
    teacher_gate_passed = None if old_teacher_gate is None else _saved_bool(str(old_teacher_gate))
    legacy_qualified = None if old_qualified is None else _saved_bool(str(old_qualified))
    return {
        "category": category, "checks": booleans,
        "comparison_common": common,
        "extended_observation": {
            "reference_stopped_s": base_800.time_s,
            "candidate_observed_until_s": cand_800.time_s,
            "nonoverlap_s": max(0., cand_800.time_s - base_800.time_s),
            "new_hit_ids": sorted(extension_hits),
            "new_miss_ids": sorted(extension_misses),
            "new_keydowns": extension_downs,
            "cannot_compare_to_unobserved_baseline_future": extension,
        },
        "baseline_hit_ids_lost_by_800": final_lost_ids,
        "teacher_reference": {"hits": expert_800.hits, "misses": expert_800.misses,
                              "too_early": expert_800.too_early,
                              "overloaded": expert_800.overloaded},
        "old_teacher_gate_passed": teacher_gate_passed,
        "old_qualified": legacy_qualified,
        "eligible_for_teacher_review": eligible,
        "teacher_adopted": False,
    }


def rejected_solely_by_followup_teacher(old: dict[str, str]) -> bool:
    """All non-teacher legacy 800ms gates must pass, and category permitted."""
    if not old or old.get("short_status") in (
        "rejected_short", "safe_early_reduction_review_only", None
    ):
        return False
    if old.get("followup_expert_relative_hits") is None or \
            _saved_bool(old["followup_expert_relative_hits"]):
        return False
    other_gates = (
        "reached_horizon", "no_overload", "short_hits_retained",
        "no_hit_regression", "late_early_within_expert_plus_one",
        "late_misses_within_expert_plus_one",
    )
    return all(old.get("followup_"+key) is not None and
               _saved_bool(old["followup_"+key]) for key in other_gates)


def outcome_signature(anchor: int, candidate: Snapshot) -> str:
    """Identical motor outcomes have a single independent evidence pattern."""
    data = {
        "anchor": anchor,
        "hits": candidate.hit_ids,
        "missed": candidate.miss_ids,
        "too_early": candidate.too_early,
        "overloaded": candidate.overloaded,
        "overload_time_ms": (round(candidate.overload_time_s * 1000, 3)
                             if candidate.overload_time_s is not None else None),
        "keydowns": [(k, round(t * 1000, 3)) for k, t in candidate.keydowns],
        "end_ms": round(candidate.time_s * 1000, 3),
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def assert_output_safe(output: Path, input_dir: Path, source_checkpoint: Path) -> None:
    if output.exists():
        raise FileExistsError(f"v1.2 output path already exists: {output}")
    if output.resolve() == input_dir.resolve() or output.resolve() == source_checkpoint.resolve():
        raise ValueError("output would overlap v1.1 inputs or selected Best")
    if input_dir.resolve() in output.resolve().parents:
        raise ValueError("v1.2 output must be independent of read-only v1.1 files")


def csv_write_new(path: Path, rows: list[dict], *, columns: tuple[str, ...] = ()) -> None:
    if not rows and not columns:
        raise RuntimeError(f"empty diagnostic CSV without explicit schema: {path}")
    fields = list(dict.fromkeys([*columns, *(key for row in rows for key in row)]))
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (tuple, list, dict))
                             else v for k, v in row.items()})


def _clean_candidate_row(anchor, case, source, old, short, long, classified, signature):
    return {
        "anchor": anchor, "source": source,
        "kind": case.kind, "keys": ",".join(map(str, case.keys)),
        "start_frame": case.start, "duration_frames": case.duration,
        "delta": case.delta, "lead_frames": case.lead,
        "old_status": old.get("short_status") if old else None,
        "old_qualified": classified["old_qualified"],
        "old_teacher_gate_passed": classified["old_teacher_gate_passed"],
        "old_teacher_only_rejection": rejected_solely_by_followup_teacher(old),
        "short_hits": short.hits, "short_misses": short.misses,
        "short_early": short.too_early, "short_overload": short.overloaded,
        "short_duration_s": short.time_s,
        "hits_800": long.hits, "misses_800": long.misses,
        "early_800": long.too_early, "overload_800": long.overloaded,
        "overload_time_s": long.overload_time_s, "duration_s": long.time_s,
        "hit_ids_800": list(long.hit_ids), "miss_ids_800": list(long.miss_ids),
        "keydowns_800": list(long.keydowns),
        "category": classified["category"],
        "eligible_for_teacher_review": classified["eligible_for_teacher_review"],
        "teacher_adopted": False,
        "checks": classified["checks"],
        "comparison_common": classified["comparison_common"],
        "extended_observation": classified["extended_observation"],
        "teacher_reference": classified["teacher_reference"],
        "lost_baseline_hit_ids": classified["baseline_hit_ids_lost_by_800"],
        "outcome_signature": signature,
    }


def run_v12(prepared, *, previous: Path, output: Path, seed: int | None = None) -> dict:
    """No validation, optimizer, checkpoint writes, or automatic teacher adoption."""
    import torch
    from dmdod.motor.n_key import n_key_names
    from dmdod.training.fixed_state_action_audit import assert_state_equal, file_sha256
    from dmdod.training.real_chart import clone_model_state
    if len(prepared.anchors) != 20 or prepared.key_count != 8 or prepared.input_dim != 263:
        raise ValueError("requires Train20 / 8-key / 263D")
    if not abs(prepared.control_dt_s - .010) < 1e-12 or not abs(prepared.physics_dt_s - .001) < 1e-12:
        raise ValueError("v1.1 motor timesteps must remain 10ms/1ms")
    assert_output_safe(output, previous, prepared.source_checkpoint)
    source_sha = file_sha256(prepared.source_checkpoint)
    if source_sha != EXPECTED_BEST_SHA256:
        raise RuntimeError("selected Best SHA256 differs from required checkpoint")
    report_v11, selected = load_v11_short_passers(previous, n_key_names(8))
    model = prepared.model
    original = clone_model_state(model)
    old_hashes = {filename: file_sha256(previous/filename)
                  for filename in ("diagnostic.json", "train_candidate_gates.csv")}
    # No directory created before read-only preflight and checkpoint verification.
    output.mkdir(parents=True, exist_ok=False)
    baselines, cases, checks = [], [], []
    outputs = {}
    try:
        for anchor, named in enumerate(prepared.anchors, 1):
            student_400, student_800, student_frames = run_trace(prepared, named)
            expert_400, expert_800, expert_frames = run_trace(prepared, named, expert=True)
            assert_prefix_consistent(student_400, student_800)
            assert_prefix_consistent(expert_400, expert_800)
            outputs[anchor] = (student_400, student_800, student_frames, expert_800)
            baselines.append({
                "anchor": anchor, "chart": named.chart_name,
                "student_400": student_400.as_json(),
                "student_800": student_800.as_json(),
                "teacher_400": expert_400.as_json(),
                "teacher_800": expert_800.as_json(),
            })
            print(f"Train baseline #{anchor:02d} student {student_800.hits}H/"
                  f"{student_800.too_early}E @ {student_800.time_s:.3f}s; "
                  f"expert {expert_800.hits}H", flush=True)
        verify_v11_baselines(report_v11, selected, baselines)
        v1_manual = None
        from dmdod.training.train_counterfactual_motor_v11 import Probe
        explicit = Probe(keys=(n_key_names(8).index("right_3"),),
                         start=0, duration=4, delta=-.20, lead=0)
        if case_identity(20, explicit) not in {
                case_identity(item["anchor"], item["case"]) for item in selected}:
            v1_manual = {"anchor": 20, "case": explicit, "old": {},
                         "source": "v1_manual_unsampled"}
        for number, item in enumerate(selected + ([v1_manual] if v1_manual else []), 1):
            anchor, case, old = item["anchor"], item["case"], item["old"]
            named = prepared.anchors[anchor-1]
            short, long, frames = run_trace(prepared, named, case=case)
            base_400, base_800, base_frames, teacher_800 = outputs[anchor]
            assert_prefix_consistent(short, long)
            if old:
                assert_v11_reproduced(old, short, long)
            classified = classify_student_counterfactual(
                base_400, base_800, base_frames, short, long, frames,
                expert_800=teacher_800, old_row=old,
            )
            signature = outcome_signature(anchor, long)
            cases.append(_clean_candidate_row(anchor,case,item["source"],old,
                                             short,long,classified,signature))
            checks.append({
                "anchor": anchor, "source": item["source"],
                "case": case_identity(anchor,case), "400ms": short.as_json(),
                "800ms": long.as_json(), "reproduced": bool(old),
            })
            print(f"Train candidate {number}/{len(selected)+(v1_manual is not None)} "
                  f"#{anchor:02d} {case.kind} {case.delta:+.2f} "
                  f"=> {classified['category']} "
                  f"{'REPRODUCED' if old else 'EXTRA v1'}",flush=True)
        if sum(x["reproduced"] for x in checks) != EXPECTED_V11_SHORT_PASSERS:
            raise RuntimeError("v1.1 reproduction missing candidates")
        if file_sha256(prepared.source_checkpoint) != source_sha:
            raise RuntimeError("Best checkpoint changed during v1.2")
        if any(file_sha256(previous/k) != v for k, v in old_hashes.items()):
            raise RuntimeError("v1.1 diagnostic inputs changed during v1.2")
        assert_state_equal(clone_model_state(model), original,
                           context="v1.2 Student must remain unchanged")
        verified = [x for x in cases if x["source"] == "v11_short_pass"]
        improvements = [x for x in verified if x["eligible_for_teacher_review"]]
        category_counts = {category: sum(x["category"]==category for x in verified)
                           for category in ("overload_rescue", "early_reduction_preserved_hits",
                                            "hit_gain", "deferred_failure", "hit_suppression",
                                            "regression", "inconclusive")}
        teacher_only = [x for x in improvements if x["old_teacher_only_rejection"]]
        teacher_gate_overlap = [
            x for x in improvements if x["old_teacher_gate_passed"] is False
            and x["old_qualified"] is False
        ]
        category_only = [x for x in improvements if x["old_status"] ==
                         "safe_early_reduction_review_only" and x["old_qualified"] is False]
        unique_patterns = {(r["anchor"],r["outcome_signature"]) for r in improvements}
        result = {
            "kind": "train_counterfactual_motor_v12",
            "source_checkpoint_sha256": source_sha,
            "source_training_seed": seed,
            "source_diagnostic_hashes": old_hashes,
            "train_anchors": 20, "validation_evaluations": 0,
            "final_used": False, "optimizer_updates": 0,
            "checkpoint_writes": 0, "teacher_adoptions": 0,
            "v11_candidates_selected": len(selected),
            "v11_candidates_reproduced": EXPECTED_V11_SHORT_PASSERS,
            "v11_reproduction_rate": 1.0,
            "student_relative_improvement_count": len(improvements),
            "student_relative_improved_anchors": sorted({x["anchor"] for x in improvements}),
            "overload_rescue_count": category_counts["overload_rescue"],
            "hit_preserved_early_reduction_count": category_counts["early_reduction_preserved_hits"],
            "hit_suppression_count": category_counts["hit_suppression"],
            "deferred_failure_count": category_counts["deferred_failure"],
            "teacher_only_gate_rejection_count": len(teacher_only),
            "teacher_gate_rejected_with_other_possible_reasons": len(teacher_gate_overlap),
            "previous_category_review_only_but_student_valid": len(category_only),
            "category_counts": category_counts,
            "unique_improvement_patterns": len(unique_patterns),
            "duplicate_improvement_candidates": len(improvements)-len(unique_patterns),
            "v1_train20_explicit_replay": (v1_manual is not None),
            "focus_anchors": [
                {"anchor": n, "baseline": baselines[n-1],
                 "cases": [x for x in cases if x["anchor"] == n]}
                for n in FOCUS_ANCHORS
            ],
            "limitations": [
                "Student uses 263D visible input, while teacher receives privileged state only outside Student.",
                "When the unmodified Student terminates early, the later reference outcome is unobserved.",
                "Interventions are evaluator-origin Train counterfactuals, NOT deployed policies.",
                "Eligibility for manual teacher review never adopts or trains a teacher.",
                "Counting identical outcomes separately can overstate independent improvement patterns.",
            ],
        }
        # Publish files only after all 151 original cases reproduced successfully.
        csv_write_new(output/"baseline_800ms.csv", baselines)
        csv_write_new(output/"candidate_comparison.csv", cases)
        csv_write_new(output/"qualified_corrections.csv", improvements,
                      columns=tuple(cases[0].keys()))
        (output/"reproduction_checks.json").write_text(
            json.dumps({"status":"PASS", "source_sha256":source_sha,
                        "v11_input_sha256":old_hashes,"candidates":checks},
                       ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8"
        )
        (output/"diagnostic.json").write_text(
            json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8"
        )
        return result
    except Exception:
        # On FAIL no partially qualified teacher is ever output; marker makes it
        # impossible to confuse a failed run with a completed diagnosis.
        (output/"FAILED_NO_QUALIFICATION.txt").write_text(
            "This v1.2 run failed closed; do not adopt any candidate.\n",encoding="utf-8"
        )
        raise
    finally:
        model.load_state_dict(original, strict=True)
        model.prepare_recurrent_runtime()
        assert_state_equal(clone_model_state(model),original,
                           context="v1.2 restored Best after diagnostics")
