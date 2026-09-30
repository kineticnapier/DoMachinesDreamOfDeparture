from __future__ import annotations

"""Fine-grained bootstrap evaluation progress for the v1.0 modern trainer.

The turbo bootstrap evaluator already processes a candidate in worker-sized
waves so it can prune hopeless candidates exactly.  This module keeps that
algorithm unchanged and only emits progress events after each completed wave.
"""

import math

import train_real_chart_v090_turbo as turbo
from dmdod.training_progress import emit_progress


BOOTSTRAP_PROGRESS_VERSION = "v100-bootstrap-wave-progress-v1"
_INSTALLED = False


def _wave_phase(batch) -> str:
    roles = [str(getattr(named, "role", "")) for named in batch]
    is_start = [role.startswith("start-micro-") for role in roles]
    if roles and all(is_start):
        return "start-micro"
    if any(is_start):
        return "mixed"
    if roles and all(role == "validation" for role in roles):
        return "validation"
    if roles and all(role.startswith("anchor-") for role in roles):
        return "anchors"
    return "anchors+validation"


def _evaluate_bootstrap_candidate_with_progress(
    model,
    state,
    anchor_segments,
    validation_segments,
    *,
    best_key: tuple,
    same_hand: bool,
    control_dt_s: float,
):
    """Turbo candidate evaluation with observational wave progress events."""

    combined = [*anchor_segments, *validation_segments]
    if not combined:
        emit_progress(
            "bootstrap_eval_start",
            total=0,
            waves=0,
            start_micro=0,
        )
        emit_progress(
            "bootstrap_eval_done",
            current=0,
            total=0,
            status="FULL",
            reason=None,
        )
        return [], [], None, None

    total_targets = sum(len(named.segment.targets) for named in combined)
    order = turbo._bootstrap_segment_order(combined)
    batch_size = max(1, min(turbo.v080._configured_workers(), len(combined)))
    wave_total = math.ceil(len(order) / batch_size)
    start_micro_total = sum(
        1
        for named in combined
        if str(getattr(named, "role", "")).startswith("start-micro-")
    )
    emit_progress(
        "bootstrap_eval_start",
        total=len(combined),
        waves=wave_total,
        start_micro=start_micro_total,
        batch_size=batch_size,
    )

    results: dict[int, object] = {}
    partial_hits = 0
    evaluated_targets = 0
    any_overloaded = False
    marker = 0.0

    for wave_index, start in enumerate(range(0, len(order), batch_size), 1):
        indices = order[start : start + batch_size]
        batch = [combined[index] for index in indices]
        raw = turbo.v080._evaluate_states_on_segments(
            model,
            {marker: state},
            batch,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
        )
        for index, named in zip(indices, batch):
            evaluation = raw[(marker, named.key)]
            results[index] = evaluation
            partial_hits += int(evaluation.stats.hits)
            evaluated_targets += int(evaluation.stats.targets)
            if evaluation.stats.overloaded:
                any_overloaded = True
                turbo._BOOTSTRAP_FAILURE_COUNTS[named.key] = (
                    turbo._BOOTSTRAP_FAILURE_COUNTS.get(named.key, 0) + 1
                )

        phase = _wave_phase(batch)
        emit_progress(
            "bootstrap_eval_step",
            current=len(results),
            total=len(combined),
            wave=wave_index,
            waves=wave_total,
            phase=phase,
            overloaded=any_overloaded,
            partial_hits=partial_hits,
            evaluated_targets=evaluated_targets,
            total_targets=total_targets,
        )

        reason = turbo._bootstrap_prune_reason(
            best_key,
            any_overloaded=any_overloaded,
            partial_hits=partial_hits,
            evaluated_targets=evaluated_targets,
            total_targets=total_targets,
        )
        if reason is None:
            reason = turbo._bootstrap_partial_prune_reason(
                best_key,
                combined=combined,
                results=results,
                any_overloaded=any_overloaded,
            )
        if reason is not None and len(results) < len(combined):
            emit_progress(
                "bootstrap_eval_done",
                current=len(results),
                total=len(combined),
                status="PRUNE",
                reason=reason,
                wave=wave_index,
                waves=wave_total,
            )
            return None, None, None, {
                "reason": reason,
                "evaluated": len(results),
                "total": len(combined),
                "any_overloaded": any_overloaded,
                "partial_hits": partial_hits,
                "evaluated_targets": evaluated_targets,
                "total_targets": total_targets,
            }

    canonical = [results[index] for index in range(len(combined))]
    split = len(anchor_segments)
    anchors = canonical[:split]
    validations = canonical[split:]
    key = turbo.v080._bootstrap_key(anchors, validations)
    emit_progress(
        "bootstrap_eval_done",
        current=len(combined),
        total=len(combined),
        status="FULL",
        reason=None,
        wave=wave_total,
        waves=wave_total,
    )
    return anchors, validations, key, None


def install_bootstrap_progress() -> None:
    """Install observational progress without changing bootstrap decisions."""

    global _INSTALLED
    if _INSTALLED:
        return
    turbo._evaluate_bootstrap_candidate = _evaluate_bootstrap_candidate_with_progress
    _INSTALLED = True
    print(
        f"bootstrap-progress={BOOTSTRAP_PROGRESS_VERSION} "
        "display=segment-wave prune-status"
    )
