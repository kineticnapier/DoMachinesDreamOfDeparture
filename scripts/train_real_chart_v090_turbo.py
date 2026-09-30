from __future__ import annotations

"""v0.9 press-persistence trainer with teacher/bootstrap acceleration.

This wrapper composes v0.9 press persistence, v0.8's fast evaluator, parallel
anchor-teacher generation with a deterministic disk cache, and exact bootstrap
candidate pruning.  Training loss, trust alphas, gameplay guards, model updates,
and final selection remain unchanged.  Bootstrap candidates are only pruned once
an optimistic upper bound proves that they cannot outrank the current best.
"""

import hashlib
import os
import sys
from pathlib import Path

import torch

import train_real_chart_v080 as v080
import train_real_chart_v080_fast as v080_fast
import train_real_chart_v090 as v090
import train_real_chart_v090_fast as v090_fast


TURBO_VERSION = "v090-teacher-cache-bootstrap-prune-v1"
TEACHER_CACHE_VERSION = "v070-hud-finger-agnostic-teacher-v1"
DEFAULT_TEACHER_CACHE_DIR = Path("checkpoints/.teacher-cache")

_ORIGINAL_BUILD_ANCHOR_SEGMENTS = v080._build_anchor_segments
_ORIGINAL_CALIBRATE = v080.calibrate_single_press_lead
_ORIGINAL_COLLECT_EXPERT = v080.v060._collect_expert_sequence

_CAPTURED_ANCHORS = []
_TEACHER_ENTRIES: dict[int, tuple[Path, object | None]] = {}
_TEACHER_CACHE_HITS = 0
_TEACHER_CACHE_MISSES = 0
_BOOTSTRAP_FAILURE_COUNTS: dict[tuple, int] = {}
_BOOTSTRAP_PRUNES = 0


def _teacher_cache_dir() -> Path:
    raw = os.environ.get("DMDOD_TEACHER_CACHE_DIR")
    return Path(raw) if raw else DEFAULT_TEACHER_CACHE_DIR


def _teacher_cache_path(named, *, lead_s: float, same_hand: bool, control_dt_s: float) -> Path:
    material = "|".join(
        (
            TEACHER_CACHE_VERSION,
            str(v080.v070.HUD_OBSERVATION_VERSION),
            named.chart_sha256,
            f"{named.start_s:.9f}",
            f"{named.end_s:.9f}",
            f"{lead_s:.9f}",
            "1" if same_hand else "0",
            f"{control_dt_s:.9f}",
        )
    ).encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return _teacher_cache_dir() / f"{digest}.pt"


def _save_teacher_cache(path: Path, x, y, evaluation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": TEACHER_CACHE_VERSION,
        "hud_observation": str(v080.v070.HUD_OBSERVATION_VERSION),
        "x": x.detach().cpu(),
        "y": y.detach().cpu(),
        "evaluation": v080.v063._eval_to_payload(evaluation),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_teacher_cache(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("version") != TEACHER_CACHE_VERSION:
        raise ValueError("teacher cache version mismatch")
    if payload.get("hud_observation") != str(v080.v070.HUD_OBSERVATION_VERSION):
        raise ValueError("teacher cache HUD version mismatch")
    evaluation = v080.v063._eval_from_payload(payload["evaluation"])
    return payload["x"].to(device), payload["y"].to(device), evaluation


def _teacher_worker(segment, lead_s: float, same_hand: bool, control_dt_s: float):
    """Collect one HUD expert sequence inside a spawned worker process."""

    # Windows ProcessPool uses spawn.  Re-install v0.7's HUD globals explicitly
    # so workers produce the same 245D observation that the parent trainer uses.
    import train_real_chart_v060 as v060
    import train_real_chart_v070 as v070

    v070._install_v070()
    return v060._collect_expert_sequence(
        segment,
        lead_s=float(lead_s),
        same_hand=bool(same_hand),
        control_dt_s=float(control_dt_s),
        device=torch.device("cpu"),
    )


def _capture_anchor_segments(*args, **kwargs):
    global _CAPTURED_ANCHORS
    result = _ORIGINAL_BUILD_ANCHOR_SEGMENTS(*args, **kwargs)
    _CAPTURED_ANCHORS = list(result)
    return result


def _calibrate_and_prefetch(*args, **kwargs):
    global _TEACHER_CACHE_HITS, _TEACHER_CACHE_MISSES

    calibration = _ORIGINAL_CALIBRATE(*args, **kwargs)
    control_dt_s = float(kwargs.get("control_dt_s", args[0] if args else 0.010))
    same_hand = bool(kwargs.get("same_hand", True))
    workers = v080._configured_workers()
    pool = v080._get_pool() if workers > 1 and _CAPTURED_ANCHORS else None

    hits = 0
    misses = 0
    _TEACHER_ENTRIES.clear()
    for named in _CAPTURED_ANCHORS:
        path = _teacher_cache_path(
            named,
            lead_s=calibration.lead_s,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
        )
        if path.exists():
            _TEACHER_ENTRIES[id(named.segment)] = (path, None)
            hits += 1
            continue

        future = None
        if pool is not None:
            future = pool.submit(
                _teacher_worker,
                named.segment,
                float(calibration.lead_s),
                bool(same_hand),
                float(control_dt_s),
            )
        _TEACHER_ENTRIES[id(named.segment)] = (path, future)
        misses += 1

    _TEACHER_CACHE_HITS += hits
    _TEACHER_CACHE_MISSES += misses
    if _CAPTURED_ANCHORS:
        print(
            f"teacher-turbo: anchors={len(_CAPTURED_ANCHORS)} cache-hit={hits} "
            f"generate={misses} workers={workers} dir={_teacher_cache_dir()}"
        )
    return calibration


def _cached_collect_expert_sequence(
    segment,
    *,
    lead_s: float,
    same_hand: bool,
    control_dt_s: float,
    device: torch.device,
):
    entry = _TEACHER_ENTRIES.get(id(segment))
    if entry is None:
        return _ORIGINAL_COLLECT_EXPERT(
            segment,
            lead_s=lead_s,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
            device=device,
        )

    path, future = entry
    if path.exists():
        try:
            return _load_teacher_cache(path, device)
        except Exception as exc:
            print(f"teacher-cache ignore {path.name} ({type(exc).__name__}: {exc})")

    try:
        if future is not None:
            x, y, evaluation = future.result()
        else:
            x, y, evaluation = _ORIGINAL_COLLECT_EXPERT(
                segment,
                lead_s=lead_s,
                same_hand=same_hand,
                control_dt_s=control_dt_s,
                device=torch.device("cpu"),
            )
        _save_teacher_cache(path, x, y, evaluation)
        return x.to(device), y.to(device), evaluation
    except Exception as exc:
        # Turbo is execution-only.  A worker/cache failure falls back to the
        # original serial collector rather than changing training behavior.
        print(f"teacher-turbo fallback ({type(exc).__name__}: {exc})")
        return _ORIGINAL_COLLECT_EXPERT(
            segment,
            lead_s=lead_s,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
            device=device,
        )


def _bootstrap_prune_reason(
    best_key: tuple,
    *,
    any_overloaded: bool,
    partial_hits: int,
    evaluated_targets: int,
    total_targets: int,
) -> str | None:
    """Return why a partial bootstrap candidate provably cannot beat best."""

    if total_targets <= 0:
        return None
    safety_upper = 0 if any_overloaded else 1
    best_safety = int(best_key[0])
    if safety_upper < best_safety:
        return "safety"
    if safety_upper > best_safety:
        return None

    remaining_targets = max(0, int(total_targets) - int(evaluated_targets))
    completion_upper = (int(partial_hits) + remaining_targets) / max(1, int(total_targets))
    if completion_upper < float(best_key[1]) - 1e-12:
        return "completion"
    return None


def _bootstrap_segment_order(segments) -> list[int]:
    return sorted(
        range(len(segments)),
        key=lambda index: (-_BOOTSTRAP_FAILURE_COUNTS.get(segments[index].key, 0), index),
    )


def _evaluate_bootstrap_candidate(
    model,
    state,
    anchor_segments,
    validation_segments,
    *,
    best_key: tuple,
    same_hand: bool,
    control_dt_s: float,
):
    """Evaluate one bootstrap state in worker-sized waves with exact pruning."""

    combined = [*anchor_segments, *validation_segments]
    if not combined:
        return [], [], None, None

    total_targets = sum(len(named.segment.targets) for named in combined)
    order = _bootstrap_segment_order(combined)
    batch_size = max(1, min(v080._configured_workers(), len(combined)))
    results: dict[int, object] = {}
    partial_hits = 0
    evaluated_targets = 0
    any_overloaded = False
    marker = 0.0

    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        batch = [combined[index] for index in indices]
        raw = v080._evaluate_states_on_segments(
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
                _BOOTSTRAP_FAILURE_COUNTS[named.key] = (
                    _BOOTSTRAP_FAILURE_COUNTS.get(named.key, 0) + 1
                )

        reason = _bootstrap_prune_reason(
            best_key,
            any_overloaded=any_overloaded,
            partial_hits=partial_hits,
            evaluated_targets=evaluated_targets,
            total_targets=total_targets,
        )
        if reason is not None and len(results) < len(combined):
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
    key = v080._bootstrap_key(anchors, validations)
    return anchors, validations, key, None


def _turbo_bootstrap(
    model,
    expert_pairs,
    anchor_segments,
    validation_segments,
    *,
    epochs: int,
    learning_rate: float,
    chunk_steps: int,
    same_hand: bool,
    control_dt_s: float,
):
    """v0.8 bootstrap with exact optimistic-bound gameplay pruning."""

    global _BOOTSTRAP_PRUNES
    optimizer = torch.optim.Adam(v080.v057._policy_parameters(model), lr=learning_rate)

    state = {key: value.clone() for key, value in model.state_dict().items()}
    anchor_evals, validation_evals = v080_fast._evaluate_bootstrap_state(
        model,
        state,
        anchor_segments,
        validation_segments,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
    )
    best_state = state
    best_anchors = anchor_evals
    best_validations = validation_evals
    best_key = v080._bootstrap_key(anchor_evals, validation_evals)
    best_epoch = 0
    history: list[dict] = []
    print(
        f"bootstrap 00: safe={bool(best_key[0])} completion={best_key[1] * 100.0:.1f}% "
        f"meanX={best_key[2]:.1f}%"
    )

    for epoch in range(1, epochs + 1):
        loss = v080.v062._bootstrap_multi_epoch(
            model,
            expert_pairs,
            optimizer=optimizer,
            chunk_steps=chunk_steps,
            reverse_order=not bool(epoch & 1),
        )
        state = {key: value.clone() for key, value in model.state_dict().items()}
        anchors, validations, key, pruned = _evaluate_bootstrap_candidate(
            model,
            state,
            anchor_segments,
            validation_segments,
            best_key=best_key,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
        )

        keep = False
        if pruned is None:
            assert key is not None and anchors is not None and validations is not None
            keep = key > best_key
            if keep:
                best_key = key
                best_state = state
                best_anchors = anchors
                best_validations = validations
                best_epoch = epoch
        else:
            _BOOTSTRAP_PRUNES += 1

        if keep or epoch == 1 or epoch == epochs or epoch % 4 == 0:
            if pruned is None:
                assert key is not None
                print(
                    f"bootstrap {epoch:02d}: loss={loss:.6f} safe={bool(key[0])} "
                    f"completion={key[1] * 100.0:.1f}% meanX={key[2]:.1f}%"
                    + (" KEEP" if keep else "")
                )
            else:
                print(
                    f"bootstrap {epoch:02d}: loss={loss:.6f} PRUNE[{pruned['reason']}] "
                    f"eval={pruned['evaluated']}/{pruned['total']}"
                )

        history.append(
            {
                "epoch": epoch,
                "loss": loss,
                "safe": None if pruned is not None else bool(key[0]),
                "completion": None if pruned is not None else key[1],
                "mean_xacc": None if pruned is not None else key[2],
                "kept": keep,
                "pruned": None if pruned is None else pruned["reason"],
                "evaluated_segments": len(anchor_segments) + len(validation_segments)
                if pruned is None
                else pruned["evaluated"],
            }
        )

    model.load_state_dict(best_state)
    print(
        f"bootstrap selected: epoch={best_epoch} safe={bool(best_key[0])} "
        f"completion={best_key[1] * 100.0:.1f}% meanX={best_key[2]:.1f}% "
        f"pruned={_BOOTSTRAP_PRUNES}/{epochs}"
    )
    return best_state, best_anchors, best_validations, history


def _install_turbo_path() -> None:
    v080._build_anchor_segments = _capture_anchor_segments
    v080.calibrate_single_press_lead = _calibrate_and_prefetch
    v080.v060._collect_expert_sequence = _cached_collect_expert_sequence
    v080._bootstrap = _turbo_bootstrap


def main() -> None:
    v090._configure_console_output()
    args, remaining = v090._consume_v090_args(sys.argv[1:])
    v090_fast._install_v090_fast_path(
        coef=args.press_persistence_coef,
        lookahead_frames=args.press_persistence_lookahead,
        commit_threshold=args.press_commit_threshold,
        hold_margin=args.press_hold_margin,
    )
    _install_turbo_path()

    print("=== DMDOD v0.9.0 Turbo ===")
    print(
        "press-persistence "
        f"coef={args.press_persistence_coef:g} "
        f"lookahead={args.press_persistence_lookahead}f "
        f"commit>={args.press_commit_threshold:+.2f} "
        f"hold>={args.press_hold_margin:+.2f}"
    )
    print(
        f"turbo={TURBO_VERSION} | fast-eval={v080_fast.FAST_EVAL_VERSION} | "
        "teacher=parallel+disk-cache | bootstrap=exact-upper-bound-prune"
    )
    print("checkpoint/signature/model/round-selection semantics=v0.9.0 unchanged")

    sys.argv = [sys.argv[0], *remaining]
    v080.main()
    print(
        f"turbo-stats: teacher-cache-hit={_TEACHER_CACHE_HITS} "
        f"teacher-generated={_TEACHER_CACHE_MISSES} bootstrap-prunes={_BOOTSTRAP_PRUNES}"
    )


if __name__ == "__main__":
    main()
