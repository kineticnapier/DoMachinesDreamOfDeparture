from __future__ import annotations

"""v1.1: make clean chart starts a bootstrap selection priority.

v1.0 adds short start-micro expert/guard segments, but the ordinary bootstrap
key still aggregates their few targets with thousands of normal anchor targets.
That can preserve a model which physically presses during the countdown.  v1.1
keeps v1.0 training/gameplay semantics and changes bootstrap model selection so
start-micro TooEarly presses are considered immediately after the existing
absolute overload-safety criterion and before aggregate completion/accuracy.

A completely clean start (zero TooEarly presses across all start-micro anchors)
is the next hard priority after safety.  While no clean safe candidate exists,
fewer start-micro TooEarly presses is the fallback ordering so bootstrap can
still make monotonic progress toward the gate instead of silently retaining a
worse warm start.
"""

import sys

import torch

import train_real_chart_v080 as v080
import train_real_chart_v080_fast as v080_fast
import train_real_chart_v090 as v090
import train_real_chart_v090_fast as v090_fast
import train_real_chart_v090_turbo as turbo
import train_real_chart_v100_start_micro as v100


TRAINER_VERSION = "1.1.0-start-gate"
DEFAULT_CHECKPOINT = "checkpoints/real_chart_v110_start_gate.pt"
START_GATE_VERSION = "v110-bootstrap-start-micro-early-gate-v2"

_BASE_BOOTSTRAP_KEY = v080._bootstrap_key
_BASE_BOOTSTRAP_PRUNE_REASON = turbo._bootstrap_prune_reason


def _is_start_micro(named) -> bool:
    return str(getattr(named, "role", "")).startswith("start-micro-")


def _start_micro_early(anchors) -> int:
    captured = list(turbo._CAPTURED_ANCHORS)
    if len(captured) != len(anchors):
        raise RuntimeError(
            "start-gate anchor/evaluation count mismatch: "
            f"anchors={len(captured)} evals={len(anchors)}"
        )
    return sum(
        int(evaluation.stats.too_early_presses)
        for named, evaluation in zip(captured, anchors)
        if _is_start_micro(named)
    )


def _bootstrap_key_start_gate(anchors, validations) -> tuple:
    """Rank safety, then clean starts, then the legacy aggregate metrics."""

    start_early = _start_micro_early(anchors)
    base = _BASE_BOOTSTRAP_KEY(anchors, validations)
    if len(base) < 2:
        raise RuntimeError("legacy bootstrap key is unexpectedly short")
    # Legacy: (safety, completion, meanX, meanPP, -early)
    # v1.1:   (safety, start-clean, -start-early, completion, meanX, meanPP, -early)
    return base[0], int(start_early == 0), -int(start_early), *base[1:]


def _base_key(gated_key: tuple) -> tuple:
    if len(gated_key) < 4:
        raise ValueError("v1.1 bootstrap key is missing the legacy suffix")
    return gated_key[0], *gated_key[3:]


def _key_fields(gated_key: tuple) -> tuple[bool, bool, int, tuple]:
    base = _base_key(gated_key)
    return bool(base[0]), bool(gated_key[1]), -int(gated_key[2]), base


def _bootstrap_prune_reason_start_gate(
    best_key: tuple,
    *,
    any_overloaded: bool,
    partial_hits: int,
    evaluated_targets: int,
    total_targets: int,
) -> str | None:
    """Keep optimistic pruning exact under safety -> start-clean ordering.

    Safety remains absolute and can always prune an already-overloaded candidate
    against a safe best.  If safety can still tie but the best is start-dirty,
    aggregate completion pruning is not sound: a candidate with worse aggregate
    completion could still win by becoming start-clean.  Once the best is both
    equally safe and start-clean, the legacy safety/completion upper bound is
    sound again.
    """

    best_safety = int(best_key[0])
    safety_upper = 0 if any_overloaded else 1
    if safety_upper < best_safety:
        return "safety"
    if safety_upper > best_safety:
        return None
    if not bool(best_key[1]):
        return None
    return _BASE_BOOTSTRAP_PRUNE_REASON(
        _base_key(best_key),
        any_overloaded=any_overloaded,
        partial_hits=partial_hits,
        evaluated_targets=evaluated_targets,
        total_targets=total_targets,
    )


def _bootstrap_partial_prune_start_gate(
    best_key: tuple,
    *,
    combined,
    results,
    any_overloaded: bool,
) -> str | None:
    """Prune as soon as accumulated start-micro Early can no longer beat best.

    TooEarly counts are monotone as more start-micro segments are evaluated.
    Once the partial count exceeds the best candidate's complete start-micro
    count, no unevaluated segment can recover that ordering.  Safety still has
    absolute priority: if the candidate can still finish safer than the current
    best, the start gate is not allowed to prune it.
    """

    best_safety = int(best_key[0])
    safety_upper = 0 if any_overloaded else 1
    if safety_upper != best_safety:
        return None

    partial_start_early = sum(
        int(evaluation.stats.too_early_presses)
        for index, evaluation in results.items()
        if _is_start_micro(combined[index])
    )
    best_start_early = -int(best_key[2])
    if partial_start_early <= best_start_early:
        return None
    return "start-dirty" if best_start_early == 0 else "start-early"


def _turbo_bootstrap_start_gate(
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
    """Turbo bootstrap with v1.1 safety -> start-micro -> legacy ordering."""

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

    safe, clean, start_early, base = _key_fields(best_key)
    print(
        f"bootstrap 00: safe={safe} start-clean={clean} start-early={start_early} "
        f"completion={base[1] * 100.0:.1f}% meanX={base[2]:.1f}%"
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
        anchors, validations, key, pruned = turbo._evaluate_bootstrap_candidate(
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
            turbo._BOOTSTRAP_PRUNES += 1

        if keep or epoch == 1 or epoch == epochs or epoch % 4 == 0:
            if pruned is None:
                assert key is not None
                safe, clean, start_early, base = _key_fields(key)
                print(
                    f"bootstrap {epoch:02d}: loss={loss:.6f} safe={safe} "
                    f"start-clean={clean} start-early={start_early} "
                    f"completion={base[1] * 100.0:.1f}% meanX={base[2]:.1f}%"
                    + (" KEEP" if keep else "")
                )
            else:
                print(
                    f"bootstrap {epoch:02d}: loss={loss:.6f} PRUNE[{pruned['reason']}] "
                    f"eval={pruned['evaluated']}/{pruned['total']}"
                )

        if pruned is None:
            assert key is not None
            safe, clean, start_early, base = _key_fields(key)
        else:
            safe = False
            clean = False
            start_early = -1
            base = ()
        history.append(
            {
                "epoch": epoch,
                "loss": loss,
                "safe": None if pruned is not None else safe,
                "start_clean": None if pruned is not None else clean,
                "start_early": None if pruned is not None else start_early,
                "completion": None if pruned is not None else base[1],
                "mean_xacc": None if pruned is not None else base[2],
                "kept": keep,
                "pruned": None if pruned is None else pruned["reason"],
                "evaluated_segments": len(anchor_segments) + len(validation_segments)
                if pruned is None
                else pruned["evaluated"],
            }
        )

    model.load_state_dict(best_state)
    safe, clean, start_early, base = _key_fields(best_key)
    print(
        f"bootstrap selected: epoch={best_epoch} safe={safe} start-clean={clean} "
        f"start-early={start_early} completion={base[1] * 100.0:.1f}% "
        f"meanX={base[2]:.1f}% pruned={turbo._BOOTSTRAP_PRUNES}/{epochs}"
    )
    return best_state, best_anchors, best_validations, history


def install_start_gate() -> None:
    v080._bootstrap_key = _bootstrap_key_start_gate
    turbo._bootstrap_prune_reason = _bootstrap_prune_reason_start_gate
    turbo._BOOTSTRAP_PARTIAL_PRUNE_HOOK = _bootstrap_partial_prune_start_gate
    turbo._turbo_bootstrap = _turbo_bootstrap_start_gate


def main() -> None:
    v090._configure_console_output()
    start_args, after_v100 = v100._consume_v100_args(sys.argv[1:])
    press_args, remaining = v090._consume_v090_args(after_v100)

    v090_fast._install_v090_fast_path(
        coef=press_args.press_persistence_coef,
        lookahead_frames=press_args.press_persistence_lookahead,
        commit_threshold=press_args.press_commit_threshold,
        hold_margin=press_args.press_hold_margin,
    )
    v100.install_start_micro(target_count=start_args.start_micro_targets)
    v080.TRAINER_VERSION = TRAINER_VERSION
    v080.DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT
    if start_args.warm_start is not None:
        v100.install_warm_start(start_args.warm_start)
    install_start_gate()
    turbo._install_turbo_path()

    print("=== DMDOD v1.1.0 Start Gate ===")
    print(
        f"start-gate={START_GATE_VERSION} | start-micro targets={start_args.start_micro_targets} | "
        "bootstrap priority=safety -> start-clean -> fewer-start-early -> legacy-quality"
    )
    print(
        f"press-persistence coef={press_args.press_persistence_coef:g} "
        f"lookahead={press_args.press_persistence_lookahead}f "
        f"commit>={press_args.press_commit_threshold:+.2f} "
        f"hold>={press_args.press_hold_margin:+.2f}"
    )
    print(
        f"turbo={turbo.TURBO_VERSION} | fast-eval={v080_fast.FAST_EVAL_VERSION} | "
        "start-micro=train+guard bootstrap-hard-priority+gate-prune"
    )

    sys.argv = [sys.argv[0], *remaining]
    v080.main()
    print(
        f"turbo-stats: teacher-cache-hit={turbo._TEACHER_CACHE_HITS} "
        f"teacher-generated={turbo._TEACHER_CACHE_MISSES} "
        f"bootstrap-prunes={turbo._BOOTSTRAP_PRUNES}"
    )


if __name__ == "__main__":
    main()
