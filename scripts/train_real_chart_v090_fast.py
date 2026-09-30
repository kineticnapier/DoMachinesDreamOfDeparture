from __future__ import annotations

"""v0.9 press-persistence trainer with execution-only fast evaluation.

This composes the existing v0.9 loss with v0.8's resume-compatible fast
evaluation scheduler. Checkpoint format, trainer signature, model updates, trust
alphas, guards, candidate ranking, and Final split semantics remain unchanged.
"""

import sys

import train_real_chart_v080 as v080
import train_real_chart_v080_fast as v080_fast
import train_real_chart_v090 as v090


def _install_v090_fast_path(
    *,
    coef: float,
    lookahead_frames: int,
    commit_threshold: float,
    hold_margin: float,
) -> None:
    # Evaluation scheduling must be installed first. The v0.9 hook only changes
    # the BC actuation loss and trainer/checkpoint identity, so the two patches
    # are orthogonal and existing v0.9 checkpoints remain resume-compatible.
    v080_fast._install_fast_path()
    v090._install_press_persistence(
        coef=coef,
        lookahead_frames=lookahead_frames,
        commit_threshold=commit_threshold,
        hold_margin=hold_margin,
    )


def main() -> None:
    v090._configure_console_output()
    args, remaining = v090._consume_v090_args(sys.argv[1:])
    _install_v090_fast_path(
        coef=args.press_persistence_coef,
        lookahead_frames=args.press_persistence_lookahead,
        commit_threshold=args.press_commit_threshold,
        hold_margin=args.press_hold_margin,
    )

    print("=== DMDOD v0.9.0 Press Persistence + Fast Eval ===")
    print(
        "press-persistence "
        f"coef={args.press_persistence_coef:g} "
        f"lookahead={args.press_persistence_lookahead}f "
        f"commit>={args.press_commit_threshold:+.2f} "
        f"hold>={args.press_hold_margin:+.2f}"
    )
    print(
        f"fast-eval={v080_fast.FAST_EVAL_VERSION} | "
        "validation=adaptive-short-circuit | "
        f"anchor-batch={v080_fast.DEFAULT_ANCHOR_BATCH_SIZE}"
    )

    sys.argv = [sys.argv[0], *remaining]
    v080.main()


if __name__ == "__main__":
    main()
