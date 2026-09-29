from __future__ import annotations

"""v0.9.0: v0.8 multi-chart HUD training with press-persistence loss.

This keeps the v0.8 data split, HUD observation, evaluator, and conservative
selection guards unchanged.  Only the behavior-cloning loss is augmented so a
student that has started moving a finger toward an imminent teacher press is
penalized for immediately cancelling that press before physical KeyDown.
"""

import argparse
import sys

import torch

import train_real_chart_v054 as v054
import train_real_chart_v080 as v080


TRAINER_VERSION = "0.9.0-press-persistence"
DEFAULT_CHECKPOINT = "checkpoints/real_chart_v090_press_persistence.pt"
DEFAULT_PERSISTENCE_COEF = 6.0
DEFAULT_LOOKAHEAD_FRAMES = 4
DEFAULT_COMMIT_THRESHOLD = 0.30
DEFAULT_HOLD_MARGIN = 0.20

_ORIGINAL_ACTUATION_LOSS = v054._actuation_loss


def _configure_text_stream(stream) -> None:
    """Keep redirected Windows output alive when chart names exceed CP932.

    PowerShell pipelines can make Python choose CP932 for stdout/stderr.  TUF
    chart names legitimately contain Korean and other characters outside that
    code page.  Preserve the active encoding but escape only unencodable
    characters instead of aborting a long training run with UnicodeEncodeError.
    """

    reconfigure = getattr(stream, "reconfigure", None)
    if not callable(reconfigure):
        return
    try:
        reconfigure(errors="backslashreplace")
    except (OSError, ValueError):
        # Some wrapped/test streams cannot be reconfigured after I/O.  In that
        # case retain their existing behavior rather than making startup fail.
        pass


def _configure_console_output() -> None:
    _configure_text_stream(sys.stdout)
    _configure_text_stream(sys.stderr)


def _imminent_press_mask(target: torch.Tensor, lookahead_frames: int) -> torch.Tensor:
    """Return [T,2] mask: this finger is commanded to press very soon.

    Teacher labels are privileged training-only data, so looking ahead here does
    not expose exact timing to the policy observation.  The mask only shapes the
    loss during BC.
    """

    press = target > v054.TEACHER_ACTIVE_THRESHOLD
    imminent = press.clone()
    for offset in range(1, max(0, int(lookahead_frames)) + 1):
        if offset >= target.shape[0]:
            break
        imminent[:-offset] |= press[offset:]
    return imminent


def _press_persistence_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    coef: float = DEFAULT_PERSISTENCE_COEF,
    lookahead_frames: int = DEFAULT_LOOKAHEAD_FRAMES,
    commit_threshold: float = DEFAULT_COMMIT_THRESHOLD,
    hold_margin: float = DEFAULT_HOLD_MARGIN,
) -> torch.Tensor:
    """v0.5.4 actuation loss plus anti-cancellation persistence penalty.

    If the previous student action has already crossed ``commit_threshold`` and
    the teacher will request a press on that same finger within the short
    look-ahead window, the next student action should remain at least mildly
    positive.  The previous-action commitment mask is detached deliberately:
    the auxiliary term should teach "finish a press you already started", not
    encourage the network to evade the penalty by reducing the preceding action.

    The auxiliary loss is averaged over actual hold-margin violations only.
    Successful persistence frames must not dilute the cost of one abrupt
    cancellation, otherwise the legacy BC term can still prefer cancelling.
    """

    base = _ORIGINAL_ACTUATION_LOSS(predicted, target)
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError("predicted and target must both have shape [T, 2]")
    if predicted.shape[0] < 2 or coef <= 0.0:
        return base

    imminent = _imminent_press_mask(target, lookahead_frames)
    committed_prev = predicted[:-1].detach() >= float(commit_threshold)
    should_hold = committed_prev & imminent[1:]
    if not bool(should_hold.any()):
        return base

    next_action = predicted[1:]
    cancellation_gap = torch.relu(float(hold_margin) - next_action).square()
    violations = should_hold & (next_action < float(hold_margin))
    if not bool(violations.any()):
        return base

    persistence = cancellation_gap[violations].mean()
    return base + float(coef) * persistence


def _install_press_persistence(
    *,
    coef: float,
    lookahead_frames: int,
    commit_threshold: float,
    hold_margin: float,
) -> None:
    def loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return _press_persistence_loss(
            predicted,
            target,
            coef=coef,
            lookahead_frames=lookahead_frames,
            commit_threshold=commit_threshold,
            hold_margin=hold_margin,
        )

    v054._actuation_loss = loss
    v080.TRAINER_VERSION = TRAINER_VERSION
    v080.DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT


def _consume_v090_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--press-persistence-coef", type=float, default=DEFAULT_PERSISTENCE_COEF)
    parser.add_argument("--press-persistence-lookahead", type=int, default=DEFAULT_LOOKAHEAD_FRAMES)
    parser.add_argument("--press-commit-threshold", type=float, default=DEFAULT_COMMIT_THRESHOLD)
    parser.add_argument("--press-hold-margin", type=float, default=DEFAULT_HOLD_MARGIN)
    args, remaining = parser.parse_known_args(argv)
    if args.press_persistence_coef < 0.0:
        raise SystemExit("--press-persistence-coef must be non-negative")
    if args.press_persistence_lookahead < 0:
        raise SystemExit("--press-persistence-lookahead must be non-negative")
    if not -1.0 <= args.press_commit_threshold <= 1.0:
        raise SystemExit("--press-commit-threshold must be in [-1,1]")
    if not -1.0 <= args.press_hold_margin <= 1.0:
        raise SystemExit("--press-hold-margin must be in [-1,1]")
    return args, remaining


def main() -> None:
    _configure_console_output()
    args, remaining = _consume_v090_args(sys.argv[1:])
    _install_press_persistence(
        coef=args.press_persistence_coef,
        lookahead_frames=args.press_persistence_lookahead,
        commit_threshold=args.press_commit_threshold,
        hold_margin=args.press_hold_margin,
    )
    print("=== DMDOD v0.9.0 Press Persistence ===")
    print(
        "press-persistence "
        f"coef={args.press_persistence_coef:g} "
        f"lookahead={args.press_persistence_lookahead}f "
        f"commit>={args.press_commit_threshold:+.2f} "
        f"hold>={args.press_hold_margin:+.2f}"
    )
    sys.argv = [sys.argv[0], *remaining]
    v080.main()


if __name__ == "__main__":
    main()
