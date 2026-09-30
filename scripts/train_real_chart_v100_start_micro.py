from __future__ import annotations

"""v1.0: emphasize chart-start countdown -> first-press generalization.

v0.9 already has a 30-second anchor starting at t=0 for every training chart.
That long anchor can hide a single bad first press inside many later successful
hits.  v1.0 keeps the existing anchors and adds one short ``start-micro`` anchor
per training chart containing only the countdown plus the first few playable
targets.  The micro anchor is used both as expert BC data and as an anti-
forgetting guard, so an early first press becomes a large completion regression
instead of roughly a 1/N error inside a 30-second window.

``--warm-start`` deliberately imports only model weights.  New v1.0 anchor and
validation references are rebuilt by the ordinary fresh-bootstrap path, so an
old v0.9 checkpoint cannot smuggle its 94-anchor guard baseline into v1.0.
"""

import argparse
import sys
from pathlib import Path

import torch

import train_real_chart_v080 as v080
import train_real_chart_v080_fast as v080_fast
import train_real_chart_v090 as v090
import train_real_chart_v090_fast as v090_fast
import train_real_chart_v090_turbo as turbo


TRAINER_VERSION = "1.0.0-start-micro"
DEFAULT_CHECKPOINT = "checkpoints/real_chart_v100_start_micro.pt"
DEFAULT_START_MICRO_TARGETS = 4
START_MICRO_VERSION = "v100-countdown-first-targets-v1"

_BASE_BUILD_ANCHOR_SEGMENTS = turbo._ORIGINAL_BUILD_ANCHOR_SEGMENTS
_BASE_POLICY_CLASS = v080.RecurrentActorCritic
_START_MICRO_TARGETS = DEFAULT_START_MICRO_TARGETS
_WARM_START_PATH: Path | None = None


def _playable_target_times(runtime) -> list[float]:
    return [
        float(floor.target_time_s)
        for floor in runtime.compiled.floors[1:]
        if not bool(floor.midspin)
    ]


def _start_micro_end_s(runtime, target_count: int) -> float:
    if int(target_count) <= 0:
        raise ValueError("target_count must be positive")
    times = _playable_target_times(runtime)
    if not times:
        raise ValueError(f"chart has no playable targets: {runtime.spec.name}")
    target_time = times[min(int(target_count), len(times)) - 1]
    # The target interval is inclusive in build_playable_segment.  Keep the
    # segment narrowly focused while avoiding a zero-duration segment on charts
    # whose first playable target is exactly at t=0.
    end_s = max(float(target_time), min(float(runtime.duration_s), 1e-6))
    return min(float(runtime.duration_s), end_s)


def _build_anchor_segments_with_start_micro(
    train_charts,
    *,
    window_s: float,
    anchors_per_chart: int,
):
    result = list(
        _BASE_BUILD_ANCHOR_SEGMENTS(
            train_charts,
            window_s=window_s,
            anchors_per_chart=anchors_per_chart,
        )
    )
    existing = {named.key for named in result}
    added = 0
    for chart in train_charts:
        end_s = _start_micro_end_s(chart, _START_MICRO_TARGETS)
        named = v080._named_segment(
            chart,
            f"start-micro-{_START_MICRO_TARGETS}",
            0.0,
            end_s,
        )
        if named.key in existing:
            continue
        result.append(named)
        existing.add(named.key)
        added += 1
    print(
        f"start-micro: version={START_MICRO_VERSION} targets={_START_MICRO_TARGETS} "
        f"added={added} total-anchors={len(result)}"
    )
    return result


def install_start_micro(*, target_count: int = DEFAULT_START_MICRO_TARGETS) -> None:
    global _START_MICRO_TARGETS
    if int(target_count) <= 0:
        raise ValueError("target_count must be positive")
    _START_MICRO_TARGETS = int(target_count)

    # Turbo captures anchors by calling this stored builder.  Point that capture
    # at the augmented builder while leaving v0.8 itself untouched.
    turbo._ORIGINAL_BUILD_ANCHOR_SEGMENTS = _build_anchor_segments_with_start_micro
    v080.TRAINER_VERSION = TRAINER_VERSION
    v080.DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT


def _load_warm_start_into(model, path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise SystemExit("warm-start checkpoint payload is not a dictionary")
    if int(payload.get("input_dim", -1)) != v080.v070.HUD_REAL_CHART_INPUT_DIM:
        raise SystemExit("warm-start checkpoint input dimension does not match the 245D HUD")
    if int(payload.get("hidden_dim", -1)) != int(model.hidden_dim):
        raise SystemExit("warm-start checkpoint hidden size does not match --hidden")
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise SystemExit("warm-start checkpoint has no model_state")
    model.load_state_dict(state)
    return payload


def install_warm_start(path: str | Path) -> None:
    global _WARM_START_PATH
    warm_path = Path(path)
    if not warm_path.exists():
        raise SystemExit(f"warm-start checkpoint not found: {warm_path}")
    _WARM_START_PATH = warm_path

    class WarmStartedRecurrentActorCritic(_BASE_POLICY_CLASS):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            assert _WARM_START_PATH is not None
            payload = _load_warm_start_into(self, _WARM_START_PATH)
            print(
                f"warm-start={_WARM_START_PATH} weights-only "
                f"source-trainer={payload.get('trainer_version', '?')} "
                f"source-round={int(payload.get('completed_round', 0))} "
                f"finalized={bool(payload.get('finalized', False))}"
            )

    # v0.8 constructs exactly one policy in main().  Replacing only that symbol
    # keeps checkpoint parsing/resume code untouched while making the fresh
    # bootstrap start from the imported weights.
    v080.RecurrentActorCritic = WarmStartedRecurrentActorCritic


def _consume_v100_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--start-micro-targets",
        type=int,
        default=DEFAULT_START_MICRO_TARGETS,
    )
    parser.add_argument("--warm-start", default=None)
    args, remaining = parser.parse_known_args(argv)
    if args.start_micro_targets <= 0:
        raise SystemExit("--start-micro-targets must be positive")
    if args.warm_start is not None and "--resume" in remaining:
        raise SystemExit("--warm-start and --resume are mutually exclusive")
    return args, remaining


def main() -> None:
    v090._configure_console_output()
    start_args, after_v100 = _consume_v100_args(sys.argv[1:])
    press_args, remaining = v090._consume_v090_args(after_v100)

    v090_fast._install_v090_fast_path(
        coef=press_args.press_persistence_coef,
        lookahead_frames=press_args.press_persistence_lookahead,
        commit_threshold=press_args.press_commit_threshold,
        hold_margin=press_args.press_hold_margin,
    )
    install_start_micro(target_count=start_args.start_micro_targets)
    if start_args.warm_start is not None:
        install_warm_start(start_args.warm_start)
    turbo._install_turbo_path()

    print("=== DMDOD v1.0.0 Start Micro ===")
    print(
        f"start-micro targets={start_args.start_micro_targets} | "
        f"press-persistence coef={press_args.press_persistence_coef:g} "
        f"lookahead={press_args.press_persistence_lookahead}f "
        f"commit>={press_args.press_commit_threshold:+.2f} "
        f"hold>={press_args.press_hold_margin:+.2f}"
    )
    print(
        f"turbo={turbo.TURBO_VERSION} | fast-eval={v080_fast.FAST_EVAL_VERSION} | "
        "start-micro=train+guard"
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
