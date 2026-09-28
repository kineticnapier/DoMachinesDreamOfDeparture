from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

import train_real_chart_v054 as v054
import train_real_chart_v058 as v058
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import load_compiled_adofai
from dmdod.privileged_teacher import calibrate_single_press_lead
from dmdod.real_chart_features import (
    DEFAULT_REAL_CHART_FEATURE_CONFIG,
    REAL_CHART_INPUT_DIM,
    encode_real_chart_observation,
)
from dmdod.recurrent_policy import RecurrentActorCritic


DIAGNOSTIC_VERSION = "0.5.9-intervention-diagnostics"
DEFAULT_CHECKPOINT = "checkpoints/real_chart_v058_trust_region.pt"
DEFAULT_DELTA_THRESHOLD = 0.25
DEFAULT_TOP = 10


@dataclass(frozen=True, slots=True)
class InterventionFrame:
    frame_index: int
    time_s: float
    target_ordinal: int | None
    floor_index: int | None
    target_time_s: float | None
    kind: str
    student_left: float
    student_right: float
    teacher_left: float
    teacher_right: float
    max_abs_delta: float


@dataclass(frozen=True, slots=True)
class InterventionSummary:
    total_frames: int
    intervention_frames: int
    intervention_runs: int
    longest_run_frames: int
    kind_counts: dict[str, int]
    mean_max_abs_delta: float
    max_abs_delta: float

    @property
    def intervention_rate(self) -> float:
        return self.intervention_frames / max(1, self.total_frames)


def _action_kind(left: float, right: float, *, active_threshold: float = 0.25) -> str:
    if max(left, right) > active_threshold:
        return "press"
    if min(left, right) < -active_threshold:
        return "release"
    return "neutral"


def _max_action_delta(
    student_left: float,
    student_right: float,
    teacher_left: float,
    teacher_right: float,
) -> float:
    return max(
        abs(student_left - teacher_left),
        abs(student_right - teacher_right),
    )


def _is_intervention(frame: InterventionFrame, *, delta_threshold: float) -> bool:
    if delta_threshold < 0.0:
        raise ValueError("delta_threshold must be non-negative")
    return frame.max_abs_delta >= delta_threshold


def _summarize(
    frames: list[InterventionFrame],
    *,
    total_frames: int,
    delta_threshold: float,
) -> InterventionSummary:
    chosen = [frame for frame in frames if _is_intervention(frame, delta_threshold=delta_threshold)]
    kinds = Counter(frame.kind for frame in chosen)

    runs = 0
    longest = 0
    current = 0
    previous_index: int | None = None
    for frame in chosen:
        if previous_index is None or frame.frame_index != previous_index + 1:
            runs += 1
            current = 1
        else:
            current += 1
        longest = max(longest, current)
        previous_index = frame.frame_index

    deltas = [frame.max_abs_delta for frame in chosen]
    return InterventionSummary(
        total_frames=total_frames,
        intervention_frames=len(chosen),
        intervention_runs=runs,
        longest_run_frames=longest,
        kind_counts={kind: kinds.get(kind, 0) for kind in ("press", "release", "neutral")},
        mean_max_abs_delta=sum(deltas) / max(1, len(deltas)),
        max_abs_delta=max(deltas, default=0.0),
    )


def _collect(
    model: RecurrentActorCritic,
    segment,
    *,
    lead_s: float,
    same_hand: bool,
    control_dt_s: float,
    device: torch.device,
) -> tuple[list[InterventionFrame], v054.StudentEvalResult, int]:
    env = v054.DiagnosticRealChartMotorEnv(
        segment,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )
    observation = env.reset()
    state = model.initial_state(device)
    frames: list[InterventionFrame] = []

    max_steps = int((segment.duration_s + 2.0) / control_dt_s) + 200
    total_frames = 0
    with torch.no_grad():
        for frame_index in range(max_steps):
            encoded = encode_real_chart_observation(observation)
            x = torch.tensor(encoded, dtype=torch.float32, device=device)
            student, state = model.deterministic_action(x, state)
            teacher = v054.base._teacher_action(env, observation, lead_s)
            target = env.privileged_next_target()
            now_s = env.privileged_episode_time_s()

            sl = float(student.left)
            sr = float(student.right)
            tl = float(teacher.left)
            tr = float(teacher.right)
            frames.append(
                InterventionFrame(
                    frame_index=frame_index,
                    time_s=now_s,
                    target_ordinal=None if target is None else int(target.ordinal),
                    floor_index=None if target is None else int(target.floor_index),
                    target_time_s=None if target is None else float(target.episode_time_s),
                    kind=_action_kind(tl, tr),
                    student_left=sl,
                    student_right=sr,
                    teacher_left=tl,
                    teacher_right=tr,
                    max_abs_delta=_max_action_delta(sl, sr, tl, tr),
                )
            )
            total_frames += 1

            step = env.step(student)
            observation = step.observation
            if step.done:
                break
        else:
            raise RuntimeError("intervention diagnostic exceeded real-chart step budget")

    return frames, v054.StudentEvalResult(env.stats, env.physical_keydowns), total_frames


def _print_target_hotspots(
    frames: list[InterventionFrame],
    *,
    delta_threshold: float,
    top: int,
) -> None:
    grouped: dict[tuple[int | None, int | None, float | None], list[InterventionFrame]] = defaultdict(list)
    for frame in frames:
        if _is_intervention(frame, delta_threshold=delta_threshold):
            grouped[(frame.target_ordinal, frame.floor_index, frame.target_time_s)].append(frame)

    ranked = sorted(
        grouped.items(),
        key=lambda item: (
            len(item[1]),
            max(frame.max_abs_delta for frame in item[1]),
        ),
        reverse=True,
    )[:top]

    print(f"target hotspots (top {min(top, len(ranked))}):")
    if not ranked:
        print("  none")
        return

    for (ordinal, floor_index, target_time_s), target_frames in ranked:
        counts = Counter(frame.kind for frame in target_frames)
        mean_delta = sum(frame.max_abs_delta for frame in target_frames) / len(target_frames)
        max_delta = max(frame.max_abs_delta for frame in target_frames)
        target_text = "after-end" if ordinal is None else f"target={ordinal} floor={floor_index} t={target_time_s:.3f}s"
        print(
            f"  {target_text} frames={len(target_frames)} "
            f"P/R/N={counts['press']}/{counts['release']}/{counts['neutral']} "
            f"meanD={mean_delta:.3f} maxD={max_delta:.3f}"
        )


def _print_worst_frames(
    frames: list[InterventionFrame],
    *,
    delta_threshold: float,
    top: int,
) -> None:
    chosen = [frame for frame in frames if _is_intervention(frame, delta_threshold=delta_threshold)]
    chosen.sort(key=lambda frame: frame.max_abs_delta, reverse=True)
    chosen = chosen[:top]

    print(f"largest action disagreements (top {min(top, len(chosen))}):")
    if not chosen:
        print("  none")
        return

    for frame in chosen:
        target_text = "after-end" if frame.target_ordinal is None else f"target={frame.target_ordinal} floor={frame.floor_index}"
        print(
            f"  t={frame.time_s:.3f}s {target_text} {frame.kind} "
            f"student=({frame.student_left:+.3f},{frame.student_right:+.3f}) "
            f"teacher=({frame.teacher_left:+.1f},{frame.teacher_right:+.1f}) "
            f"D={frame.max_abs_delta:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose where a real-chart student disagrees with the privileged teacher. "
            "This is evaluator-only analysis; privileged target data is never fed to the policy."
        )
    )
    parser.add_argument("chart")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=30.0)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--control-dt", type=float, default=0.010)
    parser.add_argument("--delta-threshold", type=float, default=DEFAULT_DELTA_THRESHOLD)
    parser.add_argument("--top", type=int, default=DEFAULT_TOP)
    parser.add_argument("--cross-hand", action="store_true")
    args = parser.parse_args()

    if args.end <= args.start:
        raise SystemExit("--end must be greater than --start")
    if args.hidden <= 0 or args.control_dt <= 0.0 or args.top <= 0:
        raise SystemExit("hidden/control-dt/top must be positive")
    if args.delta_threshold < 0.0:
        raise SystemExit("--delta-threshold must be non-negative")

    device = torch.device("cpu")
    same_hand = not args.cross_hand
    compiled = load_compiled_adofai(args.chart)
    end_s = min(args.end, compiled.duration_s)
    segment = build_playable_segment(compiled, start_s=args.start, end_s=end_s)
    if not segment.targets:
        raise SystemExit("diagnostic segment contains no playable targets")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")

    model = RecurrentActorCritic(
        input_dim=REAL_CHART_INPUT_DIM,
        hidden_dim=args.hidden,
        initial_log_std=-1.20,
    ).to(device)
    source_version = v058._load_bootstrap(
        model,
        checkpoint_path,
        hidden_dim=args.hidden,
        device=device,
    )
    model.eval()

    calibration = calibrate_single_press_lead(
        control_dt_s=args.control_dt,
        same_hand=same_hand,
    )
    frames, evaluation, total_frames = _collect(
        model,
        segment,
        lead_s=calibration.lead_s,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
        device=device,
    )
    summary = _summarize(
        frames,
        total_frames=total_frames,
        delta_threshold=args.delta_threshold,
    )

    counts = summary.kind_counts
    print(f"=== DMDOD / Real Chart Intervention Diagnostic v0.5.9 ===")
    print(
        f"chart={args.chart}\n"
        f"segment={args.start:g}..{end_s:g}s targets={len(segment.targets)} "
        f"checkpoint={checkpoint_path} ({source_version})"
    )
    print(v054._format_eval("student", evaluation))
    print(
        f"intervention threshold=max|student-teacher|>={args.delta_threshold:g} "
        f"frames={summary.intervention_frames}/{summary.total_frames} "
        f"({summary.intervention_rate * 100.0:.1f}%) "
        f"runs={summary.intervention_runs} longest={summary.longest_run_frames}f"
    )
    print(
        f"intervention P/R/N={counts['press']}/{counts['release']}/{counts['neutral']} "
        f"meanD={summary.mean_max_abs_delta:.3f} maxD={summary.max_abs_delta:.3f}"
    )
    _print_target_hotspots(
        frames,
        delta_threshold=args.delta_threshold,
        top=args.top,
    )
    _print_worst_frames(
        frames,
        delta_threshold=args.delta_threshold,
        top=args.top,
    )


if __name__ == "__main__":
    main()
