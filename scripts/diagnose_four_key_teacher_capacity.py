from __future__ import annotations

"""Diagnose whether the four-key privileged teacher is finger-capacity limited.

This is an evaluator/debug tool only.  It intentionally inspects teacher
reservations and evaluator miss flags that are never exposed to the policy.
"""

import argparse
from dataclasses import dataclass

from dmdod.four_key_calibration import calibrate_four_key_press_lead
from dmdod.four_key_motor import FOUR_KEY_NAMES
from dmdod.four_key_real_chart import DiagnosticHudFourKeyRealChartMotorEnv
from dmdod.four_key_training import CenterFirstFourKeyTeacher
from dmdod.multichart_dataset import discover_multichart_dataset
from dmdod.real_chart_features import DEFAULT_REAL_CHART_FEATURE_CONFIG

import train_real_chart_v080 as v080


@dataclass(frozen=True, slots=True)
class FourKeyTeacherCapacityResult:
    stats: object
    physical_keydowns: int
    max_due_targets: int
    max_reservations: int
    max_pressed_keys: int
    max_blocked_due_targets: int
    blocked_unique_targets: int
    reservation_capacity_frames: int
    release_wait_frames: int
    other_block_frames: int
    missed_targets: int
    missed_after_capacity_block: int
    missed_without_capacity_block: int
    peak_targets_per_lead_window: int
    min_target_gap_ms: float | None


def _target_token(target) -> tuple[int, int, float]:
    return (int(target.ordinal), int(target.floor_index), float(target.episode_time_s))


def _unresolved_tail(env) -> tuple[tuple[object, float], ...]:
    target = env.privileged_next_target()
    if target is None:
        return ()
    return tuple(
        (_target_token(future), float(future.episode_time_s))
        for future in env.segment.targets[int(target.ordinal) :]
    )


def _peak_targets_in_window(segment, window_s: float) -> int:
    times = [float(target.episode_time_s) for target in segment.targets]
    if not times:
        return 0
    left = 0
    best = 0
    for right, time_s in enumerate(times):
        while left <= right and time_s - times[left] > window_s + 1e-12:
            left += 1
        best = max(best, right - left + 1)
    return best


def _min_target_gap_ms(segment) -> float | None:
    times = [float(target.episode_time_s) for target in segment.targets]
    if len(times) < 2:
        return None
    return min((b - a) * 1000.0 for a, b in zip(times, times[1:]))


def diagnose_four_key_teacher_capacity(
    segment,
    *,
    lead_s: float,
    control_dt_s: float,
) -> FourKeyTeacherCapacityResult:
    env = DiagnosticHudFourKeyRealChartMotorEnv(
        segment,
        control_dt_s=control_dt_s,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )
    teacher = CenterFirstFourKeyTeacher()
    observation = env.reset()

    max_due = 0
    max_reservations = 0
    max_pressed = 0
    max_blocked = 0
    reservation_capacity_frames = 0
    release_wait_frames = 0
    other_block_frames = 0
    blocked_tokens: set[object] = set()

    max_steps = int((segment.duration_s + 2.0) / control_dt_s) + 200
    for _ in range(max_steps):
        now_s = float(env.privileged_episode_time_s())
        targets = _unresolved_tail(env)
        due_tokens = [
            token
            for token, target_time_s in targets
            if now_s + float(lead_s) >= float(target_time_s)
        ]
        max_due = max(max_due, len(due_tokens))

        action = teacher.pipeline_action(
            observation.motor,
            now_s=now_s,
            targets=targets,
            lead_s=lead_s,
        )

        reservations = dict(teacher._reservations)  # debug-only privileged state
        reserved_tokens = {token for token, _ in reservations.values()}
        pressed = {key for key in FOUR_KEY_NAMES if observation.motor.pressed(key)}
        occupied = pressed | set(reservations)
        blocked = [token for token in due_tokens if token not in reserved_tokens]

        max_reservations = max(max_reservations, len(reservations))
        max_pressed = max(max_pressed, len(pressed))
        max_blocked = max(max_blocked, len(blocked))
        blocked_tokens.update(blocked)

        if blocked:
            if len(reservations) >= len(FOUR_KEY_NAMES):
                reservation_capacity_frames += 1
            elif len(occupied) >= len(FOUR_KEY_NAMES):
                release_wait_frames += 1
            else:
                other_block_frames += 1

        step = env.step(action)
        observation = step.observation
        if step.done:
            break
    else:
        raise RuntimeError("four-key teacher capacity diagnostic exceeded step budget")

    missed_tokens = {
        _target_token(target)
        for target, missed in zip(segment.targets, env._missed)  # debug-only evaluator state
        if missed
    }
    capacity_misses = missed_tokens & blocked_tokens

    return FourKeyTeacherCapacityResult(
        stats=env.stats,
        physical_keydowns=int(env.physical_keydowns),
        max_due_targets=max_due,
        max_reservations=max_reservations,
        max_pressed_keys=max_pressed,
        max_blocked_due_targets=max_blocked,
        blocked_unique_targets=len(blocked_tokens),
        reservation_capacity_frames=reservation_capacity_frames,
        release_wait_frames=release_wait_frames,
        other_block_frames=other_block_frames,
        missed_targets=len(missed_tokens),
        missed_after_capacity_block=len(capacity_misses),
        missed_without_capacity_block=len(missed_tokens - blocked_tokens),
        peak_targets_per_lead_window=_peak_targets_in_window(segment, float(lead_s)),
        min_target_gap_ms=_min_target_gap_ms(segment),
    )


def _format_result(label: str, result: FourKeyTeacherCapacityResult) -> str:
    gap = "n/a" if result.min_target_gap_ms is None else f"{result.min_target_gap_ms:.2f}ms"
    stats = result.stats
    return (
        f"{label}: H={stats.hits}/{stats.targets} miss={result.missed_targets} "
        f"X={stats.x_accuracy_percent:.2f}% early={stats.too_early_presses} "
        f"over={stats.overloaded} keydowns={result.physical_keydowns}\n"
        f"  demand: peak/lead={result.peak_targets_per_lead_window} min-gap={gap} "
        f"max-due={result.max_due_targets} max-reserved={result.max_reservations} "
        f"max-pressed={result.max_pressed_keys} max-blocked={result.max_blocked_due_targets}\n"
        f"  blocked: unique={result.blocked_unique_targets} "
        f"frames(reservation-cap={result.reservation_capacity_frames}, "
        f"release-wait={result.release_wait_frames}, other={result.other_block_frames})\n"
        f"  misses: after-capacity-block={result.missed_after_capacity_block} "
        f"without-capacity-block={result.missed_without_capacity_block}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose four-key teacher misses as finger-capacity/release bottlenecks."
    )
    parser.add_argument("dataset")
    parser.add_argument("--train-window", type=float, default=v080.DEFAULT_TRAIN_WINDOW_S)
    parser.add_argument("--anchors-per-chart", type=int, default=v080.DEFAULT_ANCHORS_PER_CHART)
    parser.add_argument("--anchor-limit", type=int, default=None)
    parser.add_argument("--control-dt", type=float, default=0.010)
    parser.add_argument("--physics-dt", type=float, default=0.001)
    args = parser.parse_args()

    if args.train_window <= 0.0 or args.anchors_per_chart <= 0:
        raise SystemExit("train-window/anchors-per-chart must be positive")
    if args.anchor_limit is not None and args.anchor_limit <= 0:
        raise SystemExit("--anchor-limit must be positive")

    dataset = discover_multichart_dataset(args.dataset)
    train_charts = v080._compile_role(dataset.train)
    anchors = v080._build_anchor_segments(
        train_charts,
        window_s=args.train_window,
        anchors_per_chart=args.anchors_per_chart,
    )
    if args.anchor_limit is not None:
        anchors = anchors[: args.anchor_limit]

    calibration = calibrate_four_key_press_lead(
        control_dt_s=args.control_dt,
        physics_dt_s=args.physics_dt,
    )
    print("=== DMDOD Four-Key Teacher Capacity Diagnostic ===")
    print(
        f"anchors={len(anchors)} lead={calibration.lead_s * 1000.0:.1f}ms "
        f"control={args.control_dt * 1000.0:.1f}ms keys={len(FOUR_KEY_NAMES)}"
    )

    for index, named in enumerate(anchors, 1):
        result = diagnose_four_key_teacher_capacity(
            named.segment,
            lead_s=calibration.lead_s,
            control_dt_s=args.control_dt,
        )
        print(_format_result(f"{index:03d}/{len(anchors)} {named.chart_name}", result))


if __name__ == "__main__":
    main()
