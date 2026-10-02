from __future__ import annotations

"""Compare center-first teacher capacity across configurable even key counts."""

import argparse

from dmdod.multichart_dataset import discover_multichart_dataset
from dmdod.n_key_capacity import (
    calibrate_n_key_press_lead,
    diagnose_n_key_teacher_capacity,
)

import train_real_chart_v080 as v080


def _format_result(key_count: int, label: str, result) -> str:
    gap = "n/a" if result.min_target_gap_ms is None else f"{result.min_target_gap_ms:.2f}ms"
    stats = result.stats
    return (
        f"{key_count}K {label}: H={stats.hits}/{stats.targets} miss={result.missed_targets} "
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


def _parse_key_counts(raw: str) -> tuple[int, ...]:
    result: list[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = int(piece)
        if value < 2 or value % 2 != 0:
            raise argparse.ArgumentTypeError("key counts must be even integers >= 2")
        if value not in result:
            result.append(value)
    if not result:
        raise argparse.ArgumentTypeError("at least one key count is required")
    return tuple(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare teacher-only physical capacity across 4K/6K/8K-style bodies."
    )
    parser.add_argument("dataset")
    parser.add_argument("--keys", default="4,6,8")
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
    try:
        key_counts = _parse_key_counts(args.keys)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise SystemExit(str(exc)) from exc

    dataset = discover_multichart_dataset(args.dataset)
    train_charts = v080._compile_role(dataset.train)
    anchors = v080._build_anchor_segments(
        train_charts,
        window_s=args.train_window,
        anchors_per_chart=args.anchors_per_chart,
    )
    if args.anchor_limit is not None:
        anchors = anchors[: args.anchor_limit]

    calibrations = {
        key_count: calibrate_n_key_press_lead(
            key_count,
            control_dt_s=args.control_dt,
            physics_dt_s=args.physics_dt,
        )
        for key_count in key_counts
    }

    print("=== DMDOD N-Key Teacher Capacity Comparison ===")
    print(
        f"anchors={len(anchors)} keys={','.join(str(value) for value in key_counts)} "
        f"control={args.control_dt * 1000.0:.1f}ms physics={args.physics_dt * 1000.0:.1f}ms"
    )
    for key_count in key_counts:
        calibration = calibrations[key_count]
        print(
            f"{key_count}K calibration: lead={calibration.lead_s * 1000.0:.1f}ms "
            f"latencies="
            + ",".join(f"{key}:{latency * 1000.0:.1f}" for key, latency in calibration.key_latencies_s)
        )

    for index, named in enumerate(anchors, 1):
        label = f"{index:03d}/{len(anchors)} {named.chart_name}"
        print(f"--- {label} ---")
        for key_count in key_counts:
            calibration = calibrations[key_count]
            result = diagnose_n_key_teacher_capacity(
                named.segment,
                key_count=key_count,
                lead_s=calibration.lead_s,
                control_dt_s=args.control_dt,
                physics_dt_s=args.physics_dt,
            )
            print(_format_result(key_count, label, result))


if __name__ == "__main__":
    main()
