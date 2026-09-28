from __future__ import annotations

import argparse

from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import load_compiled_adofai
from dmdod.motor_env import MotorAction
from dmdod.privileged_teacher import calibrate_single_press_lead
from dmdod.real_chart_env import RealChartMotorEnv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a privileged alternating-finger smoke test on a real .adofai segment."
    )
    parser.add_argument("chart")
    parser.add_argument("--start", type=float, default=0.0, help="segment start in chart seconds")
    parser.add_argument("--end", type=float, default=30.0, help="segment end in chart seconds")
    parser.add_argument("--control-dt", type=float, default=0.010)
    parser.add_argument(
        "--cross-hand",
        action="store_true",
        help="use the bilateral two-finger body instead of same-hand fingers",
    )
    args = parser.parse_args()

    compiled = load_compiled_adofai(args.chart)
    segment = build_playable_segment(compiled, start_s=args.start, end_s=args.end)
    if not segment.targets:
        raise SystemExit("segment contains no playable targets")

    same_hand = not args.cross_hand
    calibration = calibrate_single_press_lead(
        control_dt_s=args.control_dt,
        same_hand=same_hand,
    )
    env = RealChartMotorEnv(
        segment,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
    )
    observation = env.reset()

    print(
        f"chart={args.chart}\n"
        f"segment={args.start:g}..{args.end:g}s chart-time "
        f"targets={len(segment.targets)} pitch={compiled.pitch_percent:g}% "
        f"lead={calibration.lead_s * 1000.0:.1f}ms"
    )
    print(
        f"first floor={segment.targets[0].floor_index} "
        f"t={segment.targets[0].episode_time_s:.6f}s | "
        f"last floor={segment.targets[-1].floor_index} "
        f"t={segment.targets[-1].episode_time_s:.6f}s"
    )

    # Privileged smoke-test controller only.  It sees exact target time and must
    # never be confused with the real policy, whose observation contains only
    # body state, rendered planet position, and relative visible floor geometry.
    max_steps = int((segment.duration_s + 2.0) / args.control_dt) + 100
    for _ in range(max_steps):
        target = env.privileged_next_target()
        now = env.privileged_episode_time_s()
        motor = observation.motor

        left = -1.0 if motor.left_pressed else 0.0
        right = -1.0 if motor.right_pressed else 0.0
        if target is not None and now + calibration.lead_s >= target.episode_time_s:
            if target.ordinal & 1:
                if not motor.right_pressed:
                    right = 1.0
            else:
                if not motor.left_pressed:
                    left = 1.0

        step = env.step(MotorAction(left, right))
        observation = step.observation
        if step.done:
            break
    else:
        raise RuntimeError("real-chart episode exceeded smoke-test step budget")

    stats = env.stats
    print(
        f"result H={stats.hits}/{stats.targets} miss={stats.misses} "
        f"X={stats.x_accuracy_percent:.2f}% PP={stats.perfect_rate * 100.0:.1f}% "
        f"MAE={stats.mean_abs_error_ms if stats.mean_abs_error_ms is not None else float('nan'):.2f}ms "
        f"early={stats.too_early_presses} overload={stats.overloaded}"
    )


if __name__ == "__main__":
    main()
