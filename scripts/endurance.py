from __future__ import annotations

import argparse

from dmdod.body import TwoFingerBody
from dmdod.keyboard import KeyEvent
from dmdod.profiles import PERSONAL_BLUE_SWITCH_V0_1_NAME, personal_blue_switch_v0_1
from dmdod.simulator import Simulation


def run_endurance(
    rate_hz: float,
    duration_s: float,
    *,
    warmup_s: float = 1.0,
    sample_s: float = 10.0,
) -> None:
    """Run one paced single-finger endurance trial.

    Control remains threshold/event driven.  After DOWN the finger releases until
    UP; before the next DOWN, the controller starts pressing early by its current
    estimate of press travel time.  This targets a cadence without imposing the
    old 50:50 square-wave duty cycle.
    """

    sim = Simulation(body=TwoFingerBody(config=personal_blue_switch_v0_1(same_hand=True)))
    dt = sim.config.dt_s
    interval_s = 1.0 / rate_hz
    warmup_steps = round(warmup_s / dt)
    measure_steps = round(duration_s / dt)
    sample_steps = max(1, round(sample_s / dt))
    total_steps = warmup_steps + measure_steps

    pressing = True
    press_start_s = 0.0
    last_down_s: float | None = None
    estimated_press_s = interval_s * 0.40
    next_press_start_s = 0.0

    presses = 0
    previous_presses = 0
    max_fatigue = 0.0
    measure_start_s = warmup_s

    print(
        f"target={rate_hz:.4f} KPS  duration={duration_s:.0f}s  "
        f"warmup={warmup_s:.1f}s  profile={PERSONAL_BLUE_SWITCH_V0_1_NAME}"
    )
    print(" time   window_KPS  target  cumulative  fatigue   max_fatigue")
    print("--------------------------------------------------------------")

    for step in range(total_steps):
        now = sim.time_s

        if pressing:
            command = 1.0
        elif sim.keyboard.left.pressed:
            command = -1.0
        elif now >= next_press_start_s:
            pressing = True
            press_start_s = now
            command = 1.0
        else:
            command = -1.0

        result = sim.step(command, 0.0)
        max_fatigue = max(max_fatigue, result.left.fatigue)

        for finger, event in result.events:
            if finger != "left":
                continue

            if event is KeyEvent.DOWN:
                observed_press_s = max(dt, result.time_s - press_start_s)
                # Smooth the travel-time estimate so pacing adapts as fatigue
                # changes instead of assuming a fixed duty cycle.
                estimated_press_s = 0.8 * estimated_press_s + 0.2 * observed_press_s
                last_down_s = result.time_s
                next_press_start_s = last_down_s + interval_s - estimated_press_s
                pressing = False
                if step >= warmup_steps:
                    presses += 1
            elif event is KeyEvent.UP:
                # Stay released until the scheduled press start.  If fatigue made
                # release take too long, the next press starts immediately.
                pressing = result.time_s >= next_press_start_s
                if pressing:
                    press_start_s = result.time_s

        if step >= warmup_steps:
            measure_index = step - warmup_steps + 1
            if measure_index % sample_steps == 0 or measure_index == measure_steps:
                elapsed = measure_index * dt
                window_duration = sample_s
                if measure_index == measure_steps and measure_index % sample_steps:
                    window_duration = (measure_index % sample_steps) * dt
                window_presses = presses - previous_presses
                window_kps = window_presses / max(window_duration, dt)
                cumulative = presses / max(elapsed, dt)
                print(
                    f"{elapsed:5.0f}s   {window_kps:9.3f}  {rate_hz:6.3f}  "
                    f"{cumulative:10.3f}  {result.left.fatigue:8.5f}  {max_fatigue:11.5f}"
                )
                previous_presses = presses

    expected = duration_s * rate_hz
    ratio = presses / expected if expected else 0.0
    final_rate = presses / duration_s
    print("--------------------------------------------------------------")
    print(
        f"total: presses={presses}, expected~={expected:.1f}, "
        f"rate={final_rate:.4f} KPS, retention={ratio:.4f}, "
        f"final_fatigue={sim.body.left.fatigue:.5f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Long-duration single-finger endurance stress test for the frozen v0.1 profile."
    )
    parser.add_argument(
        "--bpm",
        type=float,
        default=400.0,
        help="One index-finger press per beat. Default: 400 BPM = 6.6667 KPS.",
    )
    parser.add_argument(
        "--durations",
        nargs="+",
        type=float,
        default=[60.0, 180.0, 300.0],
        help="Trial durations in seconds.",
    )
    parser.add_argument("--warmup", type=float, default=1.0)
    parser.add_argument("--sample", type=float, default=10.0)
    args = parser.parse_args()

    if args.bpm <= 0.0 or any(d <= 0.0 for d in args.durations):
        raise SystemExit("BPM and durations must be positive")

    rate_hz = args.bpm / 60.0
    print("=== Endurance Stress Test ===")
    print(f"reference cadence: {args.bpm:g} BPM = {rate_hz:.4f} KPS")
    print("This is a diagnostic reference, not a fitted personal calibration target.")
    print()

    for duration_s in args.durations:
        run_endurance(rate_hz, duration_s, warmup_s=args.warmup, sample_s=args.sample)
        print()


if __name__ == "__main__":
    main()
