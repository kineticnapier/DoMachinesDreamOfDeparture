from __future__ import annotations

import argparse
from dataclasses import replace

from dmdod.body import BodyConfig, FingerConfig, TwoFingerBody
from dmdod.keyboard import KeyEvent
from dmdod.simulator import Simulation


def mechanical_body() -> BodyConfig:
    """Recent mechanics calibration with fatigue disabled to isolate the rate cliff."""
    finger = FingerConfig(
        damping_n_s_m=0.630000,
        max_force_n=1.400000,
        activation_tau_s=0.420000,
        fatigue_gain_s=0.0,
        fatigue_recovery_s=0.0,
        switch_fatigue_per_reversal=0.0,
    )
    base = BodyConfig(left=finger, right=replace(finger), same_hand=False)
    return base


def run(rate_hz: float, duration_s: float) -> None:
    sim = Simulation(body=TwoFingerBody(config=mechanical_body()))
    dt = sim.config.dt_s
    interval_ms = 1000.0 / rate_hz
    half_steps = max(1, round((interval_ms / 2.0) / (dt * 1000.0)))
    total_steps = round(duration_s / dt)

    downs = 0
    ups = 0
    last_down = None
    min_pos = float("inf")
    max_pos = float("-inf")
    min_pos_last = float("inf")
    max_pos_last = float("-inf")
    last_window_start = max(0, total_steps - round(1.0 / dt))

    for step in range(total_steps):
        phase = (step // half_steps) % 2
        command = 1.0 if phase == 0 else -1.0
        result = sim.step(command, 0.0)
        p = result.left.position_m
        min_pos = min(min_pos, p)
        max_pos = max(max_pos, p)
        if step >= last_window_start:
            min_pos_last = min(min_pos_last, p)
            max_pos_last = max(max_pos_last, p)
        for finger, event in result.events:
            if finger != "left":
                continue
            if event is KeyEvent.DOWN:
                downs += 1
                last_down = result.time_s
            elif event is KeyEvent.UP:
                ups += 1

    cfg = sim.keyboard.config
    expected = duration_s * rate_hz
    print(f"{rate_hz:5.2f} KPS  down={downs:4d}/{expected:6.1f}  up={ups:4d}  ratio={downs/expected:6.3f}")
    print(
        f"           pos(all)={min_pos*1000:7.3f}..{max_pos*1000:7.3f} mm  "
        f"pos(last1s)={min_pos_last*1000:7.3f}..{max_pos_last*1000:7.3f} mm"
    )
    print(
        f"           reset={cfg.reset_m*1000:.3f} mm  actuation={cfg.actuation_m*1000:.3f} mm  "
        f"last_down={last_down if last_down is not None else 'never'}"
    )
    if min_pos_last > cfg.reset_m:
        print("           diagnosis: finger never reaches reset point in the final second")
    elif max_pos_last < cfg.actuation_m:
        print("           diagnosis: finger never reaches actuation point in the final second")
    else:
        print("           diagnosis: final-second travel crosses both thresholds")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the mechanical tapping-rate cliff with fatigue disabled.")
    parser.add_argument("--rates", nargs="+", type=float, default=[8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0])
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()

    for rate in args.rates:
        run(rate, args.duration)
        print()


if __name__ == "__main__":
    main()
