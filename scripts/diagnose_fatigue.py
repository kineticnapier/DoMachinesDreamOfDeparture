from __future__ import annotations

import argparse
from dataclasses import replace

from dmdod.body import BodyConfig, FingerConfig, TwoFingerBody
from dmdod.keyboard import KeyEvent
from dmdod.simulator import Simulation


def calibrated_body() -> BodyConfig:
    """Current best same-hand calibration, copied from the latest calibration run."""
    finger = FingerConfig(
        damping_n_s_m=0.630000,
        max_force_n=1.400000,
        activation_tau_s=0.420000,
        fatigue_gain_s=0.040000,
        fatigue_recovery_s=0.028000,
    )
    base = BodyConfig(left=finger, right=replace(finger))
    hand = replace(
        base.hand,
        capacity=0.472500,
        fatigue_gain_s=0.030000,
        fatigue_recovery_s=0.250000,
        switch_tau_s=0.054000,
        coordination_floor=0.050000,
    )
    return replace(base, hand=hand, same_hand=True)


def run(rate_hz: float, duration_s: float, *, sample_s: float = 1.0, warmup_s: float = 1.0) -> None:
    interval_ms = 1000.0 / rate_hz
    sim = Simulation(body=TwoFingerBody(config=calibrated_body()))
    dt = sim.config.dt_s
    half_steps = max(1, round((interval_ms / 2.0) / (dt * 1000.0)))
    warmup_steps = round(warmup_s / dt)
    measure_steps = round(duration_s / dt)
    sample_steps = max(1, round(sample_s / dt))

    presses = 0
    expected = 0.0
    previous_expected = 0.0
    previous_presses = 0

    print(f"rate={rate_hz:.2f} KPS  duration={duration_s:.1f}s  warmup={warmup_s:.1f}s")
    print(" time    actual  expected  ratio    fatigue   activation   position_mm")
    print("---------------------------------------------------------------------")

    total_steps = warmup_steps + measure_steps
    for step in range(total_steps):
        phase = (step // half_steps) % 2
        left = 1.0 if phase == 0 else -1.0
        result = sim.step(left, 0.0)

        if step < warmup_steps:
            continue

        measure_index = step - warmup_steps + 1
        presses += sum(1 for _, event in result.events if event is KeyEvent.DOWN)
        expected = measure_index * dt * rate_hz

        if measure_index % sample_steps == 0 or measure_index == measure_steps:
            window_actual = presses - previous_presses
            window_expected = expected - previous_expected
            ratio = window_actual / window_expected if window_expected else 0.0
            state = sim.body.left
            print(
                f"{measure_index*dt:5.1f}s  {window_actual:7d}  {window_expected:8.2f}  "
                f"{ratio:6.3f}   {state.fatigue:8.5f}   {state.activation:10.5f}   "
                f"{state.position_m*1000.0:10.4f}"
            )
            previous_presses = presses
            previous_expected = expected

    total_expected = duration_s * rate_hz
    print("---------------------------------------------------------------------")
    print(f"total: actual={presses}, expected={total_expected:.2f}, ratio={presses/total_expected:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect fatigue and missed presses at fixed single-finger rates.")
    parser.add_argument("--rates", nargs="+", type=float, default=[8.0, 8.5, 9.0])
    parser.add_argument("--durations", nargs="+", type=float, default=[5.0, 20.0])
    parser.add_argument("--sample", type=float, default=1.0)
    args = parser.parse_args()

    for duration in args.durations:
        for rate in args.rates:
            run(rate, duration, sample_s=args.sample)
            print()


if __name__ == "__main__":
    main()
