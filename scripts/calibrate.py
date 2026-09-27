import argparse

from dmdod.fitting import CalibrationTargets, fit_body_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=("same-hand", "cross-hand"),
        default="same-hand",
        help="same-hand = RI/RM target (12.5 KPS), cross-hand = RI/LI target (16 KPS)",
    )
    args = parser.parse_args()

    targets = CalibrationTargets()
    print("=== Human Calibration v0.1 ===")
    print(f"profile: {args.profile}")
    print("Fitting provisional Finger -> Hand -> Body parameters...")
    print()

    result = fit_body_config(targets, progress=lambda message: print(f"  {message}", flush=True), profile=args.profile)
    rows = (
        ("RI single (short)", targets.short_single_hz, result.short_single_hz),
        ("RI single (20 s)", targets.sustained_single_hz, result.sustained_single_hz),
        (f"2-finger ({result.profile})", result.alternate_target_hz, result.alternate_hz),
    )

    print()
    print(f"{'Test':28s} {'Human':>9s} {'Sim':>9s} {'Error':>9s}")
    print("-" * 58)
    for name, human_rate, sim_rate in rows:
        print(f"{name:28s} {human_rate:8.2f}K {sim_rate:8.2f}K {sim_rate-human_rate:+8.2f}K")

    c = result.config
    print()
    print(f"relative loss: {result.loss:.6f}")
    print("fitted provisional parameters:")
    print(f"  activation_tau_s       = {c.left.activation_tau_s:.6f}")
    print(f"  max_force_n            = {c.left.max_force_n:.6f}")
    print(f"  damping_n_s_m          = {c.left.damping_n_s_m:.6f}")
    print(f"  finger_fatigue_gain_s  = {c.left.fatigue_gain_s:.6f}")
    print(f"  finger_recovery_s      = {c.left.fatigue_recovery_s:.6f}")
    print(f"  same_hand              = {c.same_hand}")
    if c.same_hand:
        print(f"  hand_capacity          = {c.hand.capacity:.6f}")
        print(f"  hand_fatigue_gain_s    = {c.hand.fatigue_gain_s:.6f}")
        print(f"  hand_recovery_s        = {c.hand.fatigue_recovery_s:.6f}")
    print()
    print("Measurement excludes the first 1 s warmup.")
    print("These values fit the initial blue-switch calibration profile; they are not universal human constants.")


if __name__ == "__main__":
    main()
