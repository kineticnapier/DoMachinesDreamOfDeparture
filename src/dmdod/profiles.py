from __future__ import annotations

from dataclasses import replace

from .body import BilateralConfig, BodyConfig, FingerConfig, HandConfig


PERSONAL_BLUE_SWITCH_V0_1_NAME = "personal-blue-switch-v0.1"


def personal_blue_switch_v0_1(*, same_hand: bool = True) -> BodyConfig:
    """Return the frozen v0.1 body profile fitted to the initial measurements.

    The values are a reproducible experimental profile, not universal human
    constants.  Keeping them here prevents later RL experiments from silently
    changing the calibrated body while controllers/policies are being compared.
    """

    finger = FingerConfig(
        mass_kg=0.020,
        damping_n_s_m=0.630000,
        spring_n_m=35.0,
        rest_position_m=0.0,
        max_force_n=2.600000,
        activation_tau_s=0.280000,
        fatigue_gain_s=0.020000,
        fatigue_recovery_s=0.028000,
        fatigue_threshold=0.315000,
        fatigue_exponent=2.0,
        switch_fatigue_per_reversal=0.00100000,
        min_position_m=-0.001,
        max_position_m=0.006,
    )

    hand = HandConfig(
        capacity=0.472500,
        fatigue_gain_s=0.030000,
        fatigue_recovery_s=0.250000,
        fatigue_threshold=0.55,
        fatigue_exponent=2.0,
        switch_tau_s=0.054000,
        coordination_floor=0.160000,
    )

    bilateral = BilateralConfig(
        switch_tau_s=0.040000,
        coordination_floor=0.600000,
    )

    return BodyConfig(
        left=finger,
        right=replace(finger),
        hand=hand,
        bilateral=bilateral,
        same_hand=same_hand,
    )
