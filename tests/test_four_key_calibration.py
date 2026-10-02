import pytest

from dmdod.four_key_calibration import calibrate_four_key_press_lead
from dmdod.four_key_motor import FOUR_KEY_NAMES


def test_four_key_calibration_measures_every_key_and_center_first_lead() -> None:
    calibration = calibrate_four_key_press_lead(
        control_dt_s=0.010,
        physics_dt_s=0.001,
    )

    latencies = [calibration.latency_for(key) for key in FOUR_KEY_NAMES]
    assert all(latency > 0.0 for latency in latencies)
    assert calibration.center_press_latency_s == pytest.approx(
        max(
            calibration.left_inner_press_latency_s,
            calibration.right_inner_press_latency_s,
        )
    )
    assert calibration.lead_s == pytest.approx(
        calibration.center_press_latency_s + 0.005
    )


def test_four_key_calibration_is_symmetric_from_rest_with_frozen_profile() -> None:
    calibration = calibrate_four_key_press_lead(
        control_dt_s=0.010,
        physics_dt_s=0.001,
    )

    reference = calibration.left_inner_press_latency_s
    for key in FOUR_KEY_NAMES:
        # All four fingers currently share the frozen v0.1 physical constants.
        # Independent fresh-env measurements should therefore agree to one
        # physics tick.  If later finger-specific profiles are introduced this
        # test should be replaced by explicit per-finger expected ranges.
        assert calibration.latency_for(key) == pytest.approx(reference, abs=0.001)


def test_four_key_calibration_rejects_unknown_key_lookup() -> None:
    calibration = calibrate_four_key_press_lead()
    with pytest.raises(KeyError):
        calibration.latency_for("middle")


def test_four_key_calibration_fails_cleanly_when_wait_budget_is_too_short() -> None:
    with pytest.raises(RuntimeError, match="produced no"):
        calibrate_four_key_press_lead(
            control_dt_s=0.010,
            physics_dt_s=0.001,
            max_wait_s=0.001,
        )
