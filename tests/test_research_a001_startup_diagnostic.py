from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from a001_startup_diagnostic import assign_events, run_diagnostic
from a001_periodic_motor_probe import Trial


def test_late_start_miss_is_recorded_as_first_ordinal_not_last():
    matches, extras = assign_events([1., 1.2, 1.4], [(1.201, "left"), (1.4, "right")])
    assert [i for i, x in enumerate(matches) if x is None] == [0]
    assert extras == []


def test_extra_keydowns_are_not_hidden_in_startup_diagnostic():
    matches, extras = assign_events([1., 1.2], [(1.00, "left"), (1.01, "right"), (1.20, "right")])
    assert all(matches)
    assert len(extras) == 1


def test_frozen_pilot_case_diagnosis_is_reproducible():
    trial = Trial("single_left", 500, .4, 170, 1.130, 1.600)
    x = run_diagnostic(trial)
    y = run_diagnostic(trial)
    assert x == y
    assert len(x["first_12_assignments"]) == x["target_count"]
    assert x["start_sensitivity"][0]["remaining_targets"] == x["target_count"]
