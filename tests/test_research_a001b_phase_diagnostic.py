from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from a001b_posthoc_phase_diagnostic import diagnose, CASES, CONFIRM_STARTS_S


def test_frozen_phase_cases_are_observed_without_new_controller_search():
    assert len(CASES) == 5
    assert CONFIRM_STARTS_S == (1.137,1.273,1.419)


def test_physical_keydown_offsets_are_reproducible():
    a=diagnose("single_left", 140, .55, 50, 1.273)
    b=diagnose("single_left", 140, .55, 50, 1.273)
    assert a == b
    assert a["steady_targets"] > 0
    assert len(a["first_5_steady_targets_s"]) == 5
