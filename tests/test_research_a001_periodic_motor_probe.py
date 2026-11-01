"""Small, reproducible tests for exploratory periodic Motor probe.

These tests deliberately do not assert that 001 is completed, and do not
treat periodic expert controls as the student policy.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from a001_periodic_motor_probe import (
    Trial, match_keydowns, periodic_command_at, rank_trial, run_trial,
    sweep, target_times,
)


def test_single_finger_controls_never_request_second_finger():
    trial = Trial("single_left", 120, .4, 50, .8, 1.2)
    targets = target_times(trial)
    for step in range(200):
        assert periodic_command_at(trial, step, targets).right == -1.0


def test_alternating_controls_use_both_axes_and_separate_duty_cycles():
    trial = Trial("alternating", 200, .25, 0, 1.0, 1.2)
    targets = target_times(trial)
    assert periodic_command_at(trial, 100, targets).left == 1.0  # 1.00s
    assert periodic_command_at(trial, 100, targets).right == -1.0
    assert periodic_command_at(trial, 120, targets).right == 1.0  # 1.20s
    assert periodic_command_at(trial, 120, targets).left == -1.0
    assert periodic_command_at(trial, 135, targets) == type(periodic_command_at(trial, 0, targets))(-1., -1.)


def test_ordered_event_matching_does_not_reuse_keydowns():
    due = [1.0, 1.10, 1.20]
    s = match_keydowns([(0.999, "left"), (1.002, "right"),
                        (1.101, "right")], due, window_s=.020)
    assert (s["hits"], s["misses"], s["extra_keydowns"]) == (2, 1, 1)
    assert s["wrong_axis_for_alternating_reference"] == 0


def test_out_of_window_strokes_cannot_count_as_hits():
    s = match_keydowns([(0.90, "left"), (1.03, "left")], [1.0], window_s=.020)
    assert s["hits"] == 0
    assert s["misses"] == 1


def test_fixed_trial_is_reproducible():
    trial = Trial("single_left", 220, .55, 170, .8, 1.2)
    assert run_trial(trial) == run_trial(trial)


def test_screen_and_confirm_are_distinct_start_conditions():
    trial = sweep(
        intervals_ms=(180,), duties=(.4,), leads_ms=(50,),
        modes=("single_left", "alternating"),
        screen_duration_s=1.2, confirm_duration_s=1.6,
    )
    assert len(trial["rows"]) == 2
    assert trial["rows"][0]["screen"]["start_s"] != trial["rows"][0]["confirm"]["start_s"]
    assert trial["status"].startswith("UNAPPROVED")


def test_rank_penalizes_extraneous_keydowns():
    a = {"clean_95": False, "hits": 10, "targets": 12,
         "extra_keydowns": 0, "lead_ms": 50, "duty": .4}
    b = {**a, "extra_keydowns": 3}
    assert rank_trial(a) > rank_trial(b)


@pytest.mark.parametrize("mode", ("invalid", "single_left"))
def test_invalid_configuration_rejected(mode):
    trial = Trial(mode, 180, 0 if mode == "single_left" else .5, 0, .8, 1.2)
    with pytest.raises(ValueError):
        run_trial(trial)
