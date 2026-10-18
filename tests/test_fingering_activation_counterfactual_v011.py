from __future__ import annotations

import copy

import pytest

from dmdod.motor.contact_16k import Coupled16ContactMotor
from scripts import audit_fingering_activation_counterfactual_v011 as audit


def _onset_report():
    def event(index, band, activation):
        return {
            "event_index": index,
            "reuse_band": band,
            "activation_at_command": activation,
            "axis": "right_1",
        }
    return {
        "policies": {
            "free": {
                "events": [
                    event(1, "first_use", 0.0),
                    event(2, "reuse_100_200ms", -.1),
                    event(3, "reuse_100_200ms", -.2),
                    event(4, "reuse_100_200ms", -.3),
                    event(5, "reuse_200_500ms", -.05),
                    event(6, "reuse_ge_500ms", -.01),
                    event(7, "reuse_lt_100ms", -.3),
                    event(8, "reuse_200_500ms", .01),
                ]
            },
            "human_proxy": {"events": []},
        }
    }


def test_uniform_event_sampling_excludes_nonnegative_activation():
    selected = audit.choose_events(_onset_report(), "free", 2)
    assert set(selected) == {2, 4, 5, 6}


def test_empty_activation_group_is_not_artificially_filled():
    assert audit.choose_events(_onset_report(), "human_proxy", 4) == {}


def test_two_identical_motor_copies_have_only_one_modified_state():
    original = Coupled16ContactMotor()
    motor = original.motor
    idx = motor.key_names.index("right_1")
    motor.body.states[idx].activation = -0.12
    original.motor.body.states[idx].velocity_m_s = -0.055
    modified = copy.deepcopy(original)
    modified.motor.body.states[idx].activation = 0.0
    assert original.motor.body.states[idx].activation == pytest.approx(-.12)
    assert modified.motor.body.states[idx].activation == 0
    assert modified.motor.body.states[idx].velocity_m_s == pytest.approx(-.055)
    assert original.motor.body.states[idx].velocity_m_s == pytest.approx(-.055)
    assert modified.motor.body.states[0].activation == original.motor.body.states[0].activation


def test_sampler_never_selects_more_than_requested():
    selected = audit.choose_events(_onset_report(), "free", 1)
    assert len(selected) == 3  # one from each negative-activation reuse band
    assert 1 not in selected  # zero-activation first use is not an intervention
    assert 7 not in selected  # no observed <100ms physical reuse
    assert 8 not in selected  # positive activation is not zeroed
