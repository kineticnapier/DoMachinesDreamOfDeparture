"""D23 archived D22 Motor state audit contracts; synthetic data only."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import a002v2_l0_d23_archived_motor_state as d23


def make_step(index, name):
    t = index / 100
    baseline_action = 2 if name.startswith("after") else 0
    sampled = baseline_action
    action = sampled
    overridden = index == 362 and name in ("before_to_left", "after_to_none")
    if overridden:
        action = 2 if name == "before_to_left" else 0
    position = 0.001
    if index >= 363 and name in ("before_to_left", "after", "after_negative"):
        position = 0.0015
    return {
        "step": index, "time_s": t, "end_time_s": (index + 1) / 100,
        "observation": [0.0] * 8,
        "action_probabilities": [0.25] * 4,
        "action": action, "sampled_action": sampled,
        "intervention_applied": overridden,
        "pre_motor": {
            "left_position_m": position,
            "left_velocity_m_s": position * 10,
            "left_pressed": False,
        },
        "pre_diagnostics": {
            "time_s": t, "left_activation": position,
            "left_fatigue": 0.01, "hand_fatigue": 0.02,
            "hand_coordination": 0.0,
        },
        "emitted_keydown_indices": [],
    }


def make_trace(name):
    steps = [make_step(i, name) for i in range(480)]
    if name in ("before", "before_negative"):
        specs = [(3.718, True), (4.019, False)]
    elif name in ("after_to_none",):
        specs = [(3.718, True), (4.003, False), (4.324, False)]
    else:
        specs = [(3.689, False), (4.003, False), (4.324, False)]
    events = []
    for i, (time, hit) in enumerate(specs):
        source = int(time * 100)
        event = {"time_s": time, "key": "left", "hit": hit}
        events.append({
            "event_index": i, "source_step": source, "event": event,
        })
        steps[source]["emitted_keydown_indices"].append(i)
    controls = [sum(s["action"] == i for s in steps) for i in range(4)]
    sampled = [sum(s["sampled_action"] == i for s in steps) for i in range(4)]
    return {
        "steps": steps,
        "physical_keydowns": events,
        "row": {
            "hits": sum(x[1] for x in specs),
            "extra_keydowns": sum(not x[1] for x in specs),
            "physical_keydowns": len(specs),
            "steps": 480, "controls": controls,
            "sampled_controls": sampled,
            "physical_trace": [e["event"] for e in events],
        },
    }


def fixture():
    traces = {name: make_trace(name) for name in d23.SERIES}
    # Ensure the archived negative controls are byte-for-byte identical.
    traces["before_negative"] = deepcopy(traces["before"])
    traces["after_negative"] = deepcopy(traces["after"])
    metrics = {}
    for name, trace in traces.items():
        row = trace["row"]
        first = trace["physical_keydowns"][0]["event"]
        metrics[name] = {
            "hits": row["hits"], "extra_keydowns": row["extra_keydowns"],
            "physical_keydowns": row["physical_keydowns"],
            "controls": row["controls"],
            "sampled_controls": row["sampled_controls"],
            "first_physical": {"time_s": first["time_s"],
                               "hit": first["hit"]},
        }
    result = {
        "schema": d23.D22_SCHEMA,
        "classification": "valid_single_step_intervention",
        "supports_local_command_contribution": True,
        "source_d21_run": 41505006857,
        "source_d19_run": 41485667328,
        "new_motor_episodes": 6, "new_physical_keydowns": 16,
        "intervention_step": 362, "arm": "a0",
        "case_index": 4, "repetition": 3, "sampling_seed": 320403,
        "metrics": metrics,
        "first_keydown_shift_ms": {
            "before_to_left_minus_before": -29.0,
            "after_to_none_minus_after": 29.0,
        },
    }
    return result, traces


class TestD23(unittest.TestCase):
    def test_frozen_conditions(self):
        self.assertEqual(d23.INTERVENTION_STEP, 362)
        self.assertEqual(d23.ACTUATION_M, 0.002)
        self.assertEqual(d23.SNAPSHOT_STEPS,
                         (362, 363, 365, 368, 369, 371, 372))

    def test_valid_fixture(self):
        source, traces = fixture()
        output = d23.analyze(source, traces)
        self.assertEqual(output["classification"],
                         "valid_archived_motor_state_trace")
        self.assertEqual(output["new_motor_episodes"], 0)
        self.assertEqual(output["new_physical_keydowns"], 0)
        self.assertTrue(output["early_trace_closer_to_threshold_at_step368"])
        self.assertTrue(output["reverse_trace_farther_from_threshold_at_step368"])
        self.assertEqual(output["pairs"]["before_to_left"][
            "first_state_differences"]["position_m"]["step"], 363)
        self.assertAlmostEqual(output["pairs"]["before_to_left"][
            "first_keydown"]["delta_ms"], -29)

    def test_reject_negative_control_change(self):
        source, traces = fixture()
        traces["before_negative"]["steps"][40]["action"] = 3
        with self.assertRaisesRegex(ValueError, "negative control"):
            d23.analyze(source, traces)

    def test_reject_missing_series(self):
        source, traces = fixture()
        del traces["after_to_none"]
        with self.assertRaisesRegex(ValueError, "trace keys"):
            d23.analyze(source, traces)

    def test_reject_modified_pre_intervention_step(self):
        source, traces = fixture()
        traces["before_to_left"]["steps"][361]["observation"][0] = 0.1
        with self.assertRaisesRegex(ValueError, "provenance"):
            d23.analyze(source, traces)

    def test_reject_wrong_step362_action(self):
        source, traces = fixture()
        traces["before_to_left"]["steps"][362]["action"] = 0
        with self.assertRaisesRegex(ValueError, "provenance"):
            d23.analyze(source, traces)

    def test_reject_missing_physical_event(self):
        source, traces = fixture()
        traces["after_to_none"]["physical_keydowns"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            d23.analyze(source, traces)

    def test_reject_nonfinite_position(self):
        source, traces = fixture()
        traces["before_to_left"]["steps"][368]["pre_motor"]["left_position_m"] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            d23.analyze(source, traces)

    def test_reject_bad_clock(self):
        source, traces = fixture()
        traces["before_to_left"]["steps"][370]["end_time_s"] = 3.9
        with self.assertRaisesRegex(ValueError, "step clock"):
            d23.analyze(source, traces)

    def test_negative_threshold_outcome_is_valid(self):
        source, traces = fixture()
        traces["before_to_left"]["steps"][368]["pre_motor"]["left_position_m"] = 0.0008
        output = d23.analyze(source, traces)
        self.assertEqual(output["classification"],
                         "valid_archived_motor_state_trace")
        self.assertFalse(output["early_trace_closer_to_threshold_at_step368"])


if __name__ == "__main__":
    unittest.main()
