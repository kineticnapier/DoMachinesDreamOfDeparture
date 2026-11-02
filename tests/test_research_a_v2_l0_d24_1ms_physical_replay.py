"""D24 tests: independent 1ms Motor replay and strict archived parity."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from dmdod.motor.env import MotorAction, MotorEnv
from dmdod.motor.keyboard import KeyEvent
from dmdod.research.a002_timing_only_env import ACTION_COMMANDS
import a002v2_l0_d24_1ms_physical_replay as d24


def synthetic_fast_trace(controls=80):
    env = MotorEnv(same_hand=True, control_dt_s=0.010)
    env.reset()
    steps = []
    physical = []
    for index in range(controls):
        action = 2 if index < controls // 2 else 0
        desired = ACTION_COMMANDS[action]
        steps.append({
            "step": index, "time_s": env.sim.time_s, "action": action,
            "pre_motor": asdict(env.observe()),
            "pre_diagnostics": asdict(env.diagnostics()),
        })
        outcome = env.step(MotorAction(desired.left, -1.0))
        for event in outcome.evaluator_events:
            if event.event is KeyEvent.DOWN:
                physical.append({"time_s": round(event.time_s, 6),
                                 "key": event.key})
    return {"steps": steps, "row": {
        "physical_trace": physical, "hits": 0,
        "extra_keydowns": len(physical),
    }}


class TestD24(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trace = synthetic_fast_trace()
        assert cls.trace["row"]["physical_trace"], "synthetic KeyDown absent"

    def test_preregistered_limits(self):
        self.assertEqual((d24.DT, d24.SUBSTEPS, d24.CONTROLS),
                         (0.001, 10, 480))
        self.assertEqual(d24.FLOAT_TOL, 1e-9)
        self.assertEqual(d24.TIME_TOL, 1e-8)
        self.assertEqual(d24.SOURCE_RUN, 41515190995)

    def test_fast_and_slow_motor_paths_agree(self):
        metrics, window, ticks = d24.replay_series(self.trace, name="synthetic")
        self.assertEqual(metrics["ticks"], 800)
        self.assertEqual(metrics["keydown_count"],
                         len(self.trace["row"]["physical_trace"]))
        self.assertEqual(len(ticks), 800)
        self.assertEqual(window, [])
        self.assertTrue(all(x <= d24.FLOAT_TOL
                            for x in metrics["max_abs_control_state_difference"].values()))

    def test_replay_deterministic(self):
        first = d24.replay_series(self.trace, name="synthetic")
        second = d24.replay_series(self.trace, name="synthetic")
        self.assertEqual(first, second)

    def test_source_is_read_only(self):
        before = deepcopy(self.trace)
        d24.replay_series(self.trace, name="synthetic")
        self.assertEqual(before, self.trace)

    def test_wrong_archived_position_rejected(self):
        trace = deepcopy(self.trace)
        trace["steps"][4]["pre_motor"]["left_position_m"] += 0.0001
        with self.assertRaisesRegex(ValueError, "left_position_m"):
            d24.replay_series(trace, name="synthetic")

    def test_nonfinite_archived_state_rejected(self):
        trace = deepcopy(self.trace)
        trace["steps"][5]["pre_diagnostics"]["left_activation"] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            d24.replay_series(trace, name="synthetic")

    def test_bad_action_rejected(self):
        trace = deepcopy(self.trace)
        trace["steps"][2]["action"] = 4
        with self.assertRaisesRegex(ValueError, "invalid action"):
            d24.replay_series(trace, name="synthetic")

    def test_wrong_keydown_time_rejected(self):
        trace = deepcopy(self.trace)
        trace["row"]["physical_trace"][0]["time_s"] += 0.001
        with self.assertRaisesRegex(ValueError, "KeyDown mismatch"):
            d24.replay_series(trace, name="synthetic")

    def test_first_keydown_crosses_threshold(self):
        metrics, _, _ = d24.replay_series(self.trace, name="synthetic")
        first = metrics["first_keydown"]
        self.assertLess(first["before_tick"]["left_position_m"], 0.002)
        self.assertGreaterEqual(first["after_tick"]["left_position_m"], 0.002)
        self.assertFalse(first["before_tick"]["left_pressed"])
        self.assertTrue(first["after_tick"]["left_pressed"])


if __name__ == "__main__":
    unittest.main()
