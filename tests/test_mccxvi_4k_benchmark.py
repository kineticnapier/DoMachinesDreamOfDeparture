"""MCCXVI phase-local static key-demand tests; no chart or audio required in CI."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mccxvi_4k_benchmark import Event, demand_window, min_k_spacing, compute_report


class MCCXVIPreflightTests(unittest.TestCase):
    def test_four_keys_80ms_possible_but_not_100ms(self):
        events = [Event(i, i * (60 / 2432), 1) for i in range(40)]
        self.assertEqual(demand_window(events, .080)[0], 4)
        self.assertEqual(demand_window(events, .100)[0], 5)
        self.assertEqual(demand_window(events, .150)[0], 7)
        self.assertAlmostEqual(
            min_k_spacing(events, 4)["critical_key_reuse_upper_bound_ms"],
            4 * 60 / 2432 * 1000, places=4
        )

    def test_two_keys_reuse_at_about_98ms(self):
        events = [Event(i, i * (60 / 1216), 1) for i in range(20)]
        self.assertEqual(demand_window(events, .080)[0], 2)
        self.assertEqual(demand_window(events, .100)[0], 3)

    def test_exact_cooldown_boundary_permits_reuse(self):
        events = [Event(1, 1), Event(2, 1.10), Event(3, 1.20)]
        self.assertEqual(demand_window(events, .10)[0], 1)
        self.assertEqual(demand_window(events, .100001)[0], 2)

    def test_chords_increase_key_demand(self):
        events = [Event(1, 1, 2), Event(2, 1.01, 2)]
        demanded, witness = demand_window(events, .080)
        self.assertEqual(demanded, 4)
        self.assertEqual(witness["first_floor"], 1)

    def test_original_chart_optional_exploratory_preflight(self):
        # User-supplied copyrighted chart is never stored in the repository.
        # Set MCCXVI_ZIP to the original local ZIP to enable integration tests.
        supplied = os.environ.get("MCCXVI_ZIP")
        path = Path(supplied) if supplied else Path("MCCXVI - JinoBeats.zip")
        if not path.is_file():
            self.skipTest("original MCCXVI ZIP not supplied; set MCCXVI_ZIP")
        report = compute_report(path, timing="approximate")
        self.assertEqual(report["angle_count"], 1836)
        self.assertEqual(report["compiled"]["playable_target_count"], 1820)
        self.assertEqual(
            [phase["modeled_key_limit"] for phase in report["phases"]],
            [None, 4, 2, None, 2, 4, 2, 4, 2, 4]
        )
        self.assertEqual(report["limit_markers"][-1]["floor"], 1776)
        for phase in report["phases"]:
            if phase["modeled_key_limit"] is not None:
                trial = next(item for item in phase["cooldown_tests"] if item["cooldown_ms"] == 100)
                self.assertGreater(
                    trial["minimum_keys_demanded"], phase["modeled_key_limit"]
                )

    def test_official_timing_optional_integration(self):
        supplied = os.environ.get("MCCXVI_ZIP")
        if not supplied or not Path(supplied).is_file():
            self.skipTest("official DMDOD integration requires MCCXVI_ZIP")
        report = compute_report(Path(supplied), timing="official")
        self.assertEqual(report["compiled"]["timing_authority"], "dmdod_stock_compile")
        self.assertEqual(report["compiled"]["playable_target_count"], 1820)


if __name__ == "__main__":
    unittest.main()
