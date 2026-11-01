from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from a001_interval_frontier_diagnostic import SINGLE_LEFT, ALTERNATING, report_frontier


def test_v1_winner_grid_has_same_fixed_periods_for_both_arms():
    assert [r[0] for r in SINGLE_LEFT] == [r[0] for r in ALTERNATING]
    assert len(SINGLE_LEFT) == 13


def test_posthoc_frontier_preserves_all_target_criterion():
    report = report_frontier()
    assert report["status"].startswith("unapproved")
    assert report["target_conditions"]["all_targets_are_scored_in_primary_result"]
    assert len(report["rows"]) == 26
    for row in report["rows"]:
        assert row["targets"] >= 8
        assert row["total_hits"] <= row["targets"]
        assert row["hit_rate_all"] == row["total_hits"] / row["targets"]
