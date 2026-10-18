from __future__ import annotations

from pathlib import Path

from scripts import diagnose_hud_v14_saved_integration_v02 as diagnostic
from scripts import run_hud_v14_fingering_motor_integration_v02 as runner


def test_old_summary_never_fabricates_actual_missed_ordinals():
    segment = runner.synthetic_segment(120.)
    case = {
        "format": runner.FORMAT,
        "cases": [{
            "label": "ordinary-120bpm",
            "arms": {
                "hud_network": {
                    "stats": {"hits": len(segment.targets)-1,
                              "targets": len(segment.targets), "misses": 1},
                    "first_seen_predictions": [],
                    "decisions": [],
                }
            },
        }],
    }
    rows = diagnostic.diagnose(
        case, train_chart=runner.DEFAULT_TRAIN_CHART, train_end_s=5.
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["actual_missed_ordinals_posthoc"] == "unknown_in_r1_summary"
    assert len(row["timing_without_20ms_forecast_ordinals"]) == len(segment.targets)


def test_enriched_summary_preserves_known_posthoc_miss():
    segment = runner.synthetic_segment(240.)
    first_due = segment.targets[0].episode_time_s
    record = {
        "format": runner.FORMAT,
        "cases": [{
            "label": "ordinary-240bpm",
            "arms": {
                "hud_network": {
                    "stats": {"hits": len(segment.targets)-1,
                              "targets": len(segment.targets), "misses": 1},
                    "first_seen_predictions": [
                        {"local_id": 0, "due_local_s": first_due,
                         "first_seen_s": max(0., first_due-.06)}
                    ],
                    "decisions": [
                        {"local_id": 0, "reason": "assigned"}
                    ],
                    "miss_diagnostics_posthoc": [{
                        "target_ordinal": 0, "category": "assigned_but_scorer_missed_check_physical_trace",
                        "nearest_local_forecast": {"local_id": 0},
                        "decision": {"choice": "right_1"}
                    }],
                }
            }
        }],
    }
    result = diagnostic.diagnose(
        record, train_chart=runner.DEFAULT_TRAIN_CHART, train_end_s=5.
    )
    assert result[0]["actual_missed_ordinals_posthoc"] == [
        {"ordinal": 0, "category": "assigned_but_scorer_missed_check_physical_trace",
         "nearest": {"local_id": 0}, "decision": {"choice": "right_1"}}
    ]
