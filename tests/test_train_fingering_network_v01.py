from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_fingering_network_v01 as experiment
import finalize_fingering_network_v01 as finalizer

from dmdod.training import fingering_network_v01 as network


def test_fixed_splits_are_whole_series_and_disjoint():
    groups = [
        {pattern.sha256 for pattern in experiment.TRAIN_PATTERNS},
        {pattern.sha256 for pattern in experiment.VALIDATION_PATTERNS},
        {pattern.sha256 for pattern in experiment.TEST_PATTERNS},
    ]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    families = {pattern.family for pattern in experiment.TRAIN_PATTERNS}
    assert {"alternating", "rolling", "short_repeat", "high_density"} <= families


def test_teacher_collection_keeps_scores_out_of_network_context():
    pattern = network.PatternSpec("tiny", (50.0, 35.0), (1, 1, 1), "rolling")
    rows = experiment.collect_pattern(pattern)
    assert len(rows) == 3
    assert all(row.pattern_sha256 == pattern.sha256 for row in rows)
    assert all(row.context.shape == (network.CONTEXT_DIM,) for row in rows)
    assert all(row.candidates.shape[1] == network.CANDIDATE_DIM for row in rows)
    assert any(float(row.teacher_costs[row.eligible].max()) > 0 for row in rows)


def test_three_physical_policies_share_fixture_and_network_viewer_has_candidates():
    segment, _ = experiment._fixture(900.0)
    model = network.FingeringRanker(hidden=16).eval()
    center = experiment.run_policy(segment, "center_first")
    oracle = experiment.run_policy(segment, "teacher_v0_2")
    learned = experiment.run_policy(segment, "network_v0_1", model)
    assert center["targets"] == oracle["targets"] == learned["targets"]
    assert set(center) >= {"hits", "misses", "too_early", "overloaded",
                           "x_accuracy_percent", "timing_mae_ms", "physical_keydowns"}
    assert learned["decisions"] and all("teacher_chosen" in row for row in learned["decisions"])
    assert all(len(row["candidates"]) == 10 for row in learned["decisions"])
    report = {"physical_evaluation": {"cases": [{
        "bpm": 900.0, "network_v0_1": learned,
    }]}}
    html = experiment.render_viewer(report)
    assert "12物理軸" in html and "Network" in html and "Teacher" in html
    assert "実KeyDown" in html and "左手・親指" in html


def test_finalizer_keeps_nearest_keydown_per_selected_physical_axis():
    raw = {"physical_evaluation": {"cases": [{"network_v0_1": {"decisions": [{
        "due_ms": 100.0, "outcome": "Hit", "axes": ["left_1"],
        "keydowns": [
            {"time_ms": 95.0, "physical_axis": "right_1"},
            {"time_ms": 97.0, "physical_axis": "left_1"},
            {"time_ms": 104.0, "physical_axis": "left_1"},
        ],
    }]}}]}}
    row = finalizer.correct_viewer_associations(raw)["physical_evaluation"]["cases"][0]["network_v0_1"]["decisions"][0]
    assert row["keydowns"] == [{"time_ms": 97.0, "physical_axis": "left_1"}]
    assert row["judgement"] == "Hit" and row["too_early_keydowns"] == 0


def test_checkpoint_format_is_distinct_from_motor_and_timing_students():
    assert network.FORMAT == "dmdod-fingering-network-v0.1-candidate-ranker"
    assert network.MAX_EVENTS == 3
    assert network.GOOD_COST_TOLERANCE > 0
    assert network.CONTEXT_DIM == network.MOTOR_FEATURE_DIM + 3 * network.MAX_EVENTS
