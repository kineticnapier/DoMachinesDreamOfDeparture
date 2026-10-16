from __future__ import annotations

from dataclasses import replace

import pytest

from dmdod.motor.capacity import calibrate_n_key_press_lead
from dmdod.training import prestige1 as p
from dmdod.training import prestige1_autoplay_persistent_motor_audit as baseline
from dmdod.training import prestige1_autoplay_persistent_motor_endurance_audit as endurance


@pytest.fixture(scope="module")
def calibration():
    return calibrate_n_key_press_lead(8, control_dt_s=.010, physics_dt_s=.001)


@pytest.mark.parametrize("bpm,notes,expected_hits", [
    (1200., 72, 72), (2000., 72, 72), (4000., 72, 61),
    (1200., 144, 144), (2000., 144, 144), (4000., 144, 124),
])
def test_unmodified_persistent_planner_endurance_results(bpm, notes, expected_hits, calibration):
    result = endurance.analyze_case(bpm, notes, calibration)
    assert result["targets"] == notes
    assert result["hits"] == expected_hits
    assert result["misses"] == notes-expected_hits
    assert result["motor_resets_after_start"] == 0
    assert len(result["all_target_outcomes"]) == notes
    assert result["hits"] == sum(t["hit"] for t in result["all_target_outcomes"])
    assert result["misses"] == sum(t["miss"] for t in result["all_target_outcomes"])
    assert result["physical_keydowns"] == len(result["physical_keydown_trace"])


def test_4000_18_misses_have_no_hud_plan_due_to_end_horizon(calibration):
    item = endurance.analyze_case(4000., 18, calibration)
    assert item["hits"] == 15
    assert item["miss_attribution"] == {"no_plan": 3}
    assert [t["ordinal"] for t in item["first_misses"]] == [16,17,18]
    assert item["hud_statuses"].get("insufficient_lookahead",0) > 0


def test_4000_72_misses_separate_no_plan_from_keydown_target_mismatch(calibration):
    item = endurance.analyze_case(4000., 72, calibration)
    assert item["miss_attribution"] == {
        "keydown_evaluator_target_mismatch": 7,
        "no_plan": 4,
    }
    for miss in item["first_misses"][:7]:
        assert miss["planned_key"] is not None
        assert miss["matched_physical_down_ms"] is not None
        assert miss["physical_event_evaluator_target_index"] != miss["ordinal"]-1
        assert miss["physical_down_minus_intended_ms"] > 0
    assert [t["ordinal"] for t in item["first_misses"][-4:]] == [69,70,71,72]


def test_4000_144_miss_attribution_and_no_finger_reset(calibration):
    item = endurance.analyze_case(4000., 144, calibration)
    assert item["miss_attribution"] == {
        "keydown_evaluator_target_mismatch": 16,
        "no_plan": 4,
    }
    assert item["motor_resets_after_start"] == 0
    assert item["hud_statuses"]["eligible"] > 0


def test_18_note_reference_results_agree_with_previous_fixture(calibration):
    for bpm in (1200., 2000., 4000.):
        old=baseline.simulate(bpm)
        new=endurance.analyze_case(bpm, 18, calibration)
        assert old["evaluator_hits"] == new["hits"]
        assert old["evaluator_misses"] == new["misses"]
        assert old["unique_commitments"] == new["planned_unique"]
        assert old["physical_keydowns"] == new["physical_keydowns"]


def test_reject_unfixed_fixture(calibration):
    with pytest.raises(ValueError,match="fixed"):
        endurance.analyze_case(4000., 100, calibration)


def test_report_refuses_overwrite_and_preserves_prestige_contract(tmp_path):
    config=replace(p.PrestigeConfig.load('configs/training/prestige1.toml'),
                   output_dir=str(tmp_path))
    report=endurance.run_audit(config)
    assert len(report["cases"]) == len(endurance.FIXTURES) == 9
    assert report["no_training"] is True
    assert report["no_student_checkpoint"] is True
    assert report["original_stage_a_fail_preserved"] is True
    assert report["validation_evaluated"] is False
    assert report["final_evaluated"] is False
    assert report["policy_dt_ms"] == 10
    assert report["physics_dt_ms"] == 1
    output=tmp_path/'diagnostics'/'prestige1'/endurance.REPORT
    original=output.read_bytes()
    with pytest.raises(FileExistsError):
        endurance.run_audit(config)
    assert output.read_bytes() == original
