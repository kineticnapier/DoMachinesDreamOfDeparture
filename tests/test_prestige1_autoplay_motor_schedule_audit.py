from __future__ import annotations

from dataclasses import replace

import pytest

from dmdod.motor.capacity import calibrate_n_key_press_lead
from dmdod.training import prestige1 as p
from dmdod.training import prestige1_autoplay_high_bpm_simulator_audit as simulator
from dmdod.training import prestige1_autoplay_motor_schedule_audit as m


@pytest.fixture(scope="module")
def calibration():
    return calibrate_n_key_press_lead(8,control_dt_s=.001,physics_dt_s=.001)


@pytest.fixture(scope="module")
def forecast_20k():
    segment,_=simulator._fixture(20000.,reverse=False)
    samples,_=simulator._capture(segment)
    return samples[10].hud


def test_physical_press_latency_calibrated_from_frozen_motor(calibration):
    values=dict(calibration.key_latencies_s)
    assert len(values)==8
    assert min(values.values())==pytest.approx(.039,abs=1e-9)
    assert max(values.values())==pytest.approx(.042,abs=1e-9)


def test_hud_plan_assigns_ordered_distinct_fingers_at_20k(forecast_20k,calibration):
    plan=m.plan_from_hud(forecast_20k,calibration)
    assert plan.status=="late_cold_start"
    assert [a.target_offset_s for a in plan.assignments]==pytest.approx(
        [.002,.005,.008],abs=2e-6)
    assert [a.floor_offset for a in plan.assignments]==[1,2,3]
    assert len({a.key for a in plan.assignments})==3
    assert all(a.command_start_s==0 for a in plan.assignments)
    assert all(a.cold_deadline_deficit_s>.03 for a in plan.assignments)


def test_actual_shadow_physics_is_not_magically_fast(forecast_20k,calibration):
    plan=m.plan_from_hud(forecast_20k,calibration)
    result=m.simulate_shadow_from_rest(plan)
    assert result["planned_events"]==3
    assert result["key_downs_within_10ms"]==0
    assert len(result["actual_keydowns_ms"])==3
    # Concurrent fingers interact through the coupled hand/body model;
    # cold single-finger lead is not an exact multi-key execution time.
    assert min(x["time_ms"] for x in result["actual_keydowns_ms"])>=39.0
    assert all(x>30 for x in result["timing_errors_ms"])
    assert result["no_frozen_student_control_change"] is True
    assert len(result["reference_1ms_micro_actions_during_first_10ms"])==10
    assert all(len(cmd)==8 for cmd in result["reference_1ms_micro_actions_during_first_10ms"])


def test_shadow_zero_event_plan_has_no_unrequested_presses(calibration):
    segment,_=simulator._fixture(600.,reverse=False)
    samples,_=simulator._capture(segment)
    plan=m.plan_from_hud(samples[0].hud,calibration)
    assert plan.eligible
    assert plan.assignments==()
    result=m.simulate_shadow_from_rest(plan)
    assert result["planned_events"]==0
    assert result["actual_keydowns_ms"]==[]


def test_hud_abstention_never_produces_motor_actions(calibration):
    segment,_=simulator._fixture(20000.,reverse=False)
    samples,_=simulator._capture(segment)
    plan=m.plan_from_hud(samples[30].hud,calibration)
    assert not plan.eligible
    assert plan.assignments==()
    with pytest.raises(ValueError):
        m.simulate_shadow_from_rest(plan)


def test_corrupted_hud_does_not_plan(calibration,forecast_20k):
    plan=m.plan_from_hud(forecast_20k[:20],calibration)
    assert not plan.eligible
    assert not plan.assignments


def test_20k_floor_lookahead_is_less_than_cold_motor_lead(calibration):
    # 12 ordinary 180-degree future floors: 12 * 60 / 20000 s == 36ms.
    assert 12 * 60 / 20000 * 1000 == pytest.approx(36)
    assert min(dict(calibration.key_latencies_s).values())>.036


def test_report_immutable_stage_t_and_frozen_time_contract(tmp_path):
    config=replace(p.PrestigeConfig.load("configs/training/prestige1.toml"),
                   output_dir=str(tmp_path))
    report=m.run_audit(config)
    assert len(report["results"])==26
    assert report["no_training"] is True
    assert report["validation_evaluated"] is False
    assert report["final_evaluated"] is False
    assert report["policy_control_dt_ms"]==10
    assert report["reference_motor_dt_ms"]==1
    r=next(x for x in report["results"] if x["bpm"]==20000 and x["direction"]=="cw")
    assert r["planned_multi_event_frames"]>0
    assert r["cold_deadline_deficits"]==r["planned_events"]
    assert r["straight_floor_visible_12_ahead_ms"]==pytest.approx(36)
    assert r["first_nonempty_motor_reference"]["key_downs_within_10ms"]==0
    destination=tmp_path/"diagnostics"/"prestige1"/m.REPORT
    content=destination.read_bytes()
    with pytest.raises(FileExistsError):
        m.run_audit(config)
    assert destination.read_bytes()==content
