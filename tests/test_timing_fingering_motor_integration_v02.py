from __future__ import annotations

import ast
from dataclasses import fields
import inspect
from types import SimpleNamespace

import pytest
import torch

from dmdod.motor.contact_16k import ContactAction, Coupled16ContactMotor
from dmdod.envs.contact_hud_chart_v02 import HudCoupledContactRealChartEnv
from dmdod.envs.n_key import DiagnosticHudNKeyRealChartMotorEnv
from dmdod.motor.n_key import NKeyAction
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG as FLOORS
from scripts.audit_hud_timing_tracker_transition_v14_matrix import generated
from dmdod.planning import contact_fingering_candidates_v02 as candidates
from dmdod.planning import hud_v14_forecast_source_v02 as hud_source
from dmdod.planning import oracle_forecast_source_v02 as oracle
from dmdod.planning import timing_fingering_context_v02 as context
from dmdod.planning import timing_fingering_motor_policy_v02 as policy
from dmdod.planning.contact_timing_hud_v01 import ContactTimingHUD263Frame
from dmdod.planning.hud_timing_tracker_transition_v14 import (
    CausalTimingTracker, WindowForecast,
)
from dmdod.planning.persistent_fingering_scheduler_v01 import (
    MotorPhysicalSnapshot, PersistentFingeringScheduler,
)
from dmdod.planning.timing_fingering_contract_v02 import (
    ABSTAIN, ELIGIBLE, TimingEventToken, TimingFrameSequenceValidator,
    TimingFrameUpdate,
)
from dmdod.training import fingering_network_v01 as frozen_network
from scripts import audit_oracle_fingering_v02 as old_teacher
from scripts import run_hud_v14_fingering_motor_integration_v02 as runner


def test_hud_adapter_holds_expired_events_across_abstention(monkeypatch):
    forecasts = [
        WindowForecast((1,), (.065,), ELIGIBLE),
        *[WindowForecast((), (), "pause_motion_unobservable") for _ in range(7)],
        WindowForecast((), (), ELIGIBLE),
    ]
    adapter = hud_source.HUDV14ForecastSource()
    adapter.hud._tracker = CausalTimingTracker(
        horizon_s=.080,
        predictor=lambda _: forecasts[adapter.hud._tracker._next_frame],
    )
    monkeypatch.setattr(
        hud_source.v14, "forecast_visible_window",
        lambda *args, **kwargs: forecasts[adapter.hud._tracker._next_frame]
    )
    validator = TimingFrameSequenceValidator()
    results = []
    for step in range(9):
        update = adapter.observe(ContactTimingHUD263Frame(step, (0.,) * 263))
        validator.accept(update)
        results.append(update)
    assert results[0].new_ids == (0,)
    assert results[0].pending[0].due_local_s == .065
    assert all(x.status == ABSTAIN and x.pending_ids == (0,)
               and not x.pending[0].confirmed_now for x in results[1:8])
    assert all(not x.elapsed_ids and not x.canceled_ids
               for x in results[1:8])
    assert results[8].elapsed_ids == (0,)
    assert results[8].pending == ()


def test_identical_oracle_contract_and_equal_time_chord_coalescing():
    source = oracle.OracleForecastSource((
        oracle.OracleDue(.065), oracle.OracleDue(.065),
        oracle.OracleDue(.068), oracle.OracleDue(.140),
    ))
    updates = [source.observe(step) for step in range(17)]
    assert updates[0].new_ids == (0, 1)
    assert tuple(x.chord_size for x in updates[0].pending) == (2, 1)
    assert tuple(x.due_local_s for x in updates[0].pending) == (.065, .068)
    assert updates[7].elapsed_ids == (0, 1)
    assert any(2 in update.new_ids for update in updates)
    validator = TimingFrameSequenceValidator()
    for update in updates:
        validator.accept(update)


@pytest.mark.parametrize("size", (1, 2, 3, 4, 8))
def test_policy_candidate_enumeration_matches_frozen_physical_options(size):
    old = old_teacher.finger_options(size)
    new = candidates.finger_options(size)
    assert len(old) == len(new)
    assert tuple(tuple(x.label for x in option) for option in old) == (
        tuple(tuple(x.label for x in option) for option in new)
    )


def test_frozen_65d_context_from_local_timing_is_bitwise_identical():
    motor = Coupled16ContactMotor()
    update = TimingFrameUpdate(
        step=3, status=ELIGIBLE, source="fixture",
        covered_until_s=.100,
        pending=(
            TimingEventToken(0, .060, 1),
            TimingEventToken(1, .075, 2),
            TimingEventToken(2, .090, 1),
        ),
        new_ids=(0, 1, 2),
    )
    old = tuple(SimpleNamespace(
        episode_time_s=x.due_local_s, chord_size=x.chord_size
    ) for x in update.pending)
    expected = frozen_network.encode_context(motor, old, .03, {"left_4"})
    got = context.encode_context_from_timing_update(
        motor, update, {"left_4"}
    )
    assert got.shape == (frozen_network.CONTEXT_DIM,)
    assert torch.equal(got, expected)
    later = context.encode_context_from_timing_update(
        motor, update, {"left_4"}, starting_at_local_id=1
    )
    assert torch.equal(later, frozen_network.encode_context(
        motor, old[1:], .03, {"left_4"}
    ))


def test_policy_reserves_in_local_time_not_evaluator_hit_state(monkeypatch):
    motor = Coupled16ContactMotor()
    scheduler = policy.TimingFingeringMotorPolicy(
        model=frozen_network.FingeringRanker(), selector="network"
    )
    monkeypatch.setattr(
        policy.network, "choose_candidate_styled",
        lambda model, context, allocations, eligible, alpha:
        (next(i for i, ok in enumerate(eligible) if ok), (), ())
    )
    first = TimingFrameUpdate(
        step=0, status=ELIGIBLE, source="fixture",
        covered_until_s=.080, pending=(TimingEventToken(0, .065, 1),),
        new_ids=(0,),
    )
    act = scheduler.command(first, motor)
    assert all(command == 0 for command in act.commands)
    assert scheduler.scheduler.pending_local_ids == frozenset({0})
    assert len(scheduler.scheduler.reservations) == 1
    for step in range(1, 4):
        update = TimingFrameUpdate(
            step=step, status=ELIGIBLE, source="fixture",
            covered_until_s=step*.010+.080,
            pending=(TimingEventToken(0, .065, 1, confirmations=step + 1),),
            reused_ids=(0,),
        )
        action = scheduler.command(update, motor)
    assert any(x == 1 for x in action.commands)
    # Physical response is observed independently of any target scorer.
    motor.step(action)
    scheduler.observe_physics(motor)
    assert scheduler.decisions[0].reason == "assigned"


def test_late_forecast_is_recorded_not_silently_backfilled():
    motor = Coupled16ContactMotor()
    planner = policy.TimingFingeringMotorPolicy(
        model=None, selector="center_first"
    )
    update = TimingFrameUpdate(
        step=0, status=ELIGIBLE, source="fixture",
        covered_until_s=.080,
        pending=(TimingEventToken(0, .030, 1),), new_ids=(0,),
    )
    action = planner.command(update, motor)
    assert planner.insufficient_observable_lead == 1
    assert planner.scheduler.reservations == ()
    assert all(x == 0 for x in action.commands)
    assert planner.decisions[0].reason == "insufficient_observable_lead"


def test_policy_stops_positive_commands_on_abstention():
    motor = Coupled16ContactMotor()
    planner = policy.TimingFingeringMotorPolicy(
        model=None, selector="center_first"
    )
    first = TimingFrameUpdate(
        step=0, status=ELIGIBLE, source="fixture",
        covered_until_s=.080,
        pending=(TimingEventToken(0, .065, 1),), new_ids=(0,),
    )
    planner.command(first, motor)
    assert planner.scheduler.reservations
    pause = TimingFrameUpdate(
        step=1, status=ABSTAIN, source="fixture",
        covered_until_s=.010,
        pending=(TimingEventToken(
            0, .065, 1, confirmed_now=False
        ),), reused_ids=(0,), abstention_reason="pause"
    )
    command = planner.command(pause, motor)
    assert all(x <= 0 for x in command.commands)
    assert planner.scheduler.reservations
    assert planner.hud_abstentions == 1


def test_runner_hud_arms_do_not_instantiate_oracle_source(monkeypatch):
    def oracle_forbidden(*args, **kwargs):
        raise AssertionError("Oracle must not be initialized for HUD")
    monkeypatch.setattr(runner, "OracleForecastSource", oracle_forbidden)
    # No truth reading is permitted by the standalone policy; the *runner*
    # alone owns the post-hoc scoreboard.
    chart = runner.synthetic_segment(120.)
    model = frozen_network.FingeringRanker()
    result = runner.simulate(chart, model, "hud_center_first", wall_timeout_s=90.)
    assert result["timing_source"] == "hud"
    assert result["stats"]["targets"] > 0
    assert "forecast_posthoc" in result
    assert "insufficient_observable_lead" in result["scheduling"]


def test_policy_import_surface_has_no_chart_or_evaluator_truth():
    module_ast = ast.parse(inspect.getsource(policy))
    forbidden = ("target_ordinal", "target_index", "floor_index",
                 "target_time_s", "chart_time", "judgement",
                 "_used", "_missed", "segment", "targets")
    attrs = {x.attr for x in ast.walk(module_ast)
             if isinstance(x, ast.Attribute)}
    assert not attrs.intersection(forbidden)
    imported = [x.module or "" for x in ast.walk(module_ast)
                if isinstance(x, ast.ImportFrom)]
    assert not any("adofai" in x or "envs" in x for x in imported)
    for klass in (policy.FingeringDecision,):
        assert not set(forbidden).intersection(x.name for x in fields(klass))


def test_oracle_and_hud_forecast_frames_share_public_type():
    oracle_source = oracle.OracleForecastSource((oracle.OracleDue(.055),))
    oracle_frame = oracle_source.observe(0)
    assert isinstance(oracle_frame, TimingFrameUpdate)
    assert oracle_frame.pending[0].local_id == 0
    assert oracle_frame.pending[0].due_local_s == .055
    assert not hasattr(oracle_frame.pending[0], "target_index")


@pytest.mark.parametrize("variant", ("speed_double", "twirl_speed", "pause_1"))
def test_current_contact_hud_renderer_matches_legacy_display(variant):
    segment = generated((0., 90., 180., 270.), 496., variant)
    cenv = HudCoupledContactRealChartEnv(
        segment, tail_s=.040, control_dt_s=.010, physics_dt_s=.001,
        behind_floors=FLOORS.behind_floors, ahead_floors=FLOORS.ahead_floors
    )
    renv = DiagnosticHudNKeyRealChartMotorEnv(
        segment, key_count=8, tail_s=.040,
        control_dt_s=.010, physics_dt_s=.001,
        behind_floors=FLOORS.behind_floors, ahead_floors=FLOORS.ahead_floors,
    )
    contact, legacy = cenv.reset(), renv.reset()
    source = hud_source.HUDV14ForecastSource()
    reference = CausalTimingTracker(horizon_s=.080)
    for step in range(30):
        assert contact.tile_bpm == legacy.tile_bpm
        assert contact.real_bpm == legacy.real_bpm
        frame = source.hud.encode(step, contact)
        known = tuple(__import__(
            "dmdod.envs.n_key", fromlist=["encode_n_key_hud_real_chart_observation"]
        ).encode_n_key_hud_real_chart_observation(legacy))
        assert frame.features == (0.,)*24 + known[24:]
        updated = source.observe(frame)
        expected = reference.observe(step, known)
        assert updated.status == (
            expected.status if expected.status in ("eligible", "eligible_partial")
            else "abstain"
        )
        c = cenv.step(ContactAction((0.,)*12))
        n = renv.step(NKeyAction((0.,)*8))
        assert c.done == n.done
        contact, legacy = c.observation, n.observation
        if c.done:
            break


def test_prefix_only_hud_frames_cannot_depend_on_future_heldout_suffix():
    segment = generated((0., 90., 180., 270.), 496., "speed_double")
    env = HudCoupledContactRealChartEnv(segment, tail_s=.04)
    obs = env.reset()
    frames = []
    for step in range(12):
        frames.append(hud_source.HUDV14ForecastSource().hud.encode(step, obs))
        output = env.step(ContactAction((0.,)*12))
        obs = output.observation
        if output.done:
            break
    first = hud_source.HUDV14ForecastSource()
    second = hud_source.HUDV14ForecastSource()
    prefix_count = min(5, len(frames))
    prefix1 = tuple(first.observe(x) for x in frames[:prefix_count])
    prefix2 = tuple(second.observe(x) for x in frames[:prefix_count])
    assert prefix1 == prefix2
    if len(frames) > prefix_count:
        first.observe(frames[prefix_count])
        assert prefix2 == prefix1


def test_candidate_selection_identical_for_bitwise_same_context(monkeypatch):
    motor = Coupled16ContactMotor()
    update = TimingFrameUpdate(
        step=2, status=ELIGIBLE, covered_until_s=.10, source="fixture",
        pending=(TimingEventToken(0, .065, 1),), new_ids=(0,),
    )
    adapted = context.encode_context_from_timing_update(
        motor, update, set()
    )
    direct = frozen_network.encode_context(
        motor, (SimpleNamespace(episode_time_s=.065, chord_size=1),),
        .020, set(),
    )
    assert torch.equal(adapted, direct)
    model = frozen_network.FingeringRanker()
    choices = candidates.finger_options(1)
    eligibility = (True,) * len(choices)
    aa = frozen_network.choose_candidate_styled(
        model, adapted, choices, eligibility, alpha=.5
    )
    bb = frozen_network.choose_candidate_styled(
        model, direct, choices, eligibility, alpha=.5
    )
    assert aa == bb


def test_integration_reports_full_and_supported_denominators_separately():
    segment = runner.synthetic_segment(120.)
    result = runner.simulate(
        segment, model=None, arm="hud_center_first", wall_timeout_s=90.
    )
    full = result["stats"]
    supported = result["supported_event_posthoc"]
    assert full["targets"] == len(segment.targets)
    assert supported["sufficient_lead_supported_truth_events"] <= full["targets"]
    assert supported["supported_truth_hit_by_ordered_scorer"] <= (
        supported["sufficient_lead_supported_truth_events"]
    )
    assert result["forecast_posthoc"]["true_targets"] == full["targets"]


def test_oracle_local_ids_not_preassigned_from_hidden_event_ordinals():
    source = oracle.OracleForecastSource((
        oracle.OracleDue(.030),
        oracle.OracleDue(.055),
        oracle.OracleDue(.130),
    ))
    assert source._local_id_by_event == {}
    first = source.observe(0)
    assert first.new_ids == (0, 1)
    assert source._local_id_by_event == {0: 0, 1: 1}
    assert 2 not in source._local_id_by_event
    for step in range(1, 6):
        source.observe(step)
    assert 2 not in source._local_id_by_event
    for step in range(6, 7):
        source.observe(step)
    assert 2 in source._local_id_by_event
