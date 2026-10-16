from __future__ import annotations

"""Read-only physical contact KeyDown -> existing ADOFAI judgement probe.

Actual 12-axis contact events are submitted to the *unchanged* real-chart
scorer. No trained Student or HUD policy and no new Multipress semantics.
The scorer is invoked directly (not via NKeyRealChartMotorEnv.step) because
its physical actuator count differs from the 12-axis / 16-contact adapter.
"""

import json
from dataclasses import dataclass

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.adofai_rules import TimingJudgement
from dmdod.envs.n_key import DiagnosticHudNKeyRealChartMotorEnv
from dmdod.motor.contact_16k import (
    ContactAction, Coupled16ContactMotor, PHYSICAL_AXES, ThumbMode,
)
from dmdod.motor.keyboard import KeyEvent

from . import prestige1 as p
from . import prestige1_transfer as transfer

FORMAT = 'prestige1-stage-t-contact-physical-keydown-adofai-judgement-v1'
REPORT = 'stage_t_contact_keydowns_adofai_judgement_readonly.json'
CHART_FLOORS = 10
PRESS_FRAMES = 10


@dataclass(frozen=True)
class Scenario:
    name: str
    bpm: float
    preroll_frames: int
    physical_axes: tuple[str, ...]
    left_thumb: ThumbMode = ThumbMode.OFF
    left_edge: bool = False
    right_edge: bool = False


CASES = (
    Scenario('600_single_finger_control', 600., 6, ('left_4',)),
    Scenario('600_edge_pair', 600., 6, ('left_6',), left_edge=True),
    Scenario('600_flat_thumb_pair', 600., 6, ('left_5',), left_thumb=ThumbMode.FLAT),
    Scenario('600_bilateral_edge_pairs', 600., 6, ('left_6', 'right_6'),
             left_edge=True, right_edge=True),
    Scenario('1200_edge_pair', 1200., 1, ('left_6',), left_edge=True),
    Scenario('4000_edge_pair', 4000., 0, ('left_6',), left_edge=True),
    Scenario('4000_bilateral_edge_pairs', 4000., 0, ('left_6', 'right_6'),
             left_edge=True, right_edge=True),
)


def _chart(bpm: float):
    raw = json.dumps({
        'angleData': [0.] * CHART_FLOORS,
        'settings': {'bpm': bpm, 'pitch': 100., 'countdownTicks': 0},
        'actions': [],
    }, sort_keys=True, separators=(',', ':'))
    return build_playable_segment(compile_adofai(parse_adofai_text(raw)))


def _contact_action(case: Scenario, positive: bool) -> ContactAction:
    active = set(case.physical_axes)
    if not active <= set(PHYSICAL_AXES):
        raise ValueError('Unknown physical axis')
    return ContactAction(
        tuple((1. if positive else -1.) if axis in active else 0.
              for axis in PHYSICAL_AXES),
        thumb_modes=(case.left_thumb, ThumbMode.OFF),
        edge_enabled=(case.left_edge, case.right_edge),
    )


def _capture_real_contact_downs(case: Scenario):
    motor = Coupled16ContactMotor(control_dt_s=.010, physics_dt_s=.001)
    idle = ContactAction((0.,) * len(PHYSICAL_AXES),
                         (case.left_thumb, ThumbMode.OFF),
                         (case.left_edge, case.right_edge))
    for _ in range(case.preroll_frames):
        transition = motor.step(idle)
        if transition.logical_events:
            raise RuntimeError('Unexpected key event during preroll')
    events = []
    active = _contact_action(case, True)
    for _ in range(PRESS_FRAMES):
        events.extend(e for e in motor.step(active).logical_events
                      if e.event is KeyEvent.DOWN)
    if not events or len({e.key for e in events}) != len(events):
        raise RuntimeError('Contact pulse had no presses or repeated a key')
    if any(abs(e.time_s - events[0].time_s) > 1e-9 for e in events):
        raise RuntimeError('Contact pulse did not yield simultaneous keydowns')
    return tuple(events)


def run_case(name: str) -> dict:
    scenario = next((s for s in CASES if s.name == name), None)
    if scenario is None:
        raise ValueError('Not a predeclared contact/evaluator fixture')
    chart = _chart(scenario.bpm)
    downs = _capture_real_contact_downs(scenario)
    evaluator = DiagnosticHudNKeyRealChartMotorEnv(
        chart, key_count=16, capture_overload_trace=True,
        control_dt_s=.010, physics_dt_s=.001,
    )
    evaluator.reset()
    decisions = []
    for event in downs:
        # Reproduce the existing NKeyRealChartMotorEnv per-event scoring path.
        # No newly written timing/overload approximations or privileged teacher.
        if evaluator._overload.overloaded or evaluator._failed_on_miss:
            break
        evaluator._advance_overload(event.time_s)
        evaluator._expire_misses(event.time_s)
        if evaluator._overload.overloaded or evaluator._failed_on_miss:
            break
        target_index = evaluator._next_target_index()
        before = len(evaluator.hit_margins)
        gauge_before = evaluator._overload.value
        evaluator._score_event(event)
        decisions.append({
            'key': event.key,
            'time_ms': round(event.time_s * 1000., 6),
            'evaluator_target_ordinal': target_index + 1 if target_index is not None else None,
            'margin': (evaluator.hit_margins[-1].value
                       if len(evaluator.hit_margins) > before else 'no_target'),
            'overload_before': round(gauge_before, 9),
            'overload_after': round(evaluator._overload.value, 9),
        })
    stats = evaluator.stats
    if len(decisions) != len(downs):
        raise RuntimeError('Predeclared fixture unexpectedly terminated before all keydowns')
    if stats.hits + stats.too_early_presses != len(decisions):
        raise RuntimeError('Fixture produced an unaccounted event classification')
    return {
        'name': scenario.name, 'bpm': scenario.bpm,
        'first_target_ms': chart.targets[0].episode_time_s * 1000.,
        'second_target_ms': chart.targets[1].episode_time_s * 1000.,
        'contact_physical_axes': list(scenario.physical_axes),
        'contact_preroll_ms': scenario.preroll_frames * 10.,
        'raw_physical_contact_downs': len(downs),
        'judged_keydowns': len(decisions),
        'judgement_events': decisions,
        'hits': stats.hits,
        'too_early': stats.too_early_presses,
        'overload_gauge': round(stats.overload_counter, 9),
        'overloaded': stats.overloaded,
        'fail_overloads': stats.fail_overloads,
        'judgement_sequence': [x.value for x in evaluator.hit_margins],
        'multipress_state_machine_implemented': False,
        'motor_12_axis_16_contact': True,
        'oracle_motor_control_for_diagnostic': True,
    }


def run_audit(config: p.PrestigeConfig) -> dict:
    if p.KEY_COUNT != 8 or p.INPUT_DIM != 263 or p.CONTROL_DT != .010 or p.PHYSICS_DT != .001:
        raise RuntimeError('Frozen Prestige I contract changed')
    path = transfer.output_dir(config) / REPORT
    if path.exists():
        raise FileExistsError('Refuse overwrite of contact judgement audit')
    cases = [run_case(s.name) for s in CASES]
    for case in cases:
        print(f"contact judgement {case['name']}: "
              f"H={case['hits']} TE={case['too_early']} "
              f"Overload={case['overloaded']} "
              f"margins={case['judgement_sequence']}", flush=True)
    report = {
        'format': FORMAT, 'cases': cases,
        'scorer': 'unchanged DiagnosticHudNKeyRealChartMotorEnv._score_event',
        'same_timestamp_keydown_order': 'contact motor adapter event enumeration order',
        'no_full_game_env_step': True,
        'no_multipress_or_overpress_queue_semantics': True,
        'limited_to_normal_difficulty_straight_180deg_chart': True,
        'no_student_training': True, 'no_student_checkpoint': True,
        'stage_a_fail_preserved': True, 'geometry_hybrid_untouched': True,
        'validation_evaluated': False, 'final_evaluated': False,
        'limitations': [
            'Actual physical coupled-contact keydowns, injected into unchanged scorer; not full 16K gameplay integration',
            'The scorer presently lacks complete same-frame Multipress/OverPress rules; real ADOFAI may differ',
            'Each press script and chart fixture is a diagnostic oracle alignment, never policy/HUD inference',
            'Sequential per-target judgement can count a same-time surplus contact as the next Hit',
            'No tracking of subsequent chart misses or runtime action rollout after scoring the burst',
            'No Student, 4096-CNS, original 8K Motor, frozen Stage A, Validation or FINAL changes',
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    p._no_replace_json(path, report)
    return report
