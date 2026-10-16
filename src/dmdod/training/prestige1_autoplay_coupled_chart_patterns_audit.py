from __future__ import annotations

"""Fixed variable-angle/event chart probe for a 12-axis/16-contact Motor.

Evaluator-only oracle targets supply the *teacher*, not the Motor or HUD.
Attribute each physical logical KeyDown to (a) its reserved target and
(b) the unchanged real-chart evaluator's actual target. Those are different
measurements, especially with paired simultaneous thumb/edge contacts.
"""

from hashlib import sha256
import json
from statistics import mean

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_chart_16k import CoupledContactRealChartEnv
from dmdod.motor.keyboard import KeyEvent

from . import prestige1 as p
from . import prestige1_transfer as transfer
from .prestige1_autoplay_coupled_chart_audit import (
    ContactOracleTeacher, CONTROL_DT, PHYSICS_DT, LEAD_S, MODES,
)

VERSION = 'prestige1-stage-t-coupled-contact-varied-angle-intent-v1'
REPORT = 'stage_t_coupled_16contact_varied_angles_intent_readonly.json'

# Synthetic ADOFAI-format diagrams; the project does not contain a licensed
# real-song chart. All profiles are fixed before evaluation; no random seed.
PATTERNS = (
    ('bends_1200', 1200., (0, 90, 180, 270, 45, 225, 315, 135) * 9, ()),
    ('events_1800', 1800., (0, 180, 270, 90, 135, 315, 0, 45, 225, 180, 270, 90) * 8,
     ({'floor': 24, 'eventType': 'SetSpeed', 'speedType': 'Bpm', 'beatsPerMinute': 2300},
      {'floor': 43, 'eventType': 'Twirl'},
      {'floor': 55, 'eventType': 'Pause', 'duration': .3},
      {'floor': 70, 'eventType': 'SetSpeed', 'speedType': 'Bpm', 'beatsPerMinute': 1400})),
    ('tight_bends_4000', 4000., (0, 45, 90, 180, 270, 225, 135, 315, 0, 180, 90, 270) * 9, ()),
)


def fixture(name: str):
    option = next((case for case in PATTERNS if case[0] == name), None)
    if option is None:
        raise ValueError('Unregistered fixed varied-angle chart')
    _, bpm, angles, events = option
    source = json.dumps({
        'angleData': angles,
        'settings': {'bpm': bpm, 'pitch': 100, 'countdownTicks': 0},
        'actions': events,
    }, sort_keys=True, separators=(',', ':'))
    segment = build_playable_segment(compile_adofai(parse_adofai_text(source)))
    if len(segment.targets) != len(angles):
        raise RuntimeError('Midspin or unplayable floor in predeclared pattern')
    gaps = [(b.episode_time_s - a.episode_time_s) * 1000
            for a, b in zip(segment.targets, segment.targets[1:])]
    if not gaps or any(v <= 0 for v in gaps):
        raise RuntimeError('Non-positive target interval')
    return segment, {
        'sha256_adofai_json': sha256(source.encode('utf-8')).hexdigest(),
        'angle_count': len(angles), 'unique_angle_count': len(set(angles)),
        'action_events': [{'floor': e['floor'], 'eventType': e['eventType']}
                          for e in events],
        'gap_min_ms': min(gaps), 'gap_max_ms': max(gaps),
        'gap_median_ms': sorted(gaps)[len(gaps)//2],
        'consecutive_gaps_at_most_20ms': sum(x <= 20. for x in gaps),
        'chart_duration_ms': segment.duration_s * 1000,
    }


def _channel_map(teacher: ContactOracleTeacher) -> dict[str, tuple[str, int]]:
    """Map each logical key to its physical axis and paired-contact index."""
    mapping = {}
    for axis, logical in teacher.channels:
        for index, key in enumerate(logical):
            if key in mapping:
                raise RuntimeError('Logical contact has multiple physical sources')
            mapping[key] = axis, index
    return mapping


def simulate(name: str, keys_per_hand: int) -> dict:
    if keys_per_hand not in MODES:
        raise ValueError('Unsupported fixed finger configuration')
    segment, metadata = fixture(name)
    teacher = ContactOracleTeacher(keys_per_hand)
    mapping = _channel_map(teacher)
    env = CoupledContactRealChartEnv(segment,
                                     control_dt_s=CONTROL_DT, physics_dt_s=PHYSICS_DT)
    obs = env.reset()
    details = []
    max_steps = int((segment.duration_s + 1.) / CONTROL_DT) + 4
    physical_downs = judged = unjudged = planned_downs = unplanned_downs = 0
    wrong_claimed_target = 0
    exact_planned_time_errors = []
    actual_judged_hit_time_errors = []
    paired_plan_contacts = 0
    paired_simultaneous_gap_ms = []
    unexpected_contact_downs = 0
    for tick in range(max_steps):
        # Privileged truth is used *only in this controlled oracle benchmark*.
        first = env.privileged_next_target()
        remaining = (tuple((int(t.ordinal), float(t.episode_time_s))
                           for t in segment.targets[int(first.ordinal):])
                     if first is not None else ())
        action = teacher.action(obs, env.privileged_episode_time_s(), remaining)
        # Keep the plan at the instant the physical command is issued, before
        # scorer feedback alters target state. This is diagnostic only.
        plans = {axis:tuple(plan.tokens)
                 for axis,plan in teacher.reservations.items()}
        old_judged = len(env.judgement_trace)
        step = env.step(action)
        new_judged = env.judgement_trace[old_judged:]
        real_downs = [e for e in env.last_events if e.event is KeyEvent.DOWN]
        physical_downs += len(real_downs)
        judged += len(new_judged)
        unjudged += len(real_downs)-len(new_judged)
        if len(new_judged)>len(real_downs):
            raise RuntimeError('Scorer invented KeyDown events')
        for j,event in enumerate(real_downs):
            info = new_judged[j] if j < len(new_judged) else None
            if info is not None and (info['key'] != event.key or
                                    abs(info['time_s']-event.time_s)>1e-8):
                raise RuntimeError('Physical and scorer KeyDown traces differ')
            axis, contact_index = mapping[event.key]
            tokens = plans.get(axis, ())
            planned = tokens[contact_index] if contact_index < len(tokens) else None
            if planned is None:
                unplanned_downs += 1
                unexpected_contact_downs += 1
            else:
                planned_downs += 1
                exact_planned_time_errors.append(
                    (event.time_s-segment.targets[planned].episode_time_s)*1000.)
                if len(tokens)==2:
                    paired_plan_contacts += 1
                    paired_simultaneous_gap_ms.append(abs(
                        segment.targets[tokens[1]].episode_time_s-
                        segment.targets[tokens[0]].episode_time_s)*1000.)
            credited = info['target_index'] if info is not None else None
            if planned is not None and credited != planned:
                wrong_claimed_target += 1
            if info is not None and credited is not None and info['margin'] not in (
                    'too_early', 'fail_overload', 'too_late', None):
                actual_judged_hit_time_errors.append(
                    (event.time_s-segment.targets[credited].episode_time_s)*1000.)
            details.append({
                'frame': tick, 'key': event.key,
                'physical_axis': axis, 'pair_contact_index': contact_index,
                'physical_keydown_ms': round(event.time_s*1000,6),
                'planned_ordinal': planned,
                'planned_target_ms': (round(segment.targets[planned].episode_time_s*1000,6)
                                      if planned is not None else None),
                'planned_error_ms': (round((event.time_s-segment.targets[planned].episode_time_s)*1000,6)
                                     if planned is not None else None),
                'scorer_target_ordinal': credited,
                'scorer_margin': info['margin'] if info is not None else None,
                'planned_target_matches_scorer': planned==credited if info is not None else False,
                'paired_physical_press': len(tokens)==2,
            })
        obs=step.observation
        if step.done:
            break
    else:
        raise RuntimeError('Fixed chart failed to complete within bounded horizon')
    stats=env.stats
    if stats.hits+stats.misses!=len(segment.targets):
        raise RuntimeError('Unresolved chart target')
    if judged!=len(env.judgement_trace) or physical_downs!=planned_downs+unplanned_downs:
        raise RuntimeError('Physical or scorer event accounting inconsistent')
    if physical_downs!=judged+unjudged:
        raise RuntimeError('KeyDown event accounting inconsistent')
    if len(exact_planned_time_errors)!=planned_downs:
        raise RuntimeError('Planned keydown timing sample count mismatch')
    # Preserve both planned-attribution and actual scorer-attribution errors.
    def mean_abs(values):
        return mean(abs(v) for v in values) if values else None
    def pct_larger(values, threshold):
        return sum(abs(v)>threshold for v in values)
    return {
        'fixture':name, 'per_hand_keys':keys_per_hand,
        'fixture_metadata':metadata,
        'hit':stats.hits, 'miss':stats.misses,
        'too_early':stats.too_early_presses,
        'overloaded':stats.overloaded, 'x_accuracy_percent':stats.x_accuracy_percent,
        'mean_abs_scorer_hit_error_ms':mean_abs(actual_judged_hit_time_errors),
        'mean_abs_planned_keydown_error_ms':mean_abs(exact_planned_time_errors),
        'planned_error_over_20ms_count':pct_larger(exact_planned_time_errors,20.),
        'physical_logical_keydowns':physical_downs, 'scorer_judged_downs':judged,
        'unjudged_due_to_terminal':unjudged,
        'keydowns_with_plan':planned_downs,
        'keydowns_without_plan':unplanned_downs,
        'planned_vs_scorer_target_mismatch':wrong_claimed_target,
        'unplanned_extra_contact_downs':unexpected_contact_downs,
        'paired_contact_keydowns_with_plan':paired_plan_contacts,
        'paired_plan_target_gap_min_ms':min(paired_simultaneous_gap_ms)
            if paired_simultaneous_gap_ms else None,
        'paired_plan_target_gap_mean_ms':mean(paired_simultaneous_gap_ms)
            if paired_simultaneous_gap_ms else None,
        'paired_reserved_axis_strokes':teacher.two_contact_reservations,
        'physical_axes_used':sorted(teacher.used_axes),
        'first_keydown_attributions':details[:24],
        'largest_planned_time_errors':sorted(
            (x for x in details if x['planned_error_ms'] is not None),
            key=lambda x:-abs(x['planned_error_ms']))[:10],
        'continuous_motor':True,
        'privileged_targets_given_only_to_teacher':True,
    }


def run_audit(config:p.PrestigeConfig)->dict:
    if (p.KEY_COUNT!=8 or p.INPUT_DIM!=263 or
            p.CONTROL_DT!=CONTROL_DT or p.PHYSICS_DT!=PHYSICS_DT):
        raise RuntimeError('Frozen 8-Key Student contract changed')
    destination=transfer.output_dir(config)/REPORT
    if destination.exists():
        raise FileExistsError('Refuse overwrite of varied-angle contact diagnostics')
    rows=[]
    for name, *_ in PATTERNS:
        for mode in MODES:
            row=simulate(name,mode)
            rows.append(row)
            print(f'varied-contact {name} {mode}K/hand: '
                  f'H={row["hit"]}/{row["fixture_metadata"]["angle_count"]} '
                  f'TE={row["too_early"]} '
                  f'planMAE={row["mean_abs_planned_keydown_error_ms"]:.2f}ms '
                  f'mismatch={row["planned_vs_scorer_target_mismatch"]} '
                  f'paired={row["paired_reserved_axis_strokes"]}',flush=True)
    report={
        'format':VERSION,'cases':rows,
        'fixed_generator_not_existing_song_chart':True,
        'physical_contact_keydowns_measured':True,
        'planned_vs_scorer_attribution_separate':True,
        'oracle_teacher_privileged_true_target_schedule':True,
        'hud_student_trained':False,'training':False,
        'student_8key_263d_unchanged':True,
        'stage_a_fail_preserved':True,
        'geometry_hybrid_untouched':True,
        'validation_evaluated':False,'final_evaluated':False,
        'limitations':[
            'Generated varied-angle ADOFAI charts (including Twirl, SetSpeed and Pause), not user-supplied real songs',
            'The teacher uses privileged true target times/ordinals; results are not HUD-only inference',
            'Pairs are two simultaneous logical contacts of one physical actuator, not two independent strikes',
            'Existing incomplete Multipress scorer can award simultaneous contacts to successive targets',
            'Planned-target error and scorer-hit error must not be conflated',
            'Teacher does not learn or dynamically change thumb contact posture',
            'Twelve generic actuator physics are not calibrated human anatomy',
            'Comparison is a fixed read-only diagnostic; no Validation, FINAL, or hyperparameter search',
        ],
    }
    destination.parent.mkdir(parents=True,exist_ok=True)
    p._no_replace_json(destination,report)
    return report
