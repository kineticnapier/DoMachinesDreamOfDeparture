from __future__ import annotations

"""Stage T, isolated Motor-only sequential-contact and collision diagnostic.

The prescribed target keysets here are diagnostic *scripts*, not ADOFAI
oracle labels supplied to a policy. This probe never trains or changes the
8K/263D Student and does not claim human speed, ADOFAI timing, or true pose
transition timing. Physical 12-axis dynamics and key contacts are unchanged.
"""

from dataclasses import dataclass
from math import isclose

from dmdod.motor.contact_16k import (
    HANDS, LOGICAL_KEYS, PHYSICAL_AXES, ContactAction,
    Coupled16ContactMotor, ThumbMode,
)
from dmdod.motor.keyboard import KeyEvent

from . import prestige1 as p
from . import prestige1_transfer as transfer

FORMAT = 'prestige1-stage-t-coupled-contact-scripted-sequences-v1'
REPORT = 'stage_t_coupled_contact_sequences_readonly.json'
PRESS_FRAMES = 6  # fixed 60ms motor command; not tuned for performance
MAX_RELEASE_FRAMES = 60


@dataclass(frozen=True)
class Pulse:
    label: str
    requested: tuple[str, ...]  # a desired logical keyset, *not* privileged target timing
    active_axes: tuple[str, ...]
    left_thumb: ThumbMode = ThumbMode.OFF
    right_thumb: ThumbMode = ThumbMode.OFF
    left_edge: bool = False
    right_edge: bool = False

    def __post_init__(self) -> None:
        if not self.requested or len(set(self.requested)) != len(self.requested):
            raise ValueError('Pulse must request unique nonempty logical contacts')
        if not self.active_axes or len(set(self.active_axes)) != len(self.active_axes):
            raise ValueError('Pulse must activate unique nonempty physical axes')
        if any(x not in LOGICAL_KEYS for x in self.requested):
            raise ValueError('Unknown requested key')
        if any(x not in PHYSICAL_AXES for x in self.active_axes):
            raise ValueError('Unknown physical actuator')

    def action(self, *, positive: bool) -> ContactAction:
        return ContactAction(
            tuple((1. if positive else -1.) if name in self.active_axes else 0.
                  for name in PHYSICAL_AXES),
            (self.left_thumb,self.right_thumb),
            (self.left_edge,self.right_edge),
        )


EDGE_LEFT = Pulse('edge_left_requested_only_7', ('left_7',), ('left_6',), left_edge=True)
EDGE_RIGHT = Pulse('edge_left_requested_only_8', ('left_8',), ('left_6',), left_edge=True)
EDGE_PAIR = Pulse('edge_pair', ('left_7','left_8'), ('left_6',), left_edge=True)
THUMB_TIP = Pulse('thumb_tip', ('left_5',), ('left_5',), left_thumb=ThumbMode.TIP)
THUMB_FLAT = Pulse('thumb_flat_pair', ('left_5','left_6'), ('left_5',), left_thumb=ThumbMode.FLAT)
THUMB_EXTRA = Pulse('thumb_flat_requested_only_6', ('left_6',), ('left_5',), left_thumb=ThumbMode.FLAT)
MIXED = Pulse('independent_finger_plus_edge_pair', ('left_4','left_7','left_8'),
              ('left_4','left_6'), left_edge=True)
BILATERAL = Pulse('bilateral_edge_chords', ('left_7','left_8','right_7','right_8'),
                  ('left_6','right_6'), left_edge=True,right_edge=True)

FIXTURES: tuple[tuple[str, tuple[Pulse, ...]], ...] = (
    ('edge_requested_single_7', (EDGE_LEFT,)),
    ('edge_requested_single_8', (EDGE_RIGHT,)),
    ('edge_requested_pair', (EDGE_PAIR,)),
    ('thumb_requested_tip', (THUMB_TIP,)),
    ('thumb_requested_flat_pair', (THUMB_FLAT,)),
    ('thumb_requested_second_alone', (THUMB_EXTRA,)),
    ('edge_requested_alternating_7_then_8', (EDGE_LEFT,EDGE_RIGHT)),
    ('thumb_tip_then_flat', (THUMB_TIP,THUMB_FLAT)),
    ('thumb_flat_then_tip', (THUMB_FLAT,THUMB_TIP)),
    ('independent_finger_and_edge', (MIXED,)),
    ('bilateral_edge_chords', (BILATERAL,)),
)


def _can_switch_and_rearm(motor: Coupled16ContactMotor, pulse: Pulse) -> bool:
    """The *existing* topology guard, not a new pause or optimized policy."""
    state = motor.motor.observe()
    reset_m = motor.motor.keyboard.config.reset_m
    for name in pulse.active_axes:
        i = state.key_names.index(name)
        position = state.positions_m[i]
        velocity = state.velocities_m_s[i]
        if (state.pressed_flags[i] or position > reset_m
                or (position > 0 and velocity > .0001)):
            return False
    return not any(motor.observe().logical_pressed)


def _event_dict(event) -> dict:
    return {'time_ms':round(event.time_s*1000,6), 'key':event.key,
            'event':event.event.value}


def run_case(name: str) -> dict:
    fixtures = dict(FIXTURES)
    if name not in fixtures:
        raise ValueError('Not a predeclared contact sequence')
    pulses=fixtures[name]
    motor=Coupled16ContactMotor(control_dt_s=.010,physics_dt_s=.001)
    traces=[]
    total_extra=total_missing=0
    for ordinal, pulse in enumerate(pulses,1):
        if any(motor.observe().logical_pressed):
            raise RuntimeError('Previous pulse was not released')
        press_events=[]
        start_s=motor.motor.time_s
        for _ in range(PRESS_FRAMES):
            press_events.extend(motor.step(pulse.action(positive=True)).logical_events)
        downs=[evt for evt in press_events if evt.event is KeyEvent.DOWN]
        actual={evt.key for evt in downs}
        desired=set(pulse.requested)
        if len(downs)!=len(actual):
            raise RuntimeError('Duplicate logical keydowns in a single pulse')
        if any(evt.event is KeyEvent.UP for evt in press_events):
            raise RuntimeError('Unexpected release before pulse ends')
        extra=sorted(actual-desired)
        missing=sorted(desired-actual)
        total_extra+=len(extra)
        total_missing+=len(missing)

        release_events=[]
        release_steps=0
        safe_time=None
        for _ in range(MAX_RELEASE_FRAMES):
            release_events.extend(motor.step(pulse.action(positive=False)).logical_events)
            release_steps+=1
            if _can_switch_and_rearm(motor,pulse):
                safe_time=motor.motor.time_s
                break
        if safe_time is None:
            raise RuntimeError('Physical Motor did not return to safe rest within bound')
        ups=[evt for evt in release_events if evt.event is KeyEvent.UP]
        if {evt.key for evt in ups} != actual or len(ups)!=len(actual):
            raise RuntimeError('Expected exactly one KeyUp per physically pressed contact')
        if motor.observe().thumb_modes[0]!=pulse.left_thumb:
            raise RuntimeError('Current contact mode was unexpectedly changed')
        traces.append({
            'ordinal':ordinal,'pulse':pulse.label,
            'requested':sorted(desired), 'physical_axes':list(pulse.active_axes),
            'actual_downs':sorted(actual), 'unexpected_downs':extra,
            'missing_downs':missing,
            'press_command_start_ms':round(start_s*1000,6),
            'physical_down_events':[_event_dict(e) for e in downs],
            'physical_up_events':[_event_dict(e) for e in ups],
            'release_command_start_ms':round((start_s+PRESS_FRAMES*.010)*1000,6),
            'safe_to_switch_ms':round(safe_time*1000,6),
            'elapsed_from_release_to_safe_ms':round(
                (safe_time-start_s-PRESS_FRAMES*.010)*1000,6),
            'physical_recovery_frames':release_steps,
        })
    # In these fixtures, switch happens at rest on the first subsequent action.
    for earlier,later in zip(traces,traces[1:]):
        if not isclose(earlier['safe_to_switch_ms'],later['press_command_start_ms'],abs_tol=1e-6):
            raise RuntimeError('Unaccounted gap or overlap across scripted pulses')
    return {
        'name':name,'pulse_count':len(pulses),
        'unexpected_keydown_count':total_extra,
        'missing_keydown_count':total_missing,
        'requested_keydown_count':sum(len(p.requested) for p in pulses),
        'actual_keydown_count':sum(len(t['actual_downs']) for t in traces),
        'pulses':traces,
        'continuous_physical_motor':True,
        'motor_resets_after_start':0,
        'no_adofai_evaluator_or_student':True,
    }


def run_audit(config: p.PrestigeConfig)->dict:
    if p.KEY_COUNT!=8 or p.INPUT_DIM!=263 or p.CONTROL_DT!=.010 or p.PHYSICS_DT!=.001:
        raise RuntimeError('Frozen Stage A / Student clock has changed')
    destination=transfer.output_dir(config)/REPORT
    if destination.exists():
        raise FileExistsError('Refuse to replace original contact-sequence diagnostics')
    cases=[run_case(name) for name,_ in FIXTURES]
    for case in cases:
        print(f"contact-sequence {case['name']}: expected={case['requested_keydown_count']} "
              f"downs={case['actual_keydown_count']} "
              f"extra={case['unexpected_keydown_count']} "
              f"missing={case['missing_keydown_count']}",flush=True)
    report={
        'format':FORMAT,'cases':cases,
        'motor_physical_axes':12,'keyboard_logical_contacts':16,
        'scripted_motor_trajectories':True,
        'no_chart_oracle':True,'no_adofai_evaluator':True,
        'no_training':True,'no_student_checkpoint':True,
        'stage_a_fail_preserved':True,'existing_geometry_hybrid_untouched':True,
        'validation_evaluated':False,'final_evaluated':False,
        'policy_control_dt_ms':10.,'physics_dt_ms':1.,
        'press_command_frames':PRESS_FRAMES,
        'limitations':[
            'These keysets are scripted diagnostic requests, not ADOFAI targets or a trained planner',
            'One hand-edge axis always presses logical keys 7 and 8 together; isolated edge 1K cannot be represented',
            'A flat-thumb axis always presses contacts 5 and 6 together; isolated contact 6 cannot be represented',
            'Each pulse is followed by full release; therefore this tests serial contact modes, not optimal interleaved fingering',
            'Release time is generic simulated actuator recovery, not measured human thumb posture-switch latency',
            'At-rest topology switch incurs zero movement time and is not a physical anatomical pose model',
            'No overload/accuracy/Hit claim; extra contacts are counted but no ADOFAI judgement is simulated',
            'Frozen 8-Key Student, 263D HUD, physics and all checkpoints unchanged',
        ],
    }
    destination.parent.mkdir(parents=True,exist_ok=True)
    p._no_replace_json(destination,report)
    return report
