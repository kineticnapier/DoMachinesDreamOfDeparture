from __future__ import annotations

"""Read-only 4K..8K per hand and 16K bilateral *contact topology* smoke audit.

This is a physical contact layer, NOT chart play, BPM capacity, a calibrated
anatomical model, or the frozen MaleCNS4096 Student.  No target oracle used.
"""

from dmdod.motor.contact_16k import (
    HANDS, LOGICAL_KEYS, PHYSICAL_AXES, ContactAction, Coupled16ContactMotor,
    ThumbMode,
)
from dmdod.motor.keyboard import KeyEvent
from . import prestige1 as p
from . import prestige1_transfer as transfer

FORMAT = 'prestige1-stage-t-12axis-16contact-4k8k-v1'
REPORT = 'stage_t_coupled_16contact_topology_readonly.json'
MODES = (
    ('4K', ThumbMode.OFF, False, 4),
    ('5K', ThumbMode.TIP, False, 5),
    ('6K', ThumbMode.FLAT, False, 6),
    ('7K', ThumbMode.TIP, True, 7),
    ('8K', ThumbMode.FLAT, True, 8),
)
CASES = tuple((hand, name, mode, edge, keys)
              for hand in HANDS for name, mode, edge, keys in MODES) + (
    ('both', '16K', ThumbMode.FLAT, True, 16),
)
PRESS_FRAMES = 30
RELEASE_FRAMES = 40


def _action(hands: tuple[str, ...], mode: ThumbMode, edge: bool,
            *, pushing: bool) -> ContactAction:
    commands = []
    modes = []
    edges = []
    for h in HANDS:
        enabled = h in hands
        modes.append(mode if enabled else ThumbMode.OFF)
        edges.append(edge if enabled else False)
    for channel in PHYSICAL_AXES:
        h, number = channel.split('_')
        n = int(number)
        active = h in hands and (n <= 4 or (n == 5 and mode is not ThumbMode.OFF)
                                 or (n == 6 and edge))
        commands.append((1. if pushing else -1.) if active else 0.)
    return ContactAction(tuple(commands),tuple(modes),tuple(edges))


def run_case(hand: str, mode_name: str, mode: ThumbMode,
             edge: bool, expected: int) -> dict:
    if (hand,mode_name,mode,edge,expected) not in CASES:
        raise ValueError('Unregistered contact-mode fixture')
    hands = HANDS if hand == 'both' else (hand,)
    env = Coupled16ContactMotor()
    press = _action(hands,mode,edge,pushing=True)
    release = _action(hands,mode,edge,pushing=False)
    waves: list[list] = []
    releases: list = []
    for stage,action,frames in (
        ('first_press',press,PRESS_FRAMES),
        ('release',release,RELEASE_FRAMES),
        ('second_press',press,PRESS_FRAMES),
    ):
        events=[]
        for _ in range(frames):
            tr=env.step(action)
            events.extend(tr.logical_events)
        if stage=='release':
            releases=events
            if any(tr.observation.logical_pressed):
                raise RuntimeError('Released motor retained logical press')
        else:
            waves.append(events)
    expected_keys={k for h in hands for k in
                   Coupled16ContactMotor.visible_keys_for_hand(
                       h,mode,edge_enabled=edge)}
    assert len(expected_keys)==expected
    results=[]
    for events in waves:
        downs=[x for x in events if x.event is KeyEvent.DOWN]
        if len(downs)!=expected or {x.key for x in downs}!=expected_keys:
            raise RuntimeError('Physical contact layer did not press all expected keys')
        times={x.key:x.time_s for x in downs}
        for h in hands:
            if edge and times[f'{h}_7']!=times[f'{h}_8']:
                raise RuntimeError('Coupled edge contacts de-synchronized')
            if mode is ThumbMode.FLAT and times[f'{h}_5']!=times[f'{h}_6']:
                raise RuntimeError('Coupled thumb contacts de-synchronized')
        results.append({
            'logical_down_count':len(downs),
            'physical_used_axes':expected - len(hands)*(
                (1 if mode is ThumbMode.FLAT else 0)+(1 if edge else 0)),
            'first_down_ms':min(times.values())*1000.,
            'last_down_ms':max(times.values())*1000.,
            'down_timestamps_ms':{k:round(v*1000.,6) for k,v in sorted(times.items())},
        })
    ups=[x for x in releases if x.event is KeyEvent.UP]
    if len(ups)!=expected or {x.key for x in ups}!=expected_keys:
        raise RuntimeError('Physical contact layer failed to release all contacts')
    if len(LOGICAL_KEYS)!=16 or len(PHYSICAL_AXES)!=12:
        raise RuntimeError('Contact/actuator contract changed')
    return {
        'hand':hand, 'mode':mode_name, 'thumb':mode.value, 'edge_pair':edge,
        'expected_logical_contacts':expected,
        'first_press':results[0], 'release_up_count':len(ups),
        'second_press':results[1],
        'no_motor_reset_between_strokes':True,
        'held_action_dt_ms':10., 'physical_dt_ms':1.,
    }


def run_audit(config:p.PrestigeConfig) -> dict:
    if p.KEY_COUNT!=8 or p.INPUT_DIM!=263 or p.CONTROL_DT!=.010 or p.PHYSICS_DT!=.001:
        raise RuntimeError('Frozen Prestige I control contract has changed')
    target=transfer.output_dir(config)/REPORT
    if target.exists():
        raise FileExistsError('Refuse to overwrite contact-model audit')
    cases=[run_case(*args) for args in CASES]
    for case in cases:
        print(f'contact {case["hand"]:>5} {case["mode"]:>3}: '
              f'{case["first_press"]["logical_down_count"]} down, '
              f'{case["release_up_count"]} up, '
              f'{case["second_press"]["logical_down_count"]} redown',flush=True)
    report={
        'format':FORMAT, 'cases':cases,
        'underlying_motor_physical_axes':12,
        'two_hand_logical_keyboard_contacts':16,
        'axes_per_hand':6,
        'strict_shared_thumb_contact':True,
        'strict_coupled_edge_pair':True,
        'physical_contacts_measurements_only':True,
        'not_bpm_hit_or_accuracy_measurement':True,
        'not_human_16k_speed_validation':True,
        'not_trained_student_or_teacher':True,
        'no_training':True,'no_checkpoint_changed':True,
        'stage_a_fail_untouched':True,'frozen_8key_263d_unchanged':True,
        'validation_evaluated':False,'final_evaluated':False,
        'limitations':[
            'Twelve physically independent axes (four fingers + one thumb + one hand edge per hand)',
            'Thumb flat/tip mode selects switch contacts; it does not simulate thumb joint geometry',
            'Side two contacts always co-actuate; rare independent edge technique not modeled',
            'Mode change while depressed forbidden; at-rest mode change has no time or cost',
            'Each of the six axes uses a reused generic NKeyBody finger parameter; human anatomy is not calibrated',
            'No 16K ADOFAI environment, timings, ADOFAI judgements, actual playable charts or 16K learning',
            'Physical pressed contacts are not independent physical control inputs',
            'Existing original 8K Student/HUD/checkpoints remain untouched',
        ],
    }
    target.parent.mkdir(parents=True,exist_ok=True)
    p._no_replace_json(target,report)
    return report
