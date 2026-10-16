from __future__ import annotations

"""Independent chart/teacher diagnostic for human-inspired coupled 16 contacts.

The teacher is **privileged**: it reads exact upcoming target timestamps.
It is not a Student or HUD-only policy, and is deliberately never imported
into the frozen training system. Hit totals do not validate real Multipress.
"""

from dataclasses import dataclass
import json

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_chart_16k import CoupledContactRealChartEnv
from dmdod.motor.contact_16k import (
    ContactAction, Coupled16ContactMotor, ThumbMode, HANDS, PHYSICAL_AXES,
)

from . import prestige1 as p
from . import prestige1_transfer as transfer

FORMAT = 'prestige1-stage-t-coupled-16contact-real-chart-v1'
REPORT = 'stage_t_coupled_16contact_real_chart_teacher_readonly.json'
CONTROL_DT = .010
PHYSICS_DT = .001
LEAD_S = .044  # prescribed before evaluation, not searched
FIXTURES = ((600., 24), (1200., 48), (4000., 72))
MODES = (4, 5, 6, 7, 8)


def _fixture(bpm: float, notes: int):
    if (bpm, notes) not in FIXTURES:
        raise ValueError('Not a fixed contact chart fixture')
    source = json.dumps({'angleData': [0.] * notes,
                         'settings': {'bpm': bpm, 'pitch': 100., 'countdownTicks': 0},
                         'actions': []},sort_keys=True)
    segment = build_playable_segment(compile_adofai(parse_adofai_text(source)))
    if len(segment.targets) != notes:
        raise RuntimeError('Fixture note count inconsistent')
    return segment


def _mode(count: int) -> tuple[ThumbMode, bool]:
    if count not in MODES:
        raise ValueError('Only 4K..8K per hand are supported')
    return ((ThumbMode.OFF if count==4 else ThumbMode.TIP if count in (5,7)
             else ThumbMode.FLAT), count in (7,8))


def _physical_channels(keys_per_hand: int) -> tuple[tuple[str, tuple[str,...]], ...]:
    thumb, edge = _mode(keys_per_hand)
    # Physical +logical contacts per hand, preferred least-invasive first.
    channel_map=[]
    for hand in HANDS:
        for depth in range(1,5):
            channel_map.append((f'{hand}_{depth}',(f'{hand}_{depth}',)))
        if thumb is not ThumbMode.OFF:
            channel_map.append((f'{hand}_5',
                Coupled16ContactMotor.contact_keys(hand,5,thumb,edge_enabled=edge)))
        if edge:
            channel_map.append((f'{hand}_6',(f'{hand}_7',f'{hand}_8')))
    return tuple(channel_map)


@dataclass
class Reservation:
    tokens: tuple[int,...]
    due_s: float
    key: str
    started: bool = False


class ContactOracleTeacher:
    """Simple fixed centre-first continuous Motor-only teacher.

    It reserves physical axes for successive target ordinals; a shared
    thumb/edge axis is credited with *two* target ordinals only when both
    upcoming target times are within a fixed 20ms. No hidden dynamics from
    live evaluator are used besides the authorized privileged target schedule.
    """

    def __init__(self, keys_per_hand: int):
        self.keys_per_hand=keys_per_hand
        self.thumb,self.edge=_mode(keys_per_hand)
        self.channels=_physical_channels(keys_per_hand)
        self.reservations: dict[str,Reservation]={}
        self.total_reserved=0
        self.two_contact_reservations=0
        self.skipped_due=0
        self.logical_extras=0
        self.next_channel=0
        self.used_axes: set[str] = set()

    def action(self, observed, now_s: float, remaining: tuple[tuple[int,float],...]) -> ContactAction:
        physical=observed.motor.physical
        pressed=dict(zip(physical.key_names, physical.pressed_flags))
        positions=dict(zip(physical.key_names,physical.positions_m))
        # Remove plans which are no longer pending; only oracle teacher does
        # this. The student cannot use this evaluator feedback.
        pending={token for token,_ in remaining}
        for key, plan in tuple(self.reservations.items()):
            if not any(token in pending for token in plan.tokens):
                del self.reservations[key]
        occupied_tokens={t for item in self.reservations.values() for t in item.tokens}
        assigned=set(self.reservations)
        # The known future event times are strictly oracle-only data.
        candidates=list(remaining)
        for i,(token,due) in enumerate(candidates):
            if due > now_s+LEAD_S+1e-10:
                break
            if token in occupied_tokens:
                continue
            choice=None
            for offset in range(len(self.channels)):
                channel_index=(self.next_channel+offset)%len(self.channels)
                key,logical=self.channels[channel_index]
                if key in assigned or pressed[key]:
                    continue
                # Twin contacts must be matched to two successive unreserved
                # targets. Do not knowingly generate surplus contacts at low BPM.
                count=len(logical)
                if count==2 and (i+1>=len(candidates)
                                 or candidates[i+1][0] in occupied_tokens
                                 or candidates[i+1][1]-due > .020):
                    continue
                choice=(key,count,channel_index)
                break
            if choice is None:
                self.skipped_due+=1
                continue
            key,count,channel_index=choice
            self.next_channel=(channel_index+1)%len(self.channels)
            self.used_axes.add(key)
            ordinals=tuple(x[0] for x in candidates[i:i+count])
            self.reservations[key]=Reservation(ordinals,due,key)
            assigned.add(key)
            occupied_tokens.update(ordinals)
            self.total_reserved+=1
            if count==2:self.two_contact_reservations+=1
        commands=[]
        for key in PHYSICAL_AXES:
            plan=self.reservations.get(key)
            if pressed[key]:
                command=-1.
            elif plan is not None and now_s+LEAD_S>=plan.due_s:
                command=1.
                plan.started=True
            else:
                command=-1. if positions[key]>0 else 0.
            commands.append(command)
        return ContactAction(tuple(commands),
            (self.thumb,self.thumb),(self.edge,self.edge))


def simulate(bpm: float, notes: int, keys_per_hand:int) -> dict:
    if (bpm,notes) not in FIXTURES or keys_per_hand not in MODES:
        raise ValueError('Unsupported fixed experiment')
    segment=_fixture(bpm,notes)
    env=CoupledContactRealChartEnv(segment,control_dt_s=CONTROL_DT,physics_dt_s=PHYSICS_DT)
    teacher=ContactOracleTeacher(keys_per_hand)
    obs=env.reset()
    first_events=[]
    max_frames=int((segment.duration_s+.5)/CONTROL_DT)+3
    for tick in range(max_frames):
        future=env.privileged_next_target()
        remaining=(tuple((int(t.ordinal),float(t.episode_time_s))
                         for t in segment.targets[int(future.ordinal):])
                   if future is not None else ())
        action=teacher.action(obs,env.privileged_episode_time_s(),remaining)
        out=env.step(action)
        for item in env.last_events:
            if item.event.value=='down' and len(first_events)<24:
                first_events.append((item.time_s,item.key))
        obs=out.observation
        if out.done:
            break
    else:
        raise RuntimeError('Did not finish chart within bounded horizon')
    stats=env.stats
    # This identity is authoritative even if the evaluator terminates early.
    if stats.hits+stats.misses!=notes:
        raise RuntimeError('Unresolved targets in physical chart benchmark')
    return {
        'bpm':bpm,'notes':notes,'per_hand_logical_mode':keys_per_hand,
        'hits':stats.hits,'misses':stats.misses,
        'too_early':stats.too_early_presses,'overloaded':stats.overloaded,
        'x_accuracy_percent':stats.x_accuracy_percent,
        'real_physical_contact_keydowns':env.physical_keydowns,
        'recorded_keydowns':len(env.judgement_trace),
        'two_contact_reservations':teacher.two_contact_reservations,
        'one_or_two_contact_reservations':teacher.total_reserved,
        'skipped_due_reservations':teacher.skipped_due,
        'used_physical_axes':sorted(teacher.used_axes),
        'first_logical_downs':[{'time_ms':round(ts*1000.,6),'key':k}
                               for ts,k in first_events],
        'first_judgements':env.judgement_trace[:24],
        'single_continuous_motor':True,
        'oracle_target_schedule_supplied_to_teacher':True,
    }


def run_audit(config:p.PrestigeConfig) -> dict:
    if p.KEY_COUNT!=8 or p.INPUT_DIM!=263 or p.CONTROL_DT!=CONTROL_DT or p.PHYSICS_DT!=PHYSICS_DT:
        raise RuntimeError('Frozen 8-key training contract changed')
    filename=transfer.output_dir(config)/REPORT
    if filename.exists():
        raise FileExistsError('Refuse overwrite of 16-contact chart report')
    cases=[]
    for bpm,notes in FIXTURES:
        for keys_per_hand in MODES:
            row=simulate(bpm,notes,keys_per_hand)
            cases.append(row)
            print(f'coupled-chart {bpm:>6.0f} BPM {keys_per_hand}K/hand: '
                  f'H={row["hits"]}/{notes} TE={row["too_early"]} '
                  f'paired={row["two_contact_reservations"]}',flush=True)
    result={'format':FORMAT,'cases':cases,
            'oracle_motor_only':True,'not_hud_or_student':True,
            'not_true_multipress':True,
            'no_training':True,'no_checkpoints_changed':True,
            'stage_a_fail_preserved':True,'validation_evaluated':False,
            'final_evaluated':False,'policy_dt_ms':10.,'physics_dt_ms':1.,
            'limitations':[
                'A deterministic privileged teacher reads true future target indices/times',
                'Physical model is 12 independent axes and 16 logical contacts, not real anatomy',
                'Static 4K-8K contact posture per hand: no pose-switching cost or algorithm',
                'Existing judgement can credit simultaneous presses as multiple note Hits',
                'Only fixed straight-floor synthetic ADOFAI charts, not heldout real-world charts',
                'No changes to frozen 8K HUD, Student, MaleCNS4096 or original evaluator',
            ]}
    filename.parent.mkdir(parents=True,exist_ok=True)
    p._no_replace_json(filename,result)
    return result
