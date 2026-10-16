"""Isolated multi-axis arrival-order experiment; original teacher untouched."""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
from hashlib import sha256
from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.motor.contact_16k import ContactAction, PHYSICAL_AXES
from dmdod.motor.n_key import NKeyAction
from dmdod.motor.keyboard import KeyEvent
from dmdod.training.prestige1_autoplay_coupled_chart_audit import ContactOracleTeacher
import u1_keydown_attribution_audit as base

class JointArrivalTeacher(ContactOracleTeacher):
    def __init__(self, mode, motor):
        super().__init__(mode)
        self.motor = motor
        self.forecasts = 0
        self.inversions = 0
        self.delays = 0
        self.sample_inversions = []

    def predict(self, commands, horizon_frames=7):
        model=copy.deepcopy(self.motor.motor)
        a=NKeyAction(tuple(commands))
        first={}
        for _ in range(horizon_frames):
            transition=model.step(a)
            for event in transition.evaluator_events:
                if event.event is KeyEvent.DOWN:
                    first.setdefault(event.key,event.time_s)
        return first

    def action(self, observed, now_s, remaining):
        original=super().action(observed,now_s,remaining)
        commands=list(original.commands)
        # Only forecast if multiple axes with live reservations are near due.
        active=[(axis, plan.tokens[0], plan.due_s) for axis,plan in self.reservations.items()
                if plan.tokens and plan.due_s-now_s <= .080]
        if len(active)<2:
            return original
        self.forecasts+=1
        arrival=self.predict(commands)
        arriving=sorted(((axis,ordinal,due,arrival[axis]) for axis,ordinal,due in active
                         if axis in arrival),key=lambda x:x[3])
        # Detect actual causal order inversions of *distinct* actuator strokes.
        violations=[]
        for i, later in enumerate(arriving):
            for earlier in arriving[:i]:
                if earlier[1]>later[1]:
                    violations.append((earlier,later))
        if not violations:
            return original
        self.inversions+=len(violations)
        # The prematurely arriving later note is the first element, not the second.
        candidates={p[0]:p for p,q in violations if p[1]>q[1]}
        for axis, ordinal, due, eta in sorted(candidates.values(),key=lambda x:x[1],reverse=True):
            idx=PHYSICAL_AXES.index(axis)
            if commands[idx]<=0 or observed.motor.physical.pressed_flags[idx]:
                continue
            # Only block an actuator whose reserved onset is in the future.
            if due<=now_s:
                continue
            commands[idx]=(-1. if observed.motor.physical.positions_m[idx]>0 else 0.)
            self.delays+=1
            if len(self.sample_inversions)<20:
                self.sample_inversions.append({'now_ms':round(now_s*1000,3),'axis':axis,
                 'reserved':ordinal,'projected_keydown_ms':round(eta*1000,3)})
        return ContactAction(tuple(commands),original.thumb_modes,original.edge_enabled)

def run(chart:Path,end_s:float,mode:int,output:Path):
    if output.exists(): raise FileExistsError(output)
    raw=chart.read_bytes()
    segment=build_playable_segment(compile_adofai(parse_adofai_text(raw.decode('utf-8-sig'))),end_s=end_s)
    old_t,old_e=base.ContactOracleTeacher,base.CoupledContactRealChartEnv
    holder=[]
    def make_teacher(k):
        obj=JointArrivalTeacher(k,None);holder.append(obj);return obj
    def make_env(*args,**kwargs):
        env=old_e(*args,**kwargs);holder[-1].motor=env.motor;return env
    base.ContactOracleTeacher=make_teacher
    base.CoupledContactRealChartEnv=make_env
    try:result=base.audit(segment,mode)
    finally:
        base.ContactOracleTeacher=old_t
        base.CoupledContactRealChartEnv=old_e
    teacher=holder[-1]
    result['joint_diagnostics']={k:getattr(teacher,k) for k in ('forecasts','inversions','delays','sample_inversions')}
    result['chart_sha256']=sha256(raw).hexdigest()
    output.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({'summary':{k:result['summary'][k] for k in ('hits','misses','matches','mismatches','unreserved','planned_error_mae_ms','too_early','overload')},'joint_diagnostics':result['joint_diagnostics']},indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('chart',type=Path)
    parser.add_argument('--end-s',type=float,default=None)
    parser.add_argument('--mode',type=int,default=8)
    parser.add_argument('--output',type=Path,default=Path('u1-joint-autonomous.json'))
    args=parser.parse_args()
    run(args.chart,args.end_s,args.mode,args.output)