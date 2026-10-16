"""Read-only experimental tie-aware joint arrival teacher. No scorer modification."""
import argparse, json
from pathlib import Path
from hashlib import sha256
from dmdod.motor.n_key import NKeyAction
from dmdod.motor.keyboard import KeyEvent
import copy
import compare_u1_joint_teacher_autonomous as parent

class TieAwareJointTeacher(parent.JointArrivalTeacher):
    def __init__(self, mode, motor):
        super().__init__(mode, motor)
        self.tie_inversions = 0

    def predict(self, commands, horizon_frames=7):
        model=copy.deepcopy(self.motor.motor)
        action=NKeyAction(tuple(commands))
        arrivals={}
        serial=0
        for _ in range(horizon_frames):
            step=model.step(action)
            for ev in step.evaluator_events:
                if ev.event is KeyEvent.DOWN:
                    arrivals.setdefault(ev.key, (ev.time_s, serial))
                    serial+=1
        return arrivals

    def action(self, observed, now_s, remaining):
        # Replicate parent's ordering decision with stable scorer event order
        # as tie-breaker, retaining all original safety guards.
        original=parent.ContactOracleTeacher.action(self,observed,now_s,remaining)
        commands=list(original.commands)
        active=[(axis,p.tokens[0],p.due_s) for axis,p in self.reservations.items()
                if p.tokens and p.due_s-now_s<=.080]
        if len(active)<2:return original
        self.forecasts+=1
        arrivals=self.predict(commands)
        ordered=sorted(((axis,n,due,arrivals[axis]) for axis,n,due in active if axis in arrivals),key=lambda row:row[3])
        bad=[]
        for i,later in enumerate(ordered):
            for earlier in ordered[:i]:
                if earlier[1]>later[1]: bad.append((earlier,later))
        if not bad:return original
        self.inversions+=len(bad)
        self.tie_inversions+=sum(a[3][0]==b[3][0] for a,b in bad)
        candidates={a[0]:a for a,b in bad}
        for axis,n,due,(eta,seq) in sorted(candidates.values(),key=lambda v:v[1],reverse=True):
            idx=parent.PHYSICAL_AXES.index(axis)
            if commands[idx]<=0 or observed.motor.physical.pressed_flags[idx] or due<=now_s:continue
            commands[idx]= -1. if observed.motor.physical.positions_m[idx]>0 else 0.
            self.delays+=1
            if len(self.sample_inversions)<30:
                self.sample_inversions.append({'now_ms':round(now_s*1000,3),'axis':axis,'reserved':n,'projected_keydown_ms':round(eta*1000,3),'same_time':any(a[0]==axis and a[3][0]==b[3][0] for a,b in bad)})
        return parent.ContactAction(tuple(commands),original.thumb_modes,original.edge_enabled)

def run(chart,end_s,out):
    import u1_keydown_attribution_audit as base
    from dmdod.adofai.chart import parse_adofai_text
    from dmdod.adofai.timing import compile_adofai
    from dmdod.adofai.playable import build_playable_segment
    if out.exists(): raise FileExistsError(out)
    raw=chart.read_bytes()
    segment=build_playable_segment(compile_adofai(parse_adofai_text(raw.decode('utf-8-sig'))),end_s=end_s)
    old_t,old_e=base.ContactOracleTeacher,base.CoupledContactRealChartEnv
    holder=[]
    def teacher(k):
        t=TieAwareJointTeacher(k,None);holder.append(t);return t
    def env(*a,**kw):
        e=old_e(*a,**kw);holder[-1].motor=e.motor;return e
    base.ContactOracleTeacher=teacher;base.CoupledContactRealChartEnv=env
    try:r=base.audit(segment,8)
    finally:base.ContactOracleTeacher=old_t;base.CoupledContactRealChartEnv=old_e
    t=holder[-1]
    r['joint_diagnostics']={k:getattr(t,k) for k in ('forecasts','inversions','tie_inversions','delays','sample_inversions')}
    r['chart_sha256']=sha256(raw).hexdigest()
    out.write_text(json.dumps(r,indent=2))
    print(json.dumps({'summary':{k:r['summary'][k] for k in ('hits','misses','matches','mismatches','unreserved','planned_error_mae_ms','too_early','overload')},'joint_diagnostics':r['joint_diagnostics']},indent=2),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('chart',type=Path);ap.add_argument('--end-s',type=float,default=None);ap.add_argument('--output',type=Path,default=Path('nerfed-tie-full.json'));a=ap.parse_args();run(a.chart,a.end_s,a.output)