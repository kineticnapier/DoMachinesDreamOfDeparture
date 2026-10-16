"""Read-only Stage C v0.6.2: fixed vs live TieAware commands, exact action replay."""
import argparse, copy, json
from pathlib import Path
from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.timing import compile_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.envs.contact_chart_16k import CoupledContactRealChartEnv
from dmdod.motor.contact_16k import PHYSICAL_AXES
import compare_nerfed_tie_teacher as tie


def diagnose(chart, start_s=58.4, end_s=58.65, forecast_frames=7):
    segment=build_playable_segment(compile_adofai(parse_adofai_text(Path(chart).read_text(encoding="utf-8-sig"))),end_s=end_s)
    teacher=tie.TieAwareJointTeacher(8,None)
    env=CoupledContactRealChartEnv(segment,control_dt_s=.010,physics_dt_s=.001)
    obs=env.reset()
    teacher.motor=env.motor
    controls=[]
    comparison=[]
    captured=False
    live_actions=[]
    for tick in range(int((end_s+1)/.010)+5):
        first=env.privileged_next_target()
        remaining=(tuple((int(t.ordinal),float(t.episode_time_s)) for t in segment.targets[int(first.ordinal):]) if first is not None else ())
        now=env.privileged_episode_time_s()
        action=teacher.action(obs,now,remaining)
        if not captured and now+1e-7>=start_s:
            snap=copy.deepcopy(env.motor)
            baseline=copy.deepcopy(snap)
            forecasts=[]
            for j in range(forecast_frames):
                st=baseline.step(action)
                forecasts += [(round(e.time_s*1000,3),e.key,e.event.value) for e in st.logical_events]
            captured=True
            expected=action.commands
            reference=now
        if captured and len(controls)<forecast_frames:
            live_actions.append(action)
            diffs=[PHYSICAL_AXES[i] for i,(a,b) in enumerate(zip(expected,action.commands)) if abs(a-b)>1e-9]
            controls.append({"tick_ms":round(now*1000,3),"changed_axes":diffs,
                             "changed_commands":{a:round(action.commands[PHYSICAL_AXES.index(a)],3) for a in diffs}})
        step=env.step(action)
        if captured and now < reference + forecast_frames*.010 - 1e-7:
            comparison += [(round(e.time_s*1000,3),e.key,e.event.value) for e in env.last_events]
        obs=step.observation
        if step.done:
            break
    if not captured:
        raise ValueError("start_s was not reached")
    replay=copy.deepcopy(snap)
    replay_events=[]
    for command in live_actions:
        st=replay.step(command)
        replay_events += [(round(e.time_s*1000,3),e.key,e.event.value) for e in st.logical_events]
    if replay_events!=comparison:
        raise AssertionError("same-state same-action replay differs from live events")
    return {"matching_action_replay_exact":True,"start_ms":start_s*1000,
            "window_ms":forecast_frames*10,
            "fixed_command_forecast_events":forecasts,
            "actual_teacher_events":comparison,
            "control_changes":controls,
            "changed_ticks":sum(bool(r["changed_axes"]) for r in controls),
            "teacher":"TieAwareJointTeacher","learning":False}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("chart",type=Path)
    parser.add_argument("--start-s",type=float,default=58.4)
    parser.add_argument("--end-s",type=float,default=58.65)
    parser.add_argument("--out",type=Path,default=Path("teacher-divergence-v062.json"))
    args=parser.parse_args()
    if args.out.exists():
        parser.error(f"Refusing to overwrite {args.out}")
    result=diagnose(args.chart,args.start_s,args.end_s)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(result,indent=2,ensure_ascii=False))
