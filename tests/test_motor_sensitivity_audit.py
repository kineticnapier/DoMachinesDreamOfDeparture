from __future__ import annotations

import csv
from types import SimpleNamespace

import pytest
import torch

from dmdod.adofai_chart import parse_adofai_text
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import compile_adofai
from dmdod.envs.n_key import (
    NKeyPhysicalKeyDownTrace, TooEarlyKeyDownTrace,
)
from dmdod.training.motor_sensitivity_audit import (
    AMPLITUDES, ARMS, Intervention,
    _assert_key_events, _category, _full_baseline_reference, _key_pairs,
    alter_action, interventions, shifted_action, simulate,
    verify_prior_focus_events, verify_recorded_reference,
)


class IdlePolicy(torch.nn.Module):
    key_count=8

    def __init__(self):
        super().__init__()
        self.marker=torch.nn.Parameter(torch.tensor(.0))
        self.initializations=0

    def eval(self):
        return self

    def initial_state(self,device):
        self.initializations+=1
        return torch.zeros(1,device=device)

    def forward_step(self,x,state):
        assert x.numel()==263
        return torch.zeros(8,device=x.device)+self.marker*0.,None,None,state+1.


def _setup():
    chart=parse_adofai_text(
        '{"angleData":[0,90,0,0],"settings":{"bpm":120,"pitch":100,'
        '"countdownTicks":0,"separateCountdownTime":false},"actions":[]}'
    )
    compiled=compile_adofai(chart)
    segment=build_playable_segment(
        compiled,start_s=0.,end_s=compiled.duration_s
    )
    prepared=SimpleNamespace(
        key_count=8,input_dim=263,
        device=torch.device("cpu"),control_dt_s=.01,physics_dt_s=.001,
    )
    named=SimpleNamespace(segment=segment,chart_name="short")
    return prepared,named,IdlePolicy()


def test_amplitude_variants_are_all_in_range_and_key_local():
    cases=interventions(8)
    selected=[x for x in cases if x.kind=="single_key"]
    assert len(selected)==8*4*2
    assert AMPLITUDES==(.02,.05,.10,.20)
    base=(.99,.0,-.5,.0,.0,.0,.0,.0)
    changed=alter_action(base,Intervention("single_key",(0,),.20))
    assert changed[0]==1.0
    assert changed[1:]==base[1:]
    coordinated=alter_action(base,Intervention(
        "coordinated_all",tuple(range(8)),-.10
    ))
    assert all(-1<=a<=1 for a in coordinated)
    assert coordinated[1]==pytest.approx(-.1)


def test_temporal_shift_requires_original_frames_and_never_fills_future():
    values=[(float(i),)*8 for i in range(3)]
    assert shifted_action(values,0,Intervention("time_shift",shift_frames=1),width=8)==(0.,)*8
    assert shifted_action(values,1,Intervention("time_shift",shift_frames=1),width=8)==values[0]
    assert shifted_action(values,0,Intervention("time_shift",shift_frames=-1),width=8)==values[1]
    with pytest.raises(IndexError,match="future"):
        shifted_action(values,2,Intervention("time_shift",shift_frames=-1),width=8)


def test_baseline_open_replay_matches_closed_and_restores_new_rnn():
    prepared,named,policy=_setup()
    closed=simulate(
        prepared,policy,named,case=Intervention("baseline"),
        mode="closed_loop",horizon_s=.2,physics_trace=True,
    )
    assert policy.initializations==1
    assert closed["duration_s"]==pytest.approx(.2,abs=1e-8)
    actions=[tuple(x["action"]) for x in closed["action_frames"]]
    replay=simulate(
        prepared,policy,named,case=Intervention("baseline"),
        mode="open_loop",recorded_actions=actions,horizon_s=.2,
        physics_trace=True,
    )
    assert policy.initializations==1  # no Student inference for open loop
    assert replay["duration_s"]==pytest.approx(closed["duration_s"])
    assert replay["key_events"]==closed["key_events"]
    assert replay["too_early_events"]==closed["too_early_events"]
    assert replay["action_frames"]==closed["action_frames"]
    assert replay["mean_abs_action_change"]==0.
    assert torch.equal(policy.marker.detach(),torch.tensor(0.))


def test_open_loop_never_simulates_unrecorded_tail_and_closed_no_future():
    prepared,named,policy=_setup()
    sequence=[(.3,)*8 for _ in range(6)]
    normal=simulate(
        prepared,policy,named,case=Intervention("baseline"),
        mode="open_loop",recorded_actions=sequence,horizon_s=.4,
    )
    assert normal["duration_s"]==pytest.approx(.06)
    advanced=simulate(
        prepared,policy,named,case=Intervention("time_shift",shift_frames=-2),
        mode="open_loop",recorded_actions=sequence,horizon_s=.4,
    )
    assert advanced["duration_s"]==pytest.approx(.04)
    with pytest.raises(ValueError,match="noncausal"):
        simulate(
            prepared,policy,named,
            case=Intervention("time_shift",shift_frames=-1),
            mode="closed_loop",horizon_s=.2,
        )


def test_closed_loop_intervention_reinfers_each_frame_and_is_not_training():
    prepared,named,policy=_setup()
    baseline=simulate(
        prepared,policy,named,case=Intervention("baseline"),
        mode="closed_loop",horizon_s=.2,
    )
    altered=simulate(
        prepared,policy,named,case=Intervention("coordinated_all",
                            tuple(range(8)),.2),
        mode="closed_loop",horizon_s=.2,
    )
    assert policy.initializations==2
    assert baseline["mean_abs_action_change"]==0.
    assert altered["mean_abs_action_change"]==pytest.approx(.2)
    assert len(altered["action_frames"])<=20
    assert all(-1.<=v<=1. for frame in altered["action_frames"]
               for v in frame["action"])
    assert policy.marker.grad is None


def test_keydown_matching_is_per_physical_key_and_unmatched_is_reported():
    reference={"key_events":[
        {"key":"left_1","time_s":.05,"target_index":0},
        {"key":"left_1","time_s":.10,"target_index":1},
    ]}
    changed={"key_events":[
        {"key":"left_1","time_s":.07,"target_index":0},
        {"key":"right_1","time_s":.09,"target_index":1}
    ]}
    rows,matched,unmatched=_key_pairs(reference,changed)
    assert matched==1 and unmatched==2
    assert next(x for x in rows if x["matched"])["delta_ms"]==pytest.approx(20.)


def test_fail_closed_keydown_count_key_or_time():
    old=[{"key":"left_1","time_s":.040}]
    _assert_key_events(old,[{"key":"left_1","time_s":.040001}],where="test")
    with pytest.raises(RuntimeError,match="FAIL"):
        _assert_key_events(old,[],where="test")
    with pytest.raises(RuntimeError,match="FAIL"):
        _assert_key_events(old,[{"key":"right_1","time_s":.040}],where="test")
    with pytest.raises(RuntimeError,match="FAIL"):
        _assert_key_events(old,[{"key":"left_1","time_s":.048}],where="test")


def test_replay_matching_prior_early_events_and_mismatch():
    prepared,named,policy=_setup()
    baseline=simulate(prepared,policy,named,
        case=Intervention("baseline"),mode="closed_loop",horizon_s=.2)
    from dmdod.envs.n_key import (
        NKeyOverloadTrace, EpisodeTerminationTrace,
    )
    trace=NKeyOverloadTrace(
        termination=EpisodeTerminationTrace(.2,"Time limit",0,0,0,0,0.),
        keydown_times_s=tuple(x["time_s"] for x in baseline["key_events"]),
        too_early_events=(),
        physical_keydowns=tuple(
            NKeyPhysicalKeyDownTrace(e["time_s"],e["key"],0,.5)
            for e in baseline["key_events"]
        ),
    )
    verify_recorded_reference(trace,baseline,horizon=.2,where="test")
    fabricated=dict(baseline,too_early_events=[{
        "key":"left_1","time_s":.01
    }])
    with pytest.raises(RuntimeError,match="TooEarly"):
        verify_recorded_reference(trace,fabricated,horizon=.2,where="test")


def test_no_keydown_overload_avoidance_is_not_a_success():
    before={"overload":True,"keydowns":3,"hits":0}
    suppressed={"overload":False,"keydowns":0,"hits":0}
    assert _category(suppressed,before).startswith(
        "overload_avoided_by_suppression"
    )
    changed={"overload":False,"keydowns":3,"hits":1}
    assert _category(changed,before)!="no_press_not_improved"


def test_oracle_early_keys_are_explicitly_labelled():
    cases=interventions(8,early_key_indices=(1,5,1))
    oracle=[x for x in cases if x.kind=="oracle_previous_early"]
    assert len(oracle)==2
    assert all(x.oracle and x.key_indices==(1,5) for x in oracle)
    assert all(x.oracle for x in cases if x.shift_frames<0)


def test_prior_motor_timeline_verification_uses_saved_events(tmp_path):
    from dmdod.envs.n_key import NKeyOverloadTrace, EpisodeTerminationTrace
    path=tmp_path/"motor_response_comparison.csv"
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["arm","anchor","kind","key","time_s"])
        writer.writeheader()
        writer.writerow({"arm":"best","anchor":6,"kind":"keydown",
                         "key":"left_1","time_s":.08})
    terminal=EpisodeTerminationTrace(.1,"Overload",0,0,1,0,5.)
    tr=NKeyOverloadTrace(
        termination=terminal,keydown_times_s=(.08,),too_early_events=(),
        physical_keydowns=(NKeyPhysicalKeyDownTrace(.08,"left_1",0,.5),)
    )
    from dmdod.training.motor_sensitivity_audit import verify_prior_focus_events
    checked=verify_prior_focus_events(tmp_path,{"best":[tr]*20})
    assert checked==[{"arm":"best","anchor":6,"event_count":1}]
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["arm","anchor","kind","key","time_s"])
        writer.writeheader()
        writer.writerow({"arm":"best","anchor":6,"kind":"keydown",
                         "key":"left_1","time_s":.081})
    with pytest.raises(RuntimeError,match="FAIL"):
        verify_prior_focus_events(tmp_path,{"best":[tr]*20})


def test_full_validation_reproduction_strict_per_anchor():
    from dmdod.envs.n_key import NKeyOverloadTrace, EpisodeTerminationTrace
    stats=SimpleNamespace(
        hits=10,misses=1,too_early_presses=0,overloaded=False
    )
    tr=NKeyOverloadTrace(
        termination=EpisodeTerminationTrace(1.0,"Time limit",10,0,0,2,0),
        keydown_times_s=(),too_early_events=(),physical_keydowns=()
    )
    row={
        "hits":10,"misses":1,"too_early":0,"keydowns":0,
        "overloaded":False,"early_overload":False,
        "overload_at_s":None,"initial_keydown_times_s":[]
    }
    refs={"baseline_anchors":[row.copy() for _ in range(20)]}
    result=[(stats,0)]*20
    segments=[SimpleNamespace(chart_name="a")]*20
    rows=_full_baseline_reference(
        refs,result,[tr]*20,segments,arm="best"
    )
    assert len(rows)==20 and all(x["PASS"] for x in rows)
    refs["baseline_anchors"][5]["hits"]=11
    with pytest.raises(RuntimeError,match="FAIL best anchor #06"):
        _full_baseline_reference(refs,result,[tr]*20,segments,arm="best")


def test_partial_checkpoint_load_failure_restores_best(tmp_path):
    from dmdod.training.motor_sensitivity_audit import _load_models
    from dmdod.training.fixed_state_action_audit import file_sha256
    from dmdod.training.real_chart import clone_model_state

    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.tensor([1.0]))
            self.frozen=torch.nn.Parameter(
                torch.tensor([7.0]),requires_grad=False
            )

        def prepare_recurrent_runtime(self):
            pass

    model=TestModel()
    best=clone_model_state(model)
    origin=tmp_path/"best.pt"
    origin.write_bytes(b"unchanged-selected-best")
    altered={key:v.clone() for key,v in best.items()}
    altered["weight"]=torch.tensor([2.0])
    torch.save({
        "human_visible_checkpoint_role":"diagnostic-candidate",
        "diagnostic_arm":"balanced",
        "diagnostic_source_sha256":file_sha256(origin),
        "model_state":altered
    },tmp_path/"balanced.diagnostic.pt")
    prepared=SimpleNamespace(model=model,source_checkpoint=origin)
    with pytest.raises(FileNotFoundError,match="onset_focused"):
        _load_models(
            prepared,tmp_path,{"source_sha256":file_sha256(origin)}
        )
    for key,value in best.items():
        assert torch.equal(model.state_dict()[key],value)
    assert origin.read_bytes()==b"unchanged-selected-best"


def test_mechanism_summary_does_not_select_a_model():
    from dmdod.training.motor_sensitivity_audit import mechanism_sensitivity
    data=[
        {
            "arm":"best","mode":"open_loop","kind":"single_key",
            "keydowns_delta":1,"too_early_delta":1,"hits_delta":0,
            "overload":False,"baseline_overload_time_s":None,
            "mean_keydown_delta_ms":10.0,
            "classification":"undetermined","oracle":False,
        },
        {
            "arm":"best","mode":"open_loop","kind":"single_key",
            "keydowns_delta":0,"too_early_delta":0,"hits_delta":0,
            "overload":False,"baseline_overload_time_s":None,
            "mean_keydown_delta_ms":None,
            "classification":"undetermined","oracle":False,
        }
    ]
    report=mechanism_sensitivity(data)
    assert len(report)==1
    assert report[0]["n"]==2
    assert report[0]["changed_count_or_score_fraction"]==.5
    assert report[0]["mean_abs_matched_keydown_delta_ms"]==10.
    assert "selected" not in report[0]
