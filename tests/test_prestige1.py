from __future__ import annotations
import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import pytest
import torch
from torch import nn

from dmdod.training import prestige1 as p


def cfg(tmp_path):
    return p.PrestigeConfig(dataset='dataset',output_dir=str(tmp_path),core_path='core.pt',
                            old_best_path=str(tmp_path.parent/'old_best.pt'),device='cpu',
                            stage_a_updates=2,stage_c_updates=2,batch_frames=5,temporal_hidden=8)


def sequence(role='train', anchor=1, t=9):
    x=torch.zeros(t,263)
    x[:,0]=torch.arange(t)*.01
    labels=torch.stack([torch.linspace(.05,.3,t),torch.tensor([1.]*4+[0.]*(t-4)),torch.zeros(t)],dim=-1)
    return p.TemporalSequence(role,anchor,'chart',x,torch.zeros(t,8),labels,torch.ones(t,dtype=torch.bool))


def test_fixed_contract_and_dimensions(tmp_path):
    s=sequence()
    assert s.visible.shape==(9,263)
    with pytest.raises(ValueError):replace(s,visible=torch.zeros(9,264))
    with pytest.raises(ValueError):replace(s,teacher_action=torch.zeros(9,7))
    with pytest.raises(ValueError):replace(s,role='final')
    with pytest.raises(ValueError):replace(s,mask=torch.ones(9,dtype=torch.float32))


def test_immutable_teacher_and_label_inputs_to_readouts():
    s=sequence()
    torch.manual_seed(20)
    m=p.TemporalReadout(8,recurrent=True)
    one=m(s.visible).detach().clone()
    replaced=replace(s,truth=torch.stack((s.truth[:,0]+10,s.truth[:,1],s.truth[:,2]),dim=-1),teacher_action=s.teacher_action+30)
    assert torch.equal(one,m(replaced.visible))
    changed=s.visible.clone();changed[1,0]+=1
    assert not torch.equal(one,m(changed))


def test_target_visibility_and_simultaneous_chord_without_key_identity():
    targets=[SimpleNamespace(episode_time_s=.1,floor_index=9),SimpleNamespace(episode_time_s=.1,floor_index=9),SimpleNamespace(episode_time_s=.25,floor_index=12)]
    label,valid=p.target_labels(targets,0.,8,behind_floors=2,ahead_floors=2)
    assert valid and label==(.1,1.,1.)
    assert p.target_labels(targets,0.,3,behind_floors=2,ahead_floors=2)[1] is False
    assert p.target_labels(targets,.26,12,behind_floors=2,ahead_floors=2)[1] is False


def test_prediction_causality_future_frame_cannot_affect_past():
    x=torch.randn(15,263); x2=x.clone();x2[10:]+=100
    for recurrent in (False,True):
        m=p.TemporalReadout(10,recurrent=recurrent)
        assert torch.allclose(m(x)[:10],m(x2)[:10],atol=1e-6)


def test_temporal_mask_ignores_invalid_truth_and_is_finite():
    s=sequence()
    m=p.TemporalReadout(8,recurrent=True)
    out=m(s.visible)
    mask=s.mask.clone();mask[1]=False
    labels=s.truth.clone(); labels[1,0]=999.
    assert torch.allclose(p.timing_loss(out,s.truth,mask),p.timing_loss(out,labels,mask))
    assert torch.isfinite(p.timing_loss(out,s.truth,mask))
    assert p.timing_loss(out,s.truth,torch.zeros_like(mask)).item()==0


def test_stage_a_train_does_not_modify_validation():
    train=[sequence()]
    val=[sequence('validation')]
    m=p.TemporalReadout(8,recurrent=True)
    before=val[0].visible.clone()
    history=p.train_stage_a_readout(m,train,updates=2,batch_frames=5,learning_rate=3e-4,seed=4)
    assert len(history)==2
    assert torch.equal(before,val[0].visible)
    out=p.evaluate_stage_a({'single':p.TemporalReadout(8,recurrent=False),'gru':m},val,median_s=.12,constant_probability=.5)
    assert out['aggregate']['gru']['valid_frames']==9


def test_all_metrics_mask_unobservable_frames():
    s=sequence();m=p.TemporalReadout(8,recurrent=True)
    observed=p._evaluation(torch.zeros(9),torch.zeros(9),s.truth,s.mask,anchor=1)
    masked=p._evaluation(torch.zeros(9),torch.zeros(9),s.truth,torch.zeros_like(s.mask),anchor=1)
    assert observed['valid_frames']==9
    assert masked['mae_ms'] is None and masked['masked_frames']==9
    assert masked['false_positive_rate'] is None


def test_gate_requires_clear_outperformance_not_assumption(tmp_path):
    config=cfg(tmp_path)
    score={'median':{'mae_ms':110,'false_positive_rate':1.},
           'single':{'mae_ms':90,'false_positive_rate':.2},
           'gru':{'mae_ms':80,'false_positive_rate':.21,'observable_anchors':5,
                  'positive_frames':50,'negative_frames':50,'balanced_accuracy':.75}}
    gate=p.stage_a_gate({'aggregate':score},config)
    assert not gate['passed'] and gate['gate_data_role']=='train_heldout'
    score['gru']['mae_ms']=60
    assert p.stage_a_gate({'aggregate':score},config)['passed']
    score['gru']['balanced_accuracy']=.5
    assert not p.stage_a_gate({'aggregate':score},config)['passed']
    score['gru']['balanced_accuracy']=.75
    score['gru']['false_positive_rate']=.96
    assert not p.stage_a_gate({'aggregate':score},config)['passed']
    score['gru']['false_positive_rate']=.2
    score['gru']['observable_anchors']=2
    assert not p.stage_a_gate({'aggregate':score},config)['passed']

def test_config_enforces_no_physics_change(tmp_path):
    c=cfg(tmp_path)
    f=tmp_path/'config.toml'
    f.write_text('[prestige1]\ndataset="x"\noutput_dir="."\ncore_path="c"\nold_best_path="o"\nphysics_dt_s=0.002\n')
    with pytest.raises(ValueError):p.PrestigeConfig.load(f)
    f.write_text('[prestige1]\ndataset="x"\noutput_dir="."\ncore_path="c"\nold_best_path="o"\n')
    assert p.PrestigeConfig.load(f).control_dt_s==.01


def test_stage_c_rejects_missing_or_failed_stage_a(tmp_path):
    c=cfg(tmp_path)
    with pytest.raises(RuntimeError,match='Stage A not yet completed'):
        p.run_stage_c(c)
    dst=tmp_path/'diagnostics'/'prestige1';dst.mkdir(parents=True)
    (dst/p.STAGE_A_REPORT).write_text(json.dumps({'format':p.VERSION,'stage':'A','gate':{'passed':False}}))
    with pytest.raises(RuntimeError,match='gate did not pass'):
        p.run_stage_c(c)


def test_output_refuses_existing_and_old_best_protected(tmp_path):
    folder=tmp_path/'diagnostics'/'prestige1';folder.mkdir(parents=True)
    with pytest.raises(FileExistsError):p._reserve_output(folder)
    c=replace(cfg(tmp_path),old_best_path=str(tmp_path/'checkpoints'/'prestige1'/'old.pt'))
    with pytest.raises(RuntimeError,match='Old Best'):
        p._protect_old_best(c)


def test_checkpoint_strict_schema_and_config(tmp_path):
    c=cfg(tmp_path)
    md={'seed':c.seed,'config':c.signature()}
    valid={'format':p.VERSION,'stage':'C','arm':'control','metadata':md,
           'model':{'layer.weight':torch.zeros(1)},'optimizer':{'param_groups':[{'params':[]}]},
           'updates':2,'initial_seed':c.seed,'parameter_names':[]}
    p._validate_resume(valid,arm='control',metadata=md,updates=2)
    with pytest.raises(RuntimeError,match='identity changed'):
        p._validate_resume(valid,arm='control',metadata={'seed':c.seed,'config':'bad'},updates=2)
    with pytest.raises(RuntimeError,match='Incomplete'):
        p._validate_resume({'format':p.VERSION},arm='control',metadata=md,updates=2)


class FakeCNS(nn.Module):
    def __init__(self,*,input_dim,key_count,core_path):
        super().__init__()
        self.hidden_dim=4096
        self.input_dim=input_dim
        self.key_count=key_count
        self.sensory=nn.Linear(input_dim,8)
        self.controller=nn.Linear(8,8)
    def initial_state(self,device):return torch.zeros(1,device=device)
    def prepare_recurrent_runtime(self):pass
    def forward_step(self,x,state):
        mean=self.controller(self.sensory(x))
        return mean,torch.ones(8),torch.tensor(0.),state+1


def fake_backbone(monkeypatch):
    pkg=types.ModuleType('dmdod.connectome')
    mod=types.ModuleType('dmdod.connectome.human_visible_policy')
    mod.NKeyHumanVisibleControllerActorCritic=FakeCNS
    monkeypatch.setitem(sys.modules,'dmdod.connectome',pkg)
    monkeypatch.setitem(sys.modules,'dmdod.connectome.human_visible_policy',mod)


def test_fresh_student_fair_initialization_and_outputs(monkeypatch):
    fake_backbone(monkeypatch)
    torch.manual_seed(2401)
    a=p.PrestigeMotorStudent('fake.pt',8)
    torch.manual_seed(2401)
    b=p.PrestigeMotorStudent('fake.pt',8)
    p._identical_initial_states(a,b)
    state=a.initial_state(torch.device('cpu'))
    result=a.forward_step_with_timing(torch.rand(263),state)
    assert result[0].shape==(8,) and result[4].shape==(10,)
    result[0].sum().backward()
    assert a.timing_head.weight.grad is not None
    assert torch.count_nonzero(a.timing_head.weight.grad) > 0


def test_motor_prediction_cannot_receive_truth_even_if_modified(monkeypatch):
    fake_backbone(monkeypatch)
    m=p.PrestigeMotorStudent('fake',8).eval()
    s=sequence()
    first=m.forward_sequence(s.visible,m.initial_state(torch.device('cpu')))[0]
    altered=replace(s,truth=torch.stack((s.truth[:,0]*100,s.truth[:,1],s.truth[:,2]),dim=-1),teacher_action=s.teacher_action+3)
    second=m.forward_sequence(altered.visible,m.initial_state(torch.device('cpu')))[0]
    assert torch.equal(first,second)


def test_optimizer_state_never_inherited_from_old_checkpoint(monkeypatch,tmp_path):
    fake_backbone(monkeypatch)
    torch.manual_seed(4)
    model=p.PrestigeMotorStudent('core',8)
    assert len(list(model.parameters()))>0
    # Only a new AdamW is constructed; no legacy checkpoint loader is called.
    opt=torch.optim.AdamW((w for w in model.parameters() if w.requires_grad),lr=3e-4)
    assert opt.state_dict()['state']=={}


def test_no_final_or_legacy_policy_checkpoint_source():
    source=Path(p.__file__).read_text()
    assert 'eval_role_final(' not in source
    assert 'dataset.final' not in source
    assert 'model.load_state_dict(checkpoint["model_state"])' not in source
    assert 'run_human_visible_curriculum' not in source


def test_stage_a_cpu_integrated_no_motor_training(monkeypatch,tmp_path):
    config=cfg(tmp_path)
    split={'train':[{'anchor':i,'name':'X','sha256':str(i)} for i in range(1,21)],
           'validation':[{'anchor':i,'name':'Y','sha256':str(i+20)} for i in range(1,21)]}
    def fake_loader(*args,**kwargs):
        return {'train':[sequence('train',i) for i in range(1,21)],
                'validation':[sequence('validation',i) for i in range(1,21)]},split,{}
    monkeypatch.setattr(p,'load_role_sequences',fake_loader)
    monkeypatch.setattr(p,'_metadata',lambda config,signature:{'seed':config.seed,'config_sha256':config.signature(),'split':signature})
    monkeypatch.setattr(p,'_protect_old_best',lambda *_:None)
    result=p.run_stage_a(config)
    assert result['train_fit']['aggregate']['gru']['observable_anchors']==15
    assert result['train_heldout']['aggregate']['gru']['observable_anchors']==5
    assert result['validation']['aggregate']['gru']['observable_anchors']==20
    assert result['gate']['gate_data_role']=='train_heldout'
    assert result['training_coverage']['minimum_full_pass'] is True
    assert all(len(h)==30 for h in result['training_loss'].values())
    root=tmp_path/'diagnostics'/'prestige1'
    assert (root/p.STAGE_A_REPORT).exists()
    for name in ('single','gru'):
        assert p.sha256(root/(name+'_v2.pt'))==result['checkpoint_sha256'][name]
    with pytest.raises(FileExistsError):p.run_stage_a(config)

def test_stage_c_mock_fair_budgets_and_no_validation_updates(monkeypatch,tmp_path):
    fake_backbone(monkeypatch)
    config=replace(cfg(tmp_path),core_path=str(tmp_path/'core.pt'))
    Path(config.core_path).write_bytes(b'fake CNS topology for CPU mock')
    split={'train':[{'anchor':i,'name':'X','sha256':str(i)} for i in range(1,21)],
           'validation':[{'anchor':i,'name':'Y','sha256':str(i+20)} for i in range(1,21)]}
    data={'train':[sequence('train',i) for i in range(1,21)],
          'validation':[sequence('validation',i) for i in range(1,21)]}
    monkeypatch.setattr(p,'load_role_sequences',lambda *args,**kwargs:(data,split,{'validation':[SimpleNamespace(chart_name='chart')]*20}))
    monkeypatch.setattr(p,'_metadata',lambda config,signature:{'seed':config.seed,'config_sha256':config.signature(),'split':signature})
    monkeypatch.setattr(p,'_protect_old_best',lambda *_:None)
    fake_n_key=types.ModuleType('dmdod.training.n_key')
    fake_n_key.n_key_actuation_loss=lambda pred,target:(pred-target).square().mean()
    monkeypatch.setitem(sys.modules,'dmdod.training.n_key',fake_n_key)
    captured=[]
    def mock_evaluate(model,segments,**kwargs):
        assert len(segments)==20 and kwargs['control_dt_s']==.010 and kwargs['physics_dt_s']==.001
        captured.append((kwargs['label'],[w.detach().clone() for w in model.parameters()]))
        stats=SimpleNamespace(hits=2,misses=1,too_early_presses=0,overloaded=False,
                              x_accuracy_percent=82.5,perfect_rate=.5,mean_abs_error_ms=26.0)
        return [(stats,3)]*20
    fake_real=types.ModuleType('dmdod.training.real_chart')
    fake_real.evaluate_role_continuous=mock_evaluate
    monkeypatch.setitem(sys.modules,'dmdod.training.real_chart',fake_real)
    root=tmp_path/'diagnostics'/'prestige1';root.mkdir(parents=True)
    scores={'median':{'mae_ms':150,'false_positive_rate':1.},
            'single':{'mae_ms':120,'false_positive_rate':.15},
            'gru':{'mae_ms':60,'false_positive_rate':.1,'observable_anchors':5,
                   'positive_frames':40,'negative_frames':50,'balanced_accuracy':.85}}
    stage={'format':p.VERSION,'stage':'A','metadata':{**p._metadata(config,split),'old_best_verified_sha256':None},
           'train_heldout':{'aggregate':scores},
           'validation':{'aggregate':{'gru':{'mae_ms':9999.}}},
           'gate':p.stage_a_gate({'aggregate':scores},config)}
    for name in ('single','gru'):
        (root/(name+'_v2.pt')).write_bytes(b'staged fake readout')
    stage['checkpoint_sha256']={name:p.sha256(root/(name+'_v2.pt')) for name in ('single','gru')}
    (root/p.STAGE_A_REPORT).write_text(json.dumps(stage))
    result=p.run_stage_c(config)
    assert len(captured)==2
    assert result['fixed_budget']==2
    assert result['models']['control']['loss_history'][0]['step']==1
    assert len(result['models']['prestige']['loss_history'])==2
    assert result['models']['prestige']['validation']['safe']==20
    assert result['models']['control']['parameter_count']==result['models']['prestige']['parameter_count']
    for arm in ('control','prestige'):
        assert result['models'][arm]['timing_validation']['aggregate']['observable_anchors']==20
        cp=tmp_path/'checkpoints'/'prestige1'/(arm+'.pt')
        assert cp.exists() and p.sha256(cp)==result['models'][arm]['checkpoint_sha256']
    with pytest.raises(FileExistsError):p.run_stage_c(config)

def test_stage_c_rejects_altered_stage_a_hash_before_data_loading(monkeypatch,tmp_path):
    config=cfg(tmp_path)
    root=tmp_path/'diagnostics'/'prestige1';root.mkdir(parents=True)
    scores={'median':{'mae_ms':150,'false_positive_rate':1.},
            'single':{'mae_ms':120,'false_positive_rate':.15},
            'gru':{'mae_ms':60,'false_positive_rate':.1,'observable_anchors':5,
                   'positive_frames':40,'negative_frames':50,'balanced_accuracy':.85}}
    stage={'format':p.VERSION,'stage':'A','metadata':{'config_sha256':config.signature()},
           'train_heldout':{'aggregate':scores},
           'validation':{'aggregate':{'gru':{'mae_ms':9999.}}},
           'gate':p.stage_a_gate({'aggregate':scores},config),
           'checkpoint_sha256':{'single':'0'*64,'gru':'0'*64}}
    (root/p.STAGE_A_REPORT).write_text(json.dumps(stage))
    with pytest.raises(RuntimeError,match='altered or absent'):
        p.run_stage_c(config)




@pytest.mark.parametrize('train_charts,requested,expected_per_chart',[
    (7,0,3), (7,4,4), (21,0,1),
])
def test_uneven_train_chart_count_uses_first_20_prefix(
        monkeypatch,tmp_path,train_charts,requested,expected_per_chart):
    # Real legacy selection makes uniform per-chart windows, then [:20].
    config=replace(cfg(tmp_path),anchors_per_chart=requested)
    fake_data=types.ModuleType('dmdod.data.multichart')
    train=[SimpleNamespace(chart_name=f'train-{i}',chart_sha256=f'train-sha-{i}')
           for i in range(train_charts)]
    validation=[SimpleNamespace(chart_name=f'val-{i}',chart_sha256=f'val-sha-{i}')
                for i in range(20)]
    fake_data.discover_multichart_dataset=lambda _:SimpleNamespace(
        train=train,validation=validation,final=())
    fake_real=types.ModuleType('dmdod.training.real_chart')
    fake_real.compile_role=lambda xs:xs
    requested_counts=[]
    def anchor_builder(charts,*,window_s,anchors_per_chart):
        requested_counts.append(anchors_per_chart)
        return [SimpleNamespace(role=f'anchor-{j+1}',
                    chart_name=chart.chart_name,chart_sha256=chart.chart_sha256)
                for chart in charts for j in range(anchors_per_chart)]
    fake_real.build_anchor_segments=anchor_builder
    fake_real.build_validation_segments=lambda charts,**_: [
        SimpleNamespace(role='validation',chart_name=chart.chart_name,
                        chart_sha256=chart.chart_sha256) for chart in charts]
    monkeypatch.setitem(sys.modules,'dmdod.data.multichart',fake_data)
    monkeypatch.setitem(sys.modules,'dmdod.training.real_chart',fake_real)
    captured=[]
    def capture(named,*,role,anchor,lead_s,device):
        captured.append((role,anchor,named.chart_name))
        return sequence(role=role,anchor=anchor)
    monkeypatch.setattr(p,'_collect_one',capture)
    sequences,signature,segments=p.load_role_sequences(config,device=torch.device('cpu'))
    expected=[f'train-{i}' for i in range(train_charts)
              for _ in range(expected_per_chart)][:20]
    assert requested_counts==[expected_per_chart]
    assert [s.chart_name for s in segments['train']]==expected
    assert len(sequences['train'])==len(signature['train'])==20
    assert len(sequences['validation'])==len(signature['validation'])==20
    assert [a for role,a,_ in captured if role=='train']==list(range(1,21))
    assert all(name.startswith('val-') for name in
               (s.chart_name for s in segments['validation']))


def test_uneven_train_chart_count_does_not_fabricate_missing_windows(monkeypatch,tmp_path):
    config=cfg(tmp_path)
    fake_data=types.ModuleType('dmdod.data.multichart')
    fake_data.discover_multichart_dataset=lambda _:SimpleNamespace(
        train=[SimpleNamespace(chart_name=f'train-{i}',chart_sha256=f'T{i}')
               for i in range(7)],
        validation=[SimpleNamespace(chart_name=f'val-{i}',chart_sha256=f'V{i}')
                    for i in range(20)], final=())
    fake_real=types.ModuleType('dmdod.training.real_chart')
    fake_real.compile_role=lambda charts:charts
    # Mimic _anchor_windows() de-duplicating windows for too-short charts.
    fake_real.build_anchor_segments=lambda charts,**_: [
        SimpleNamespace(role='anchor-1',chart_name=c.chart_name,chart_sha256=c.chart_sha256)
        for c in charts]
    fake_real.build_validation_segments=lambda charts,**_:charts
    monkeypatch.setitem(sys.modules,'dmdod.data.multichart',fake_data)
    monkeypatch.setitem(sys.modules,'dmdod.training.real_chart',fake_real)
    with pytest.raises(RuntimeError,match='only 7 generated'):
        p.load_role_sequences(config,device=torch.device('cpu'))


def test_dataset_split_detection_prevents_train_validation_contamination(monkeypatch,tmp_path):
    c=cfg(tmp_path)
    # Role-specific truth remains separate; detector refuses checksum overlap.
    class Chart:
        def __init__(self,role,digest):
            self.role='anchor-1' if role=='train' else role
            self.chart_sha256=digest
            self.chart_name='chart'
    dataset=SimpleNamespace(train=[Chart('train','X')]*20,
                            validation=[Chart('validation','X')]*20,final=[])
    fake_data=types.ModuleType('dmdod.data.multichart')
    fake_data.discover_multichart_dataset=lambda _:dataset
    fake_real=types.ModuleType('dmdod.training.real_chart')
    fake_real.compile_role=lambda charts:charts
    fake_real.build_anchor_segments=lambda charts,**kwargs:charts
    fake_real.build_validation_segments=lambda charts,**kwargs:charts
    monkeypatch.setitem(sys.modules,'dmdod.data.multichart',fake_data)
    monkeypatch.setitem(sys.modules,'dmdod.training.real_chart',fake_real)
    with pytest.raises(RuntimeError,match='overlap'):
        p.load_role_sequences(c,device=torch.device('cpu'))
