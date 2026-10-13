from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dmdod.training import prestige1 as p


def make_sequence(anchor, *, role='train'):
    x=torch.zeros(10,263)
    x[:,0]=torch.arange(10,dtype=torch.float32)*.01
    truth=torch.tensor([[.04,1.,0.],[.09,1.,1.],[.12,1.,0.],[.16,1.,0.]]
                       + [[.2,0.,0.]]*6,dtype=torch.float32)
    return p.TemporalSequence(role,anchor,'test',x,torch.zeros(10,8),
                              truth,torch.ones(10,dtype=torch.bool))


def test_targets_near_far_and_observation_mask():
    target=SimpleNamespace(episode_time_s=.7,floor_index=10)
    truth,seen=p.target_labels([target],0.,10,behind_floors=2,ahead_floors=2)
    assert seen and truth==(.2,0.,0.)
    truth,seen=p.target_labels([target],.55,10,behind_floors=2,ahead_floors=2)
    assert seen and truth[1:]==(1.,0.) and truth[0]==pytest.approx(.15)
    truth,seen=p.target_labels([target],0.,0,behind_floors=2,ahead_floors=2)
    assert seen is False
    chord=[SimpleNamespace(episode_time_s=.1,floor_index=10) for _ in range(3)]
    truth,seen=p.target_labels(chord,0.,10,behind_floors=2,ahead_floors=2)
    assert seen and truth==(.1,1.,2.)


def test_negative_frames_only_contribute_classifier_loss():
    seq=make_sequence(1)
    torch.manual_seed(1)
    model=p.TemporalReadout(8,recurrent=True)
    out=model(seq.visible)
    changed=seq.truth.clone()
    changed[4:,0]=400.
    changed[4:,2]=7.
    assert torch.allclose(p.timing_loss(out,seq.truth,seq.mask),
                          p.timing_loss(out,changed,seq.mask))
    no_pos=seq.mask & (seq.truth[:,1]<.5)
    assert torch.isfinite(p.timing_loss(out,seq.truth,no_pos))


def test_metrics_count_near_events_and_negatives_separately():
    labels=torch.tensor([[.08,1.,0.],[.15,1.,0.],[.2,0.,0.],[.2,0.,0.]])
    predicted=torch.tensor([.09,.17,0.,0.])
    probability=torch.tensor([.9,.1,.8,.1])
    row=p._evaluation(predicted,probability,labels,torch.ones(4,dtype=torch.bool),anchor=1)
    assert (row['positive_frames'],row['negative_frames'])==(2,2)
    assert (row['tp'],row['tn'],row['fp'],row['fn'])==(1,1,1,1)
    assert row['mae_ms']==pytest.approx(15.,abs=.001)
    assert row['balanced_accuracy']==pytest.approx(.5)
    totals=p.summarize_metrics([row])
    assert totals['mae_ms']==pytest.approx(15.,abs=.001)
    assert totals['false_positive_rate']==pytest.approx(.5)
    assert totals['event_recall']==pytest.approx(.5)
    assert totals['balanced_accuracy']==pytest.approx(.5)


def test_global_metrics_weight_by_positive_and_negative_counts():
    def row(anchor, positives, negatives, good):
        labels=torch.tensor([[.1,1.,0.]]*positives+[[.2,0.,0.]]*negatives)
        times=torch.tensor(([.1 if good else .2]*positives)+[0.]*negatives)
        probabilities=torch.tensor([float(good)]*positives+[float(not good)]*negatives)
        return p._evaluation(times,probabilities,labels,
                             torch.ones(len(labels),dtype=torch.bool),anchor=anchor)
    metrics=p.summarize_metrics([row(1,1,9,True),row(2,9,1,False)])
    assert (metrics['positive_frames'],metrics['negative_frames'])==(10,10)
    assert metrics['mae_ms']==pytest.approx(90.,abs=.001)
    assert metrics['event_recall']==pytest.approx(.1)
    assert metrics['false_positive_rate']==pytest.approx(.1)


def test_holdout_is_chart_disjoint_deterministic_and_train_only():
    sequences=[make_sequence(i) for i in range(1,21)]
    manifest={'train':[{'anchor':i,'sha256':f'chart-{(i-1)//2}'} for i in range(1,21)]}
    fit,heldout,part=p.split_stage_a_train(sequences,manifest,seed=2401)
    assert len(fit)+len(heldout)==20
    assert len(fit)>=10 and len(heldout)>=5
    assert set(part['fit_chart_hashes']).isdisjoint(part['heldout_chart_hashes'])
    assert set(part['fit_anchor_ids']).isdisjoint(part['heldout_anchor_ids'])
    assert p.split_stage_a_train(sequences,manifest,seed=2401)[2]==part
    assert part['validation_used_for_selection'] is False


def test_stage_a_v2_preserves_stage_a_v1_outputs(monkeypatch,tmp_path):
    folder=tmp_path/'diagnostics'/'prestige1'
    folder.mkdir(parents=True)
    (folder/'stage_a.json').write_text('saved v1 report')
    (folder/'gru.pt').write_bytes(b'saved v1 weights')
    config=p.PrestigeConfig(dataset='test',output_dir=str(tmp_path),
                             core_path='core.pt',old_best_path=str(tmp_path.parent/'old.pt'),
                             device='cpu',stage_a_updates=2,batch_frames=5,temporal_hidden=8)
    signature={
      'train':[{'anchor':i,'sha256':f'T{i}'} for i in range(1,21)],
      'validation':[{'anchor':i,'sha256':f'V{i}'} for i in range(1,21)],
    }
    data={'train':[make_sequence(i) for i in range(1,21)],
          'validation':[make_sequence(i,role='validation') for i in range(1,21)]}
    monkeypatch.setattr(p,'load_role_sequences',lambda *_args,**_kwargs:(data,signature,{}))
    monkeypatch.setattr(p,'_metadata',lambda cfg,signature:{'config_sha256':cfg.signature()})
    monkeypatch.setattr(p,'_protect_old_best',lambda *_:None)
    report=p.run_stage_a(config)
    assert (folder/'stage_a.json').read_text()=='saved v1 report'
    assert (folder/'gru.pt').read_bytes()==b'saved v1 weights'
    assert (folder/p.STAGE_A_REPORT).exists()
    assert report['gate']['gate_data_role']=='train_heldout'
    assert report['training_coverage']['updates_per_model']==30
    with pytest.raises(FileExistsError):
        p.run_stage_a(config)


def test_stage_a_gate_rejects_majority_class_and_missing_negative_data(tmp_path):
    cfg=p.PrestigeConfig(dataset='test',output_dir=str(tmp_path),
                        core_path='core',old_best_path='old')
    audit={'aggregate':{
      'median':{'mae_ms':100.},
      'single':{'mae_ms':90.,'false_positive_rate':.99},
      'gru':{'mae_ms':55.,'false_positive_rate':.99,'balanced_accuracy':.505,
             'positive_frames':10,'negative_frames':10,'observable_anchors':5}
    }}
    assert not p.stage_a_gate(audit,cfg)['passed']
    audit['aggregate']['gru']['false_positive_rate']=.3
    audit['aggregate']['gru']['balanced_accuracy']=.75
    assert p.stage_a_gate(audit,cfg)['passed']
    audit['aggregate']['gru']['negative_frames']=0
    assert not p.stage_a_gate(audit,cfg)['passed']
