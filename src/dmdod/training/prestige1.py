from __future__ import annotations

"""Prestige I: fresh temporal observability and fixed-budget motor BC.

The ONLY Student input is the causal 263D visible HUD. Privileged target
schedules are consumed exclusively by evaluator-side label construction.
There is no DAgger, checkpoint warm-start, FINAL evaluation or selection.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from bisect import bisect_left
import hashlib
import json
import math
import os
import random
import subprocess
import tomllib

import torch
from torch import nn
from torch.nn import functional as F


INPUT_DIM = 263
KEY_COUNT = 8
CONTROL_DT = .010
PHYSICS_DT = .001
BEST_SHA256 = '476696a2a157c15a00d602845208a21049dcca94479cee9b91bfeb105cac62e0'
VERSION = 'prestige1-temporal-visible-v1'
FORECAST_WINDOW_S = .200
CHORD_CLASSES = 8  # one through eight simultaneous input events; no key identity


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _strict_keys(raw: dict, allowed: set[str], context: str):
    if not set(raw).issubset(allowed):
        raise ValueError(f'{context}: unknown keys {sorted(set(raw)-allowed)}')


@dataclass(frozen=True)
class PrestigeConfig:
    dataset: str
    output_dir: str
    core_path: str
    old_best_path: str
    seed: int = 2401
    device: str = 'cuda'
    train_window_s: float = 8.0
    validation_window_s: float = 8.0
    anchors_per_chart: int = 1
    lead_s: float = .12
    stage_a_updates: int = 40
    stage_c_updates: int = 24
    batch_frames: int = 96
    temporal_hidden: int = 128
    learning_rate: float = 3e-4
    timing_weight: float = .5
    control_dt_s: float = CONTROL_DT
    physics_dt_s: float = PHYSICS_DT
    gate_mae_improvement_fraction: float = .20
    gate_fpr_tolerance: float = .05

    @classmethod
    def load(cls, path: str | Path) -> 'PrestigeConfig':
        with Path(path).open('rb') as f:
            raw = tomllib.load(f)
        _strict_keys(raw, {'prestige1'}, 'top-level config')
        _strict_keys(raw['prestige1'], set(cls.__dataclass_fields__), 'prestige1 config')
        obj = cls(**raw['prestige1'])
        if not all((obj.dataset, obj.output_dir, obj.core_path, obj.old_best_path)):
            raise ValueError('paths must be explicit and non-empty')
        if obj.control_dt_s != CONTROL_DT or obj.physics_dt_s != PHYSICS_DT:
            raise ValueError('Prestige I must retain control 10ms / physics 1ms')
        if obj.device not in ('cpu', 'cuda'):
            raise ValueError('device must be cpu or cuda')
        if min(obj.train_window_s, obj.validation_window_s) <= 0 or obj.anchors_per_chart < 1:
            raise ValueError('invalid data windows')
        if min(obj.stage_a_updates, obj.stage_c_updates, obj.batch_frames, obj.temporal_hidden) < 1:
            raise ValueError('training budget and dimensions must be positive')
        if not (0 < obj.learning_rate <= .01 and 0 < obj.timing_weight <= 10):
            raise ValueError('invalid fixed learning hyperparameters')
        if not (0 < obj.gate_mae_improvement_fraction < 1 and 0 <= obj.gate_fpr_tolerance < 1):
            raise ValueError('invalid Stage A gate')
        return obj

    def signature(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class TemporalSequence:
    role: str
    anchor: int
    name: str
    visible: torch.Tensor  # [T,263] ONLY policy input
    teacher_action: torch.Tensor  # [T,8] BC-only, never timing predictor input
    truth: torch.Tensor  # [T,3] evaluator-only time_s, event_0_200, chord_class
    mask: torch.Tensor  # [T] evaluator-only, observable & unambiguous

    def __post_init__(self):
        t = self.visible.shape[0]
        if self.role not in ('train', 'validation') or not (1 <= self.anchor <= 20):
            raise ValueError('role/anchor outside the frozen Train20/Validation20 split')
        if self.visible.shape != (t, INPUT_DIM) or self.teacher_action.shape != (t, KEY_COUNT):
            raise ValueError('visible/teacher shape contract violated')
        if self.truth.shape != (t, 3) or self.mask.shape != (t,) or t < 1:
            raise ValueError('label or observation mask shape invalid')
        if not torch.isfinite(self.visible).all() or not torch.isfinite(self.teacher_action).all():
            raise ValueError('nonfinite visible HUD or teacher actions')
        if not torch.isfinite(self.truth).all() or self.mask.dtype is not torch.bool:
            raise ValueError('nonfinite labels or invalid observation mask')
        if bool((self.mask & ((self.truth[:, 2] < 0) | (self.truth[:, 2] >= CHORD_CLASSES))).any()):
            raise ValueError('observable chord count outside 1..8')


def target_labels(targets, now_s: float, current_floor: int,
                  *, behind_floors: int, ahead_floors: int):
    """Evaluator-only truth. Never return target metadata to the Student."""
    if not targets:
        return (0., 0., 0.), False
    times = [float(target.episode_time_s) for target in targets]
    index = bisect_left(times, now_s - 1e-9)
    if index >= len(targets):
        return (0., 0., 0.), False
    event = targets[index]
    remaining = max(0., times[index] - now_s)
    visible = current_floor - behind_floors <= event.floor_index <= current_floor + ahead_floors
    count = sum(abs(t - times[index]) < 1e-7 for t in times[index:])
    if count > KEY_COUNT:
        visible = False
    return (remaining, float(remaining <= FORECAST_WINDOW_S + 1e-9), float(min(count, KEY_COUNT) - 1)), bool(visible)


def _collect_one(named, *, role: str, anchor: int, lead_s: float, device: torch.device):
    # This is an Expert-controlled *data capture*. It has no Student policy.
    from dmdod.training.n_key import _make_env, _teacher_action
    from dmdod.motor.capacity import CenterFirstNKeyTeacher
    from dmdod.envs.n_key import encode_n_key_hud_real_chart_observation
    from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG as floors
    env = _make_env(named.segment, key_count=KEY_COUNT, control_dt_s=CONTROL_DT, physics_dt_s=PHYSICS_DT)
    teacher = CenterFirstNKeyTeacher(KEY_COUNT)
    observation = env.reset()
    xs, actions, labels, masks = [], [], [], []
    max_steps = math.ceil((named.segment.duration_s + 2.0) / CONTROL_DT) + 200
    for _ in range(max_steps):
        visible = encode_n_key_hud_real_chart_observation(observation)
        if len(visible) != INPUT_DIM:
            raise RuntimeError('263D visible HUD changed')
        # Access to episode clock and chart is confined to truth generation.
        now = float(env.privileged_episode_time_s())
        current_floor = env._floor_index_at_chart_time(named.segment.chart_time_from_episode(now))
        truth, mask = target_labels(named.segment.targets, now, current_floor,
                                    behind_floors=floors.behind_floors, ahead_floors=floors.ahead_floors)
        act = _teacher_action(env, teacher, observation, lead_s=lead_s)
        xs.append(visible)
        actions.append(act.as_tuple())
        labels.append(truth)
        masks.append(mask)
        step = env.step(act)
        observation = step.observation
        if step.done:
            break
    else:
        raise RuntimeError('Expert capture exceeded step budget')
    return TemporalSequence(role, anchor, named.chart_name,
        torch.tensor(xs, dtype=torch.float32, device=device),
        torch.tensor(actions, dtype=torch.float32, device=device),
        torch.tensor(labels, dtype=torch.float32, device=device),
        torch.tensor(masks, dtype=torch.bool, device=device))


def load_role_sequences(config: PrestigeConfig, *, device: torch.device):
    from dmdod.data.multichart import discover_multichart_dataset
    from dmdod.training.real_chart import compile_role, build_anchor_segments, build_validation_segments
    dataset = discover_multichart_dataset(config.dataset)
    train = build_anchor_segments(compile_role(dataset.train), window_s=config.train_window_s,
                                  anchors_per_chart=config.anchors_per_chart)
    validation = build_validation_segments(compile_role(dataset.validation), window_s=config.validation_window_s)
    if len(train) != 20 or len(validation) != 20:
        raise RuntimeError(f'Fail Closed: need Train20/Validation20, got {len(train)}/{len(validation)}')
    if any(n.role.lower() != 'train' for n in train) or any(n.role.lower() != 'validation' for n in validation):
        raise RuntimeError('Train/Validation role contamination')
    if {n.chart_sha256 for n in train} & {n.chart_sha256 for n in validation}:
        raise RuntimeError('Train/Validation overlap')
    sequences = {}
    for role, items in (('train', train), ('validation', validation)):
        seqs = []
        for i, named in enumerate(items, 1):
            seqs.append(_collect_one(named, role=role, anchor=i, lead_s=config.lead_s, device=device))
            print(f'Prestige I capture {role} {i:02d}/20, frames={seqs[-1].visible.shape[0]}', flush=True)
        sequences[role] = seqs
    signature = {'train': [dict(anchor=i, name=s.chart_name, sha256=s.chart_sha256) for i,s in enumerate(train,1)],
                 'validation': [dict(anchor=i, name=s.chart_name, sha256=s.chart_sha256) for i,s in enumerate(validation,1)]}
    return sequences, signature, {'train':train, 'validation':validation}


class TemporalReadout(nn.Module):
    def __init__(self, hidden: int, *, recurrent: bool):
        super().__init__()
        self.recurrent = bool(recurrent)
        self.hidden = int(hidden)
        if recurrent:
            self.encoder = nn.GRU(INPUT_DIM, hidden, batch_first=True)
        else:
            self.encoder = nn.Sequential(nn.Linear(INPUT_DIM, hidden), nn.GELU(),
                                         nn.Linear(hidden, hidden), nn.GELU())
        self.head = nn.Linear(hidden, 2 + CHORD_CLASSES)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != INPUT_DIM:
            raise ValueError('TemporalReadout input must be causal [T,263]')
        if self.recurrent:
            encoded, _ = self.encoder(x.unsqueeze(0))
            encoded = encoded.squeeze(0)
        else:
            encoded = self.encoder(x)
        return self.head(encoded)


def decode_temporal(raw):
    # Time is nonnegative, event logit and chord logits unconstrained.
    return F.softplus(raw[..., 0]) * FORECAST_WINDOW_S, raw[..., 1], raw[..., 2:]


def timing_loss(raw, labels, mask):
    if raw.shape[:-1] != labels.shape[:-1] or raw.shape[-1] != 10 or labels.shape[-1] != 3:
        raise ValueError('Timing head and labels must be [T,10], [T,3]')
    if mask.shape != labels.shape[:-1] or mask.dtype != torch.bool:
        raise ValueError('Timing mask incorrect')
    if not bool(mask.any()):
        # Creates a graph-connected zero, but callers should skip unlabeled chunks.
        return raw.sum() * 0.
    predicted_s, event_logit, chord_logits = decode_temporal(raw[mask])
    target = labels[mask]
    return (F.smooth_l1_loss(predicted_s / FORECAST_WINDOW_S, target[:, 0] / FORECAST_WINDOW_S)
            + F.binary_cross_entropy_with_logits(event_logit, target[:, 1])
            + .2 * F.cross_entropy(chord_logits, target[:, 2].long()))


def _evaluation(predicted_time, event_probability, labels, mask, *, anchor: int, chord_pred=None):
    if not bool(mask.any()):
        return {'anchor':anchor, 'valid_frames':0, 'masked_frames':int(mask.numel()),
                'mae_ms':None, 'within_50ms':None, 'within_100ms':None,
                'within_200ms':None, 'event_recall':None, 'false_positive_rate':None, 'chord_accuracy':None}
    time_err = (predicted_time[mask] - labels[mask,0]).abs()
    observed_events = labels[mask,1] >= .5
    predicted_events = event_probability[mask] >= .5
    recall = float((observed_events & predicted_events).sum()/observed_events.sum()) if bool(observed_events.any()) else None
    negatives = ~observed_events
    fpr = float((negatives & predicted_events).sum()/negatives.sum()) if bool(negatives.any()) else None
    return {'anchor':anchor, 'valid_frames':int(mask.sum()), 'masked_frames':int((~mask).sum()),
            'mae_ms':float(1000*time_err.mean()),
            'within_50ms':float((time_err <= .050).float().mean()),
            'within_100ms':float((time_err <= .100).float().mean()),
            'within_200ms':float((time_err <= .200).float().mean()),
            'event_recall':recall, 'false_positive_rate':fpr,
            'chord_accuracy':float((chord_pred[mask]==labels[mask,2].long()).float().mean()) if chord_pred is not None else None}


def summarize_metrics(rows):
    valid = [r for r in rows if r['valid_frames']]
    if not valid:
        raise RuntimeError('Fail Closed: zero temporally observable frames')
    total=sum(x['valid_frames'] for x in valid)
    weighted=lambda key: sum(x[key]*x['valid_frames'] for x in valid if x[key] is not None)/sum(x['valid_frames'] for x in valid if x[key] is not None) if any(x[key] is not None for x in valid) else None
    return {key:weighted(key) for key in ('mae_ms','within_50ms','within_100ms','within_200ms','event_recall','false_positive_rate','chord_accuracy')} | {'valid_frames':total,'masked_frames':sum(x['masked_frames'] for x in rows),'observable_anchors':len(valid)}


def _fixed_chunks(sequences, frames, *, require_labels=True):
    options=[]
    for seq in sequences:
        for start in range(0, seq.visible.shape[0], frames):
            end=min(start+frames, seq.visible.shape[0])
            if not require_labels or bool(seq.mask[start:end].any()): options.append((seq,start,end))
    if not options: raise RuntimeError('Fail Closed: no Train labels visible')
    return options


def train_stage_a_readout(model, train, *, updates:int, batch_frames:int, learning_rate:float, seed:int):
    choices=_fixed_chunks(train,batch_frames)
    rng=random.Random(seed)
    order=list(range(len(choices)))
    rng.shuffle(order)
    opt=torch.optim.AdamW(model.parameters(),lr=learning_rate)
    history=[]
    model.train()
    for step in range(updates):
        if step and step % len(order)==0: rng.shuffle(order)
        seq,start,end=choices[order[step%len(order)]]
        opt.zero_grad(set_to_none=True)
        raw=model(seq.visible[start:end])
        loss=timing_loss(raw,seq.truth[start:end],seq.mask[start:end])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        opt.step()
        history.append(float(loss.detach()))
    return history


@torch.no_grad()
def evaluate_stage_a(model, sequences, *, median_s:float, constant_probability:float, constant_chord:int=0):
    rows={'median':[],'single':[],'gru':[]}
    for seq in sequences:
        mask=seq.mask
        for key in ('median','single','gru'):
            if key == 'median':
                p=seq.truth[:,0].new_full((seq.visible.shape[0],),median_s)
                chance=p.new_full(p.shape,constant_probability)
                chord=torch.full((seq.visible.shape[0],),constant_chord,device=seq.visible.device,dtype=torch.long)
            else:
                model[key].eval()
                raw=model[key](seq.visible)
                p,logit,chord_logits=decode_temporal(raw)
                chance=torch.sigmoid(logit)
                chord=chord_logits.argmax(dim=-1)
            rows[key].append(_evaluation(p,chance,seq.truth,mask,anchor=seq.anchor,chord_pred=chord))
    return {'per_anchor':rows,'aggregate':{k:summarize_metrics(v) for k,v in rows.items()}}


def stage_a_gate(validation:dict, config:PrestigeConfig):
    scores=validation['aggregate']
    simple=min(scores['median']['mae_ms'], scores['single']['mae_ms'])
    gru=scores['gru']
    simple_fpr=scores['single']['false_positive_rate']
    reasons=[]
    if gru['observable_anchors'] < 20: reasons.append('fewer_than_20_observable_validation_anchors')
    if gru['mae_ms'] > simple*(1-config.gate_mae_improvement_fraction):
        reasons.append('gru_mae_not_20pct_better_than_constant_and_single')
    if simple_fpr is not None and (gru['false_positive_rate'] is None or gru['false_positive_rate'] > simple_fpr+config.gate_fpr_tolerance):
        reasons.append('false_positive_rate_regression')
    return {'passed':not reasons,'reasons':reasons, 'criterion':'GRU MAE at least 20% below best median/single; FPR <= single +0.05; 20 observable Validation anchors',
            'validation_used_for_selection':False, 'may_run_stage_c':not reasons}


def _metadata(config,signature):
    git=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=False)
    if git.returncode or len(git.stdout.strip())!=40:
        raise RuntimeError('Git commit SHA unavailable; run from repository root')
    return {'git_commit':git.stdout.strip(), 'config':asdict(config),'config_sha256':config.signature(),
            'dataset_split':signature,'expected_old_best_sha256':BEST_SHA256,
            'seed':config.seed,'control_dt_s':CONTROL_DT,'physics_dt_s':PHYSICS_DT,
            'visible_dim':INPUT_DIM,'keys':KEY_COUNT, 'no_final':True,
            'old_best_loaded_as_student':False}


def _protect_old_best(config):
    old=Path(config.old_best_path)
    if old.exists() and sha256(old)!=BEST_SHA256:
        raise RuntimeError('Fail Closed: old Best SHA256 mismatch')
    protected_outputs=[Path(config.output_dir)/'checkpoints'/'prestige1',
                       Path(config.output_dir)/'diagnostics'/'prestige1']
    if any(old.resolve().is_relative_to(folder.resolve()) for folder in protected_outputs):
        raise RuntimeError('Old Best checkpoint cannot be under Prestige output')
    return sha256(old) if old.exists() else None


def _no_replace_json(path:Path, contents:dict):
    with path.open('x',encoding='utf-8') as f:
        json.dump(contents,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')


def _reserve_output(path:Path):
    if path.exists(): raise FileExistsError('Existing Prestige diagnostic output: '+str(path))
    path.mkdir(parents=True,exist_ok=False)


def run_stage_a(config:PrestigeConfig):
    output=Path(config.output_dir)/'diagnostics'/'prestige1'/'stage_a.json'
    if output.exists() or output.parent.exists():
        raise FileExistsError('Stage A output already exists')
    old_best_verified=_protect_old_best(config)
    torch.manual_seed(config.seed)
    device=torch.device(config.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    sequences,signature,_=load_role_sequences(config,device=device)
    meta=_metadata(config,signature)
    meta['old_best_verified_sha256']=old_best_verified
    masked=torch.cat([s.truth[s.mask,0] for s in sequences['train']])
    if masked.numel()<1:raise RuntimeError('No observable Train labels')
    median=float(masked.median())
    positive=torch.cat([s.truth[s.mask,1] for s in sequences['train']])
    prob=float(positive.mean())
    chord=int(torch.mode(torch.cat([s.truth[s.mask,2].long() for s in sequences['train']])).values)
    models={}
    curves={}
    for name,recurrent in (('single',False),('gru',True)):
        torch.manual_seed(config.seed)
        models[name]=TemporalReadout(config.temporal_hidden,recurrent=recurrent).to(device)
        curves[name]=train_stage_a_readout(models[name],sequences['train'],updates=config.stage_a_updates,
                                           batch_frames=config.batch_frames,learning_rate=config.learning_rate,seed=config.seed)
    reference={'median':median,'event_frequency':prob,'train_chord_mode':chord}
    train=evaluate_stage_a(models,sequences['train'],median_s=median,constant_probability=prob,constant_chord=chord)
    validation=evaluate_stage_a(models,sequences['validation'],median_s=median,constant_probability=prob,constant_chord=chord)
    gate=stage_a_gate(validation,config)
    report={'format':VERSION,'stage':'A','metadata':meta, 'reference':reference,
            'model_parameter_counts':{k:sum(p.numel() for p in m.parameters()) for k,m in models.items()},
            'train':train,'validation':validation,'training_loss':curves,'gate':gate,
            'label_visibility':'next event floor within visible floor window; unobservable frames excluded from all metrics',
            'leakage_contract':'only 263D visible input, truth generated separately from Train/Validation targets'}
    # Save trained Stage A baselines separately, with no motor Student checkpoint.
    _reserve_output(output.parent)
    saved_hashes={}
    for name,m in models.items():
        path=output.parent/(name+'.pt')
        with path.open('xb') as f:torch.save({'format':VERSION,'stage':'A','name':name,'state':m.state_dict(),
            'metadata':meta,'initial_seed':config.seed,'updates':config.stage_a_updates},f)
        saved_hashes[name]=sha256(path)
    report['checkpoint_sha256']=saved_hashes
    _no_replace_json(output,report)
    if _protect_old_best(config) is not None and sha256(Path(config.old_best_path))!=BEST_SHA256:
        raise RuntimeError('Best modified by Stage A')
    return report


class PrestigeMotorStudent(nn.Module):
    """From-scratch MaleCNS4096 visual controller with causal predicted timing."""
    def __init__(self, core_path: str|Path, temporal_hidden:int=128):
        super().__init__()
        from dmdod.connectome.human_visible_policy import NKeyHumanVisibleControllerActorCritic
        self.backbone=NKeyHumanVisibleControllerActorCritic(input_dim=INPUT_DIM,key_count=KEY_COUNT,core_path=core_path)
        if self.backbone.hidden_dim!=4096:
            raise RuntimeError('Prestige I requires MaleCNS4096 topology')
        # Structural fixed MaleCNS weights are new clean buffers, not inherited
        # Student state; legacy sensory path is fixed but freshly initialized.
        for p in self.backbone.sensory.parameters():p.requires_grad_(False)
        self.temporal=nn.GRUCell(INPUT_DIM,temporal_hidden)
        self.timing_head=nn.Linear(temporal_hidden,2+CHORD_CLASSES)
        self.motor_head=nn.Sequential(nn.Linear(KEY_COUNT+temporal_hidden+2+CHORD_CLASSES,128),
                                      nn.GELU(),nn.Linear(128,KEY_COUNT))
        self.key_count=KEY_COUNT
        self.input_dim=INPUT_DIM
        self.temporal_hidden=temporal_hidden
        self._last_timing=None

    def prepare_recurrent_runtime(self):
        self.backbone.prepare_recurrent_runtime()

    def initial_state(self,device):
        return (self.backbone.initial_state(device),torch.zeros(self.temporal_hidden,device=device))

    def forward_step_with_timing(self,x,state):
        if x.shape!=(INPUT_DIM,):raise ValueError('Prestige Student visible HUD must be 263D')
        base_state,temporal_state=state
        mean,std,value,new_base=self.backbone.forward_step(x,base_state)
        new_temporal=self.temporal(x,temporal_state)
        timing=self.timing_head(new_temporal)
        predicted_s,event_logit,chord_logits=decode_temporal(timing)
        # Only PREDICTIONS (no truth labels) enter motor features.
        motor_timing=torch.cat((predicted_s.reshape(1)/FORECAST_WINDOW_S,
                                torch.sigmoid(event_logit).reshape(1),
                                torch.softmax(chord_logits,dim=-1)),dim=0)
        augmented=torch.cat((mean,new_temporal,motor_timing),dim=0)
        return mean+self.motor_head(augmented),std,value,(new_base,new_temporal),timing

    def forward_step(self,x,state):
        mean,std,value,new_state,_=self.forward_step_with_timing(x,state)
        return mean,std,value,new_state

    def forward_sequence(self,x,state):
        if x.ndim!=2 or x.shape[1]!=INPUT_DIM:raise ValueError('Invalid visible sequence')
        means,timing=[],[]
        for frame in x:
            mean,_,_,state,estimate=self.forward_step_with_timing(frame,state)
            means.append(mean)
            timing.append(estimate)
        return torch.stack(means),torch.stack(timing),state


def _identical_initial_states(control,prestige):
    a,b=control.state_dict(),prestige.state_dict()
    if a.keys()!=b.keys():raise RuntimeError('Models have different parameter topology')
    for key in a:
        x,y=a[key],b[key]
        if x.layout == torch.sparse_coo:
            xx=x.coalesce();yy=y.coalesce()
            equal=torch.equal(xx.indices(),yy.indices()) and torch.equal(xx.values(),yy.values())
        elif x.layout == torch.sparse_csr:
            equal=(torch.equal(x.crow_indices(),y.crow_indices())
                   and torch.equal(x.col_indices(),y.col_indices())
                   and torch.equal(x.values(),y.values()))
        else:equal=torch.equal(x,y)
        if not equal:raise RuntimeError('Control/Prestige did not initialize identically: '+key)


def _checkpoint_path(config:PrestigeConfig,arm:str)->Path:
    return Path(config.output_dir)/'checkpoints'/'prestige1'/f'{arm}.pt'


def _validate_resume(payload,*,arm,metadata,updates):
    if not isinstance(payload,dict) or set(('format','stage','arm','metadata','model','optimizer','updates','initial_seed','parameter_names')).difference(payload):
        raise RuntimeError('Incomplete Prestige checkpoint')
    if payload['format']!=VERSION or payload['stage']!='C' or payload['arm']!=arm:
        raise RuntimeError('Prestige checkpoint format/arm mismatch')
    if payload['metadata']!=metadata or payload['initial_seed']!=metadata['seed']:
        raise RuntimeError('Prestige checkpoint dataset/config/Git identity changed')
    if not isinstance(payload['model'],dict) or not payload['model']:
        raise RuntimeError('Incomplete Prestige model state')
    if not isinstance(payload['optimizer'],dict) or not payload['optimizer'].get('param_groups'):
        raise RuntimeError('Incomplete Prestige optimizer')
    if type(payload['updates']) is not int or not (1<=payload['updates']<=updates):
        raise RuntimeError('Prestige checkpoint update count invalid')


def _save_checkpoint(path,content):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:torch.save(content,f)
    return sha256(path)


def _results_to_dict(results,segments):
    rows=[]
    for i,((stats,kd),named) in enumerate(zip(results,segments),1):
        rows.append({'anchor':i,'chart':named.chart_name,'hits':int(stats.hits),
                     'misses':int(stats.misses),'too_early':int(stats.too_early_presses),
                     'overload':bool(stats.overloaded),'keydowns':int(kd),
                     'accuracy':float(stats.x_accuracy_percent),'pp':float(stats.perfect_rate)*100,
                     'timing_mae_ms':float(stats.mean_abs_error_ms) if stats.mean_abs_error_ms is not None else None})
    return rows


@torch.no_grad()
def _motor_timing_metrics(model,seqs):
    rows=[]
    model.eval()
    for seq in seqs:
        state=model.initial_state(seq.visible.device)
        raw=[]
        for frame in seq.visible:
            _,_,_,state,timing=model.forward_step_with_timing(frame,state)
            raw.append(timing)
        when,logit,chord_logits=decode_temporal(torch.stack(raw))
        rows.append(_evaluation(when,torch.sigmoid(logit),seq.truth,seq.mask,anchor=seq.anchor,chord_pred=chord_logits.argmax(dim=-1)))
    return {'per_anchor':rows,'aggregate':summarize_metrics(rows),
            'evaluation_input':'separate frozen Expert-controlled Validation HUD sequences'}


def _aggregate_motor(rows):
    return {'safe':sum(not r['overload'] for r in rows),'hits':sum(r['hits'] for r in rows),
            'misses':sum(r['misses'] for r in rows),'too_early':sum(r['too_early'] for r in rows),
            'overload':sum(r['overload'] for r in rows),'keydowns':sum(r['keydowns'] for r in rows),
            'accuracy_mean':sum(r['accuracy'] for r in rows)/len(rows),'pp_mean':sum(r['pp'] for r in rows)/len(rows)}


def run_stage_c(config:PrestigeConfig, *, resume:bool=False):
    # This file is mandatory: Stage A must explicitly pass before any BC update.
    stage_a=Path(config.output_dir)/'diagnostics'/'prestige1'/'stage_a.json'
    completed_report=stage_a.parent/'stage_c.json'
    if completed_report.exists():raise FileExistsError('Stage C report already exists')
    if not resume and any(_checkpoint_path(config,arm).exists() for arm in ('control','prestige')):
        raise FileExistsError('Stage C checkpoint already exists')
    if not stage_a.is_file():raise RuntimeError('Fail Closed: Stage A not yet completed')
    stage=json.loads(stage_a.read_text('utf-8'))
    if stage.get('format')!=VERSION or stage.get('stage')!='A' or not stage.get('gate',{}).get('passed'):
        raise RuntimeError('Fail Closed: Stage A temporal observability gate did not pass')
    if stage['metadata']['config_sha256']!=config.signature():
        raise RuntimeError('Stage A and Stage C config differ')
    if stage['gate'] != stage_a_gate(stage['validation'],config):
        raise RuntimeError('Stage A saved gate does not match saved Validation metrics')
    checks=stage.get('checkpoint_sha256')
    if not isinstance(checks,dict) or set(checks)!= {'single','gru'}:
        raise RuntimeError('Incomplete Stage A checkpoint provenance')
    for name,expected in checks.items():
        checkpoint=stage_a.parent/(name+'.pt')
        if not checkpoint.is_file() or sha256(checkpoint)!=expected:
            raise RuntimeError('Stage A checkpoint altered or absent')
    old_best_verified=_protect_old_best(config)
    core_file=Path(config.core_path)
    if not core_file.is_file():
        raise FileNotFoundError('MaleCNS4096 core artifact required for Stage C: '+str(core_file))
    device=torch.device(config.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    sequences,signature,segments=load_role_sequences(config,device=device)
    base_metadata=_metadata(config,signature)
    base_metadata['old_best_verified_sha256']=old_best_verified
    if stage['metadata']!=base_metadata:raise RuntimeError('Stage A provenance differs from Stage C')
    metadata={**base_metadata,'core_artifact_sha256':sha256(core_file),
              'backbone':'fresh NKeyHumanVisibleControllerActorCritic MaleCNS4096',
              'fixed_projection_seed':1701,
              'initialization':'torch.manual_seed(config.seed) before EACH fresh model',
              'warm_start':False}
    from dmdod.training.n_key import n_key_actuation_loss
    from dmdod.training.real_chart import evaluate_role_continuous
    all_options=_fixed_chunks(sequences['train'],config.batch_frames,require_labels=False)
    indices=list(range(len(all_options)))
    rng=random.Random(config.seed)
    rng.shuffle(indices)
    schedule=[]
    for update in range(config.stage_c_updates):
        if update and update % len(indices)==0:rng.shuffle(indices)
        schedule.append(indices[update%len(indices)])
    # Both models start from bit-identical randomly initialized weights and
    # fixed connectome buffers. No checkpoint source or optimizer warm start.
    torch.manual_seed(config.seed)
    control=PrestigeMotorStudent(config.core_path,config.temporal_hidden).to(device)
    torch.manual_seed(config.seed)
    prestige=PrestigeMotorStudent(config.core_path,config.temporal_hidden).to(device)
    control.prepare_recurrent_runtime();prestige.prepare_recurrent_runtime()
    _identical_initial_states(control,prestige)
    result={}
    for arm,model in (('control',control),('prestige',prestige)):
        path=_checkpoint_path(config,arm)
        if path.exists():
            if not resume:raise FileExistsError('Existing checkpoint, use explicit --resume only')
        elif resume:raise FileNotFoundError('Resume requested but checkpoint absent')
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config.learning_rate)
        start=0
        logs=[]
        if resume:
            payload=torch.load(path,map_location=device,weights_only=False)
            _validate_resume(payload,arm=arm,metadata=metadata,updates=config.stage_c_updates)
            if payload['parameter_names']!=[n for n,p in model.named_parameters() if p.requires_grad]:
                raise RuntimeError('Resume parameter order mismatch')
            model.load_state_dict(payload['model'],strict=True)
            model.prepare_recurrent_runtime()
            opt.load_state_dict(payload['optimizer'])
            for parameter, moments in opt.state.items():
                for name in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
                    if name in moments and tuple(moments[name].shape)!=tuple(parameter.shape):
                        raise RuntimeError('Resume optimizer moment shape mismatch')
            start=payload['updates']
            logs=payload.get('loss_history',[])
            if len(logs)!=start:raise RuntimeError('Resume loss/update mismatch')
        for update in range(start,config.stage_c_updates):
            seq,left,right=all_options[schedule[update]]
            model.train()
            opt.zero_grad(set_to_none=True)
            means,timing,_=model.forward_sequence(seq.visible[left:right],model.initial_state(device))
            motor_loss=n_key_actuation_loss(torch.tanh(means),seq.teacher_action[left:right])
            auxiliary=timing_loss(timing,seq.truth[left:right],seq.mask[left:right])
            total=motor_loss+(config.timing_weight*auxiliary if arm=='prestige' else 0.)
            if not bool(torch.isfinite(total)):raise RuntimeError('Nonfinite loss')
            total.backward()
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0,error_if_nonfinite=True)
            opt.step()
            logs.append({'step':update+1,'motor_loss':float(motor_loss.detach()),
                         'timing_loss':float(auxiliary.detach()),'total_loss':float(total.detach()),
                         'grad_norm':float(norm)})
            print(f'Prestige {arm} step {update+1}/{config.stage_c_updates}: motor={logs[-1]["motor_loss"]:.4f} timing={logs[-1]["timing_loss"]:.4f}',flush=True)
        checkpoint={'format':VERSION,'stage':'C','arm':arm,'metadata':metadata,
                    'initial_seed':config.seed,'model':model.state_dict(),
                    'optimizer':opt.state_dict(),'updates':config.stage_c_updates,
                    'parameter_names':[n for n,p in model.named_parameters() if p.requires_grad],
                    'loss_history':logs,'parameters':sum(p.numel() for p in model.parameters())}
        if not resume:
            checkpoint_hash=_save_checkpoint(path,checkpoint)
        else:
            # A completed resume is read-only; do not overwrite existing state.
            if start!=config.stage_c_updates:raise RuntimeError('Partial resume requires a new checkpoint path; overwriting forbidden')
            checkpoint_hash=sha256(path)
        if sha256(core_file)!=metadata['core_artifact_sha256']:
            raise RuntimeError('MaleCNS core changed during Stage C training')
        model.eval()
        with torch.no_grad():
            evaluated=evaluate_role_continuous(model,segments['validation'],label='Prestige '+arm,
                    control_dt_s=CONTROL_DT,physics_dt_s=PHYSICS_DT,device=device,verbose=False)
        rows=_results_to_dict(evaluated,segments['validation'])
        result[arm]={'parameter_count':checkpoint['parameters'],'checkpoint':str(path),
                     'checkpoint_sha256':checkpoint_hash,'loss_history':logs,
                     'validation_per_anchor':rows,'validation':_aggregate_motor(rows),
                     'timing_validation':_motor_timing_metrics(model,sequences['validation'])}
    if _protect_old_best(config) is not None and sha256(Path(config.old_best_path))!=BEST_SHA256:
        raise RuntimeError('Old Best modified')
    output=Path(config.output_dir)/'diagnostics'/'prestige1'/'stage_c.json'
    if output.exists():raise FileExistsError('Stage C diagnostics already exist')
    result={'format':VERSION,'metadata':metadata,'fixed_budget':config.stage_c_updates,
            'baseline_old_best_safe':12,'no_final':True,'no_validation_selection':True,
            'models':result,'gate':stage['gate'],
            'experiment':'same fresh initial tensors and Train minibatch schedule, timing loss off/on',
            'timing_metric_context':'held-out Expert-controlled Validation HUD; closed-loop physical metrics separate'}
    _no_replace_json(output,result)
    return result
