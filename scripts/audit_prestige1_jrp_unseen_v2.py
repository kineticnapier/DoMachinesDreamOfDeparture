"""Checkpoint-unseen matched Motor audit for hit-epoch JRP Timing v2."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

from dmdod.data.multichart import discover_multichart_dataset
from dmdod.evaluation.jrp_closed_loop_v1 import FrozenPrestigeMotorPolicy, run_jrp_closed_loop
from dmdod.evaluation.jrp_unseen_split_v2 import select_unseen_charts
from dmdod.planning.jrp_cbpm_anchor_v2 import CausalJrpTimingBridgeV2, VERSION as V2_VERSION
from dmdod.planning.jrp_cbpm_anchor_v3 import CausalJrpTimingBridgeV3, VERSION as V3_VERSION
from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.real_chart import compile_role, build_validation_segments


VERSION='prestige1-jrp-checkpoint-unseen-v2'


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def aggregate(rows):
    targets=sum(r['target_count'] for r in rows)
    matched=sum(r['matched_combined'] for r in rows)
    baseline=sum(r['matched_baseline_elapsed'] for r in rows)
    return {
        'charts':len(rows),'targets':targets,
        'baseline_matches':baseline,
        'baseline_tracking_rate':baseline/targets if targets else 0.,
        'baseline_false':sum(r['unique_baseline_elapsed']-r['matched_baseline_elapsed'] for r in rows),
        'combined_matches':matched,
        'combined_tracking_rate':matched/targets if targets else 0.,
        'combined_false':sum(r['false_combined'] for r in rows),
        'jrp_added':sum(r['unique_jrp_forecasts'] for r in rows),
        'jrp_matched':sum(r['matched_jrp_forecasts'] for r in rows),
        'jrp_false':sum(r['false_jrp_forecasts'] for r in rows),
        'motor_hits':sum(r['actual_hits'] for r in rows),
        'motor_misses':sum(r['actual_misses'] for r in rows),
        'motor_too_early':sum(r['actual_too_early'] for r in rows),
        'motor_overloads':sum(bool(r['actual_overload']) for r in rows),
        'physical_keydowns':sum(r['physical_keydowns'] for r in rows),
    }


def audit(config_path:Path,checkpoint:Path,*,device:str,partition:str,bridge_version:str):
    config=PrestigeConfig.load(config_path)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    manifest=payload.get('train_anchor_manifest')
    if not isinstance(manifest,list) or len(manifest)!=20:
        raise RuntimeError('checkpoint Train20 manifest missing')
    trained_hashes={row['sha256'] for row in manifest}
    dataset=discover_multichart_dataset(config.dataset)
    selected=select_unseen_charts(dataset.train,trained_hashes,partition)
    segments=build_validation_segments(compile_role(selected),window_s=config.validation_window_s)
    policy=FrozenPrestigeMotorPolicy.load(
        checkpoint,config.core_path,device=device,expected_arm='prestige')
    if bridge_version=='v2':
        bridge_factory=CausalJrpTimingBridgeV2; bridge_label=V2_VERSION
    elif bridge_version=='v3':
        bridge_factory=CausalJrpTimingBridgeV3; bridge_label=V3_VERSION
    else:
        raise ValueError('unsupported bridge version')
    rows=[]
    parity=True
    for index,named in enumerate(segments,1):
        hidden=asdict(run_jrp_closed_loop(
            named.segment,policy,policy_label='prestige_v2_jrp_hidden',
            jrp_visible=False,timing_bridge=bridge_factory()))
        visible=asdict(run_jrp_closed_loop(
            named.segment,policy,policy_label='prestige_v2_'+bridge_label,
            jrp_visible=True,timing_bridge=bridge_factory()))
        physical=('actual_hits','actual_misses','actual_too_early','actual_overload',
                  'physical_keydowns','frames')
        same=all(hidden[k]==visible[k] for k in physical)
        parity &= same
        rows.append({'index':index,'chart':named.chart_name,
            'chart_sha256':named.chart_sha256,'window_s':[named.start_s,named.end_s],
            'same_physical_outcome':same,'hidden':hidden,'visible':visible})
        print(f'{partition} {index:02d}/20 {named.chart_name}: '
              f'base={hidden["matched_baseline_elapsed"]}/{hidden["target_count"]} '
              f'combined={visible["matched_combined"]}/{visible["target_count"]} '
              f'false={visible["false_combined"]}',flush=True)
    base=aggregate([r['hidden'] for r in rows])
    visible=aggregate([r['visible'] for r in rows])
    improved=(visible['combined_matches']>base['baseline_matches'] and
              visible['combined_false']<=base['baseline_false'])
    return {'version':VERSION,'bridge_version':bridge_label,
        'status':'PASS' if improved and parity else 'FAIL',
        'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'partition':partition,
        'qualification_claim_permitted':partition=='qualification',
        'selection':{'source_role':'Train','checkpoint_train_excluded':True,
            'ordering':'content_sha256','partition_sizes':[20,20,20],
            'window':'middle 8 seconds','no_metric_based_selection':True},
        'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),
        'core_sha256':sha256(Path(config.core_path)),'physical_outcome_parity':parity,
        'baseline':base,'jrp_visible':visible,'improved':improved,'rows':rows}


def main():
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
    if hasattr(sys.stderr,'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8',errors='backslashreplace')
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=Path('configs/training/prestige1.toml'))
    p.add_argument('--checkpoint',type=Path,default=Path('checkpoints/prestige1/prestige_v2.pt'))
    p.add_argument('--device',default='cuda')
    p.add_argument('--partition',choices=('development','qualification'),required=True)
    p.add_argument('--bridge-version',choices=('v2','v3'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--status-file',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or args.status_file.exists():p.error('refusing to overwrite output/status')
    started=datetime.now(timezone.utc);args.status_file.parent.mkdir(parents=True,exist_ok=True)
    try:
        report=audit(args.config,args.checkpoint,device=args.device,
                     partition=args.partition,bridge_version=args.bridge_version)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False)
        code=0 if report['status']=='PASS' else 1
        status={'pid':os.getpid(),'started_at_utc':started.isoformat(),
                'finished_at_utc':datetime.now(timezone.utc).isoformat(),
                'exit_code':code,'report':str(args.output)}
        with args.status_file.open('x',encoding='utf-8') as f:json.dump(status,f,indent=2)
        print(json.dumps({k:report[k] for k in ('status','partition','baseline','jrp_visible','improved')},indent=2))
        if code:raise SystemExit(code)
    except BaseException as exc:
        if not args.status_file.exists():
            status={'pid':os.getpid(),'started_at_utc':started.isoformat(),
                    'finished_at_utc':datetime.now(timezone.utc).isoformat(),
                    'exit_code':1,'report':str(args.output),
                    'error_type':type(exc).__name__,'error':str(exc)}
            with args.status_file.open('x',encoding='utf-8') as f:json.dump(status,f,indent=2)
        raise


if __name__=='__main__':main()
