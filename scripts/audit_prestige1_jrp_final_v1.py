"""Locked one-shot Final10 comparison of Timing v1.4 versus causal JRP.

This script performs no fitting or threshold selection.  It runs the same
frozen Prestige v2 Motor checkpoint twice per chart: JRP hidden and JRP visible.
Ground-truth target times remain inside the post-hoc evaluator.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import torch

from dmdod.data.multichart import discover_multichart_dataset
from dmdod.evaluation.jrp_closed_loop_v1 import (
    FrozenPrestigeMotorPolicy,
    run_jrp_closed_loop,
)
from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.real_chart import compile_role, build_validation_segments


VERSION = 'prestige1-jrp-final-v1'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _aggregate(rows: list[dict]) -> dict:
    targets = sum(r['target_count'] for r in rows)
    baseline_matches = sum(r['matched_baseline_elapsed'] for r in rows)
    combined_matches = sum(r['matched_combined'] for r in rows)
    return {
        'charts': len(rows),
        'targets': targets,
        'baseline_matches': baseline_matches,
        'baseline_tracking_rate': baseline_matches / targets if targets else 0.,
        'baseline_false': sum(r['unique_baseline_elapsed'] - r['matched_baseline_elapsed'] for r in rows),
        'combined_matches': combined_matches,
        'combined_tracking_rate': combined_matches / targets if targets else 0.,
        'combined_false': sum(r['false_combined'] for r in rows),
        'jrp_added': sum(r['unique_jrp_forecasts'] for r in rows),
        'jrp_matched': sum(r['matched_jrp_forecasts'] for r in rows),
        'jrp_false': sum(r['false_jrp_forecasts'] for r in rows),
        'motor_hits': sum(r['actual_hits'] for r in rows),
        'motor_misses': sum(r['actual_misses'] for r in rows),
        'motor_too_early': sum(r['actual_too_early'] for r in rows),
        'motor_overloads': sum(bool(r['actual_overload']) for r in rows),
        'physical_keydowns': sum(r['physical_keydowns'] for r in rows),
    }


def audit(config_path: Path, checkpoint: Path, *, device: str) -> dict:
    config = PrestigeConfig.load(config_path)
    dataset = discover_multichart_dataset(config.dataset)
    # The selection is locked before any Final result is observed: all Final10,
    # deterministic middle 8-second windows, identical to the old Validation
    # window rule.  No metric from this run changes a threshold or model.
    segments = build_validation_segments(
        compile_role(dataset.final), window_s=config.validation_window_s)
    if len(segments) != 10:
        raise RuntimeError(f'locked Final10 expected, got {len(segments)}')
    if {x.chart_sha256 for x in segments} & {
            x.content_sha256 for x in (*dataset.train, *dataset.validation)}:
        raise RuntimeError('Final chart overlaps prior Train/Validation')

    policy = FrozenPrestigeMotorPolicy.load(
        checkpoint, config.core_path, device=device, expected_arm='prestige')
    rows = []
    action_parity = True
    for index, named in enumerate(segments, 1):
        hidden = asdict(run_jrp_closed_loop(
            named.segment, policy, policy_label='prestige_v2_jrp_hidden',
            jrp_visible=False))
        hidden_student = {
            'frames': len(policy.estimates),
            'mean_confidence': (sum(x.confidence for x in policy.estimates) /
                                len(policy.estimates)),
        }
        visible = asdict(run_jrp_closed_loop(
            named.segment, policy, policy_label='prestige_v2_jrp_visible',
            jrp_visible=True))
        visible_student = {
            'frames': len(policy.estimates),
            'mean_confidence': (sum(x.confidence for x in policy.estimates) /
                                len(policy.estimates)),
        }
        physical_keys = ('actual_hits', 'actual_misses', 'actual_too_early',
                         'actual_overload', 'physical_keydowns', 'frames')
        same_physical = all(hidden[k] == visible[k] for k in physical_keys)
        action_parity &= same_physical
        rows.append({
            'index': index, 'chart': named.chart_name,
            'chart_sha256': named.chart_sha256,
            'window_s': [named.start_s, named.end_s],
            'same_physical_outcome': same_physical,
            'student_hidden': hidden_student,
            'student_visible': visible_student,
            'hidden': hidden, 'visible': visible,
        })
        print(f'Final {index:02d}/10 {named.chart_name}: '
              f'base={hidden["matched_baseline_elapsed"]}/{hidden["target_count"]} '
              f'combined={visible["matched_combined"]}/{visible["target_count"]} '
              f'jrp_false={visible["false_jrp_forecasts"]}', flush=True)

    hidden_aggregate = _aggregate([r['hidden'] for r in rows])
    visible_aggregate = _aggregate([r['visible'] for r in rows])
    improved = (visible_aggregate['combined_matches'] > hidden_aggregate['baseline_matches']
                and visible_aggregate['combined_false'] <= hidden_aggregate['baseline_false'])
    status = 'PASS' if improved and action_parity else 'FAIL'
    return {
        'version': VERSION,
        'status': status,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'locked_protocol': {
            'role': 'Final', 'charts': 10, 'window': 'middle 8 seconds',
            'selection_or_tuning_on_final': False,
            'student_input': 'unchanged causal 263D HUD',
            'jrp_input': 'rounded visible TBPM/CBPM/KPS strings only',
            'truth_access': 'post-hoc matching only',
            'comparison': 'same checkpoint, Motor, initial state and chart window; JRP visibility only',
        },
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha256(checkpoint),
        'core_sha256': sha256(Path(config.core_path)),
        'device': str(torch.device(device)),
        'physical_outcome_parity': action_parity,
        'baseline': hidden_aggregate,
        'jrp_visible': visible_aggregate,
        'improved': improved,
        'rows': rows,
        'interpretation': (
            'JRP improves held-out event tracking without added false detections.'
            if improved else
            'No held-out improvement was demonstrated; inspect per-chart counterexamples.'),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/training/prestige1.toml'))
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('checkpoints/prestige1/prestige_v2.pt'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--status-file', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite Final audit')
    if args.status_file.exists():
        parser.error('refusing to overwrite run status')
    started = datetime.now(timezone.utc)
    args.status_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = audit(args.config, args.checkpoint, device=args.device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
        exit_code = 0 if report['status'] == 'PASS' else 1
        status = {'pid':os.getpid(), 'started_at_utc':started.isoformat(),
                  'finished_at_utc':datetime.now(timezone.utc).isoformat(),
                  'exit_code':exit_code, 'report':str(args.output)}
        with args.status_file.open('x', encoding='utf-8') as stream:
            json.dump(status, stream, indent=2)
        print(json.dumps({k: report[k] for k in
                         ('status', 'baseline', 'jrp_visible', 'improved')}, indent=2))
        if exit_code:
            raise SystemExit(exit_code)
    except BaseException as exc:
        if not args.status_file.exists():
            status = {'pid':os.getpid(), 'started_at_utc':started.isoformat(),
                      'finished_at_utc':datetime.now(timezone.utc).isoformat(),
                      'exit_code':1, 'report':str(args.output),
                      'error_type':type(exc).__name__, 'error':str(exc)}
            with args.status_file.open('x', encoding='utf-8') as stream:
                json.dump(status, stream, indent=2)
        raise


if __name__ == '__main__':
    main()
