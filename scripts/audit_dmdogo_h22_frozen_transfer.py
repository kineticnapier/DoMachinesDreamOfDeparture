from __future__ import annotations
"""H2.2: independently test a frozen H2.1 hand-switch rule on another chart.

Read-only; human keydown logs remain local. No selector or model training.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from audit_dmdogo_h20_human_plan_alignment import parse_human, best_offset, sparse_align

SOURCE_CHART = '02971 - ALL MY FELLAS - RetroBean.adofai'
BINS = ('<=20', '(20,50]', '(50,100]', '(100,150]', '(150,300]', '>300')

def gap_bin(ms):
    for limit, name in zip((20, 50, 100, 150, 300), BINS):
        if ms <= limit + 1e-5:
            return name
    return BINS[-1]

def frozen_rule(h21):
    if (h21.get('format') != 'dmdogo-h21-human-transition-rules-v1'
            or h21.get('chart') != SOURCE_CHART):
        raise ValueError('expected frozen H2.1 report for ALL MY FELLAS')
    bins = h21['by_reference_interval_ms']
    if set(bins) != set(BINS):
        raise ValueError('unexpected frozen interval bins')
    rule = {}
    for b in BINS:
        rec = bins[b]
        if rec['pairs'] <= 0:
            raise ValueError('empty H2.1 training bin')
        rule[b] = 2 * rec['human_switches'] > rec['pairs']
    return rule

def motor_events(run):
    if run.get('stage') != 'executed' or run.get('termination_reason') != 'chart_complete':
        raise ValueError('completed executed Motor run required')
    decisions = run.get('decisions', [])
    if not decisions or len(decisions) != run.get('chart_event_count'):
        raise ValueError('incomplete Motor decisions')
    if (run.get('stats', {}).get('hits') != len(decisions)
            or run['stats'].get('misses') != 0 or run.get('physical_local_replan_count') != 0):
        raise ValueError('run must be all-hit with no local replans')
    events = []
    for i, d in enumerate(decisions):
        kd = d.get('keydowns', [])
        if d.get('event_index') != i or len(kd) != 1 or d.get('chosen') != kd[0].get('axis'):
            raise ValueError('missing or ambiguous Motor keydown')
        axis = kd[0]['axis']
        if not (axis.startswith('left_') or axis.startswith('right_')):
            raise ValueError('invalid Motor axis')
        due = float(d['due_ms']) / 1000
        if events and due <= events[-1]['due']:
            raise ValueError('reference times must increase')
        events.append({'due': due, 'time': float(kd[0]['episode_s']),
                       'hand': 'L' if axis.startswith('left_') else 'R'})
    return events

def evaluate(h21, human, run):
    if run.get('chart') == SOURCE_CHART:
        raise ValueError('training chart is not a transfer test')
    rule = frozen_rule(h21)
    events = motor_events(run)
    human_times = [x['time_s'] for x in human]
    due = [x['due'] for x in events]
    offset = best_offset(human_times, due)
    _, aligned = sparse_align(human_times, due, offset)
    pairs = []
    for (ia, ja), (ib, jb) in zip(aligned, aligned[1:]):
        if ib != ia + 1 or jb != ja + 1:
            continue
        bin_name = gap_bin(1000 * (due[jb] - due[ja]))
        human_switch = human[ia]['hand'] != human[ib]['hand']
        motor_switch = events[ja]['hand'] != events[jb]['hand']
        pairs.append((bin_name, human_switch, motor_switch))
    def score(seq):
        n = len(seq)
        frozen = sum(rule[b] == h for b, h, m in seq)
        actual = sum(m == h for b, h, m in seq)
        return {'pairs': n, 'human_switches': sum(h for b, h, m in seq),
                'frozen_correct': frozen, 'motor_correct': actual,
                'frozen_accuracy_percent': round(100 * frozen / n, 3) if n else None,
                'motor_accuracy_percent': round(100 * actual / n, 3) if n else None}
    return {
        'format': 'dmdogo-h22-frozen-rule-transfer-v1',
        'chart': run['chart'], 'chart_sha256': run.get('chart_sha256'),
        'training_chart': SOURCE_CHART, 'frozen_switch_by_interval': rule,
        'human_keydowns': len(human), 'motor_events': len(events),
        'matched_events': len(aligned),
        'unpaired_human': len(human) - len(aligned),
        'unpaired_motor': len(events) - len(aligned),
        'estimated_offset_s': offset, 'overall': score(pairs),
        'by_interval': {b: score([p for p in pairs if p[0] == b]) for b in BINS},
        'limitations': [
            'Single held-out chart, not proof of general human preferences.',
            'Time offset is inferred; unpaired inputs are not game misses.',
            'Only consecutive human and Motor event pairs are scored.',
            'Suspected hardware chatter is not automatically removed.',
            'The Motor run is oracle-supported, not HUD-only.',
            'Frozen rule is evaluated without re-fitting.'
        ],
        'training_performed': False, 'human_likeness_verified': False,
        'human_preference_observed': False
    }

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--h21-report', required=True, type=Path)
    p.add_argument('--human-log', required=True, type=Path)
    p.add_argument('--motor-json', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error('refusing to overwrite output')
    h21 = json.loads(args.h21_report.read_text(encoding='utf-8-sig'))
    human = parse_human(args.human_log.read_text(encoding='utf-8-sig'))
    motor = json.loads(args.motor_json.read_text(encoding='utf-8-sig'))
    report = evaluate(h21, human, motor)
    report['source_sha256'] = {
        'h21': sha256(args.h21_report.read_bytes()).hexdigest(),
        'human': sha256(args.human_log.read_bytes()).hexdigest(),
        'motor': sha256(args.motor_json.read_bytes()).hexdigest()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'chart': report['chart'], 'overall': report['overall'],
                      'saved': str(args.output)}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
