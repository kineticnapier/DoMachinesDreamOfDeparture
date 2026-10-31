from __future__ import annotations

"""H2.1: read-only first-sight human vs executed Motor hand-transition audit.

Only one Train chart and one human run: measurements are descriptive,
not human preference labels or evidence of across-chart generalization.
"""

import argparse
import csv
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

FORMAT = 'dmdogo-h21-human-transition-rules-v1'
CHART = '02971 - ALL MY FELLAS - RetroBean.adofai'
CANDIDATE = 'e142af91b178-1112-1113'
BINS = ('<=20', '(20,50]', '(50,100]', '(100,150]', '(150,300]', '>300')


def read_events(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'event_index', 'reference_due_s', 'human_index',
                    'human_hand', 'human_finger', 'human_key',
                    'dmd_actual_axis', 'dmd_actual_hand', 'dmd_actual_time_s'}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError('H2.0 CSV required columns absent')
        rows = []
        previous_human_index = -1
        for raw in reader:
            event_idx = int(raw['event_index'])
            if event_idx != len(rows):
                raise ValueError('noncontiguous event indices')
            h_idx = int(raw['human_index']) if raw['human_index'] else None
            if h_idx is not None:
                if h_idx <= previous_human_index:
                    raise ValueError('nonmonotone human event alignment')
                previous_human_index = h_idx
                if raw['human_hand'] not in ('L', 'R'):
                    raise ValueError('invalid human hand')
                if not raw['human_finger'].startswith(raw['human_hand']):
                    raise ValueError('invalid human finger/hand mapping')
            elif raw['human_hand']:
                raise ValueError('unpaired event must not have human hand')
            dmd_axis = raw['dmd_actual_axis']
            dmd_hand = ('L' if dmd_axis.startswith('left_') else
                        'R' if dmd_axis.startswith('right_') else None)
            if raw['dmd_actual_hand'] != dmd_hand:
                raise ValueError('invalid DMD hand/axis mapping')
            due_s = float(raw['reference_due_s'])
            if rows and due_s < rows[-1]['due_s']:
                raise ValueError('nonmonotone chart timeline')
            rows.append({
                'event_index': event_idx, 'due_s': due_s,
                'human_index': h_idx, 'human_hand': raw['human_hand'] or None,
                'human_finger': raw['human_finger'] or None,
                'human_key': raw['human_key'] or None,
                'dmd_axis': dmd_axis, 'dmd_hand': dmd_hand,
                'dmd_time_s': float(raw['dmd_actual_time_s']),
            })
    if not rows:
        raise ValueError('empty CSV')
    return rows


def verify_motor(rows, motor):
    if motor.get('chart') != CHART or motor.get('candidate_id') != CANDIDATE:
        raise ValueError('wrong Motor chart or candidate')
    if motor.get('stage') != 'executed' or motor.get('termination_reason') != 'chart_complete':
        raise ValueError('not a completed Motor execution')
    decisions = motor.get('decisions', [])
    if len(decisions) != len(rows) or motor.get('chart_event_count') != len(rows):
        raise ValueError('Motor event count does not match CSV')
    if motor.get('stats', {}).get('hits') != len(rows) or motor['stats'].get('misses') != 0:
        raise ValueError('Motor not all-hit')
    if motor.get('physical_local_replan_count') != 0:
        raise ValueError('Motor replans found')
    for row, d in zip(rows, decisions):
        if row['event_index'] != d['event_index']:
            raise ValueError('Motor event index mismatch')
        pressed = d.get('keydowns', [])
        if len(pressed) != 1 or pressed[0]['axis'] != row['dmd_axis']:
            raise ValueError('Motor keydown axis mismatch')
        if abs(pressed[0]['episode_s'] - row['dmd_time_s']) > 1e-6:
            raise ValueError('Motor keydown time mismatch')
        if abs(d['due_ms'] / 1000 - row['due_s']) > 2e-5:
            raise ValueError('reference due time mismatch')
        if d['chosen'] != pressed[0]['axis']:
            raise ValueError('Motor chosen/actual mismatch')
    return True


def gap_bin(ms):
    if ms <= 20 + 1e-5:
        return BINS[0]
    if ms <= 50 + 1e-5:
        return BINS[1]
    if ms <= 100 + 1e-5:
        return BINS[2]
    if ms <= 150 + 1e-5:
        return BINS[3]
    if ms <= 300 + 1e-5:
        return BINS[4]
    return BINS[5]


def transitions(rows):
    result = []
    for a, b in zip(rows, rows[1:]):
        # Only event-consecutive and Human-keydown-consecutive pairs.
        # Unpaired events or extra Human keydowns invalidate a transition.
        if (a['human_index'] is None or b['human_index'] is None or
                b['human_index'] != a['human_index'] + 1):
            continue
        dt = 1000 * (b['due_s'] - a['due_s'])
        if dt <= 0:
            raise ValueError('nonpositive reference gap')
        result.append({
            'event_start': a['event_index'],
            'start_s': a['due_s'],
            'gap_ms': dt,
            'gap_bin': gap_bin(dt),
            'rounded_gap_ms': round(dt),
            'human_switch': a['human_hand'] != b['human_hand'],
            'dmd_switch': a['dmd_hand'] != b['dmd_hand'],
            'human_finger_pair': a['human_finger'] + '->' + b['human_finger'],
            'human_key_pair': a['human_key'] + '->' + b['human_key'],
        })
    return result


def describe(items):
    n = len(items)
    hs = sum(x['human_switch'] for x in items)
    ds = sum(x['dmd_switch'] for x in items)
    agree = sum(x['human_switch'] == x['dmd_switch'] for x in items)
    def pct(v):
        return round(100 * v / n, 3) if n else None
    return {'pairs': n, 'human_switches': hs,
            'human_switch_percent': pct(hs),
            'dmd_switches': ds, 'dmd_switch_percent': pct(ds),
            'switch_decisions_agree': agree,
            'switch_decision_agreement_percent': pct(agree)}


def heldout_windows(items, window_s=20):
    """Within-chart leave-one-time-block-out: not an independent holdout."""
    windows = sorted({int(p['start_s'] // window_s) for p in items})
    predictions = []
    for w in windows:
        train = [p for p in items if int(p['start_s'] // window_s) != w]
        test = [p for p in items if int(p['start_s'] // window_s) == w]
        if not train or not test:
            continue
        total = Counter(p['human_switch'] for p in train)
        global_majority = total[True] > total[False]
        by_bin = defaultdict(Counter)
        for p in train:
            by_bin[p['gap_bin']][p['human_switch']] += 1
        for p in test:
            c = by_bin[p['gap_bin']]
            interval_majority = (c[True] > c[False] if c[True] != c[False]
                                 else global_majority)
            predictions.append({
                'window': w,
                'interval_correct': interval_majority == p['human_switch'],
                'global_correct': global_majority == p['human_switch'],
                'dmd_correct': p['dmd_switch'] == p['human_switch'],
            })
    out = {'window_length_s': window_s,
           'distinct_windows': len(windows),
           'evaluated_pairs': len(predictions),
           'warning': ('This is within-one-chart blocked diagnostic; repeated motifs '
                       'can occur in both fit and evaluation windows.')}
    for name in ('interval', 'global', 'dmd'):
        correct = sum(x[name + '_correct'] for x in predictions)
        out[name + '_correct'] = correct
        out[name + '_accuracy_percent'] = round(100 * correct / len(predictions), 3) if predictions else None
    return out


def analyze(rows, motor, *, csv_digest=None, motor_digest=None):
    verify_motor(rows, motor)
    items = transitions(rows)
    if not items:
        raise ValueError('no comparable consecutive keydowns')
    by_bin = {name: describe([p for p in items if p['gap_bin'] == name]) for name in BINS}
    by_gap = defaultdict(list)
    for p in items:
        by_gap[p['rounded_gap_ms']].append(p)
    common_gaps = []
    for ms, group in sorted(by_gap.items(), key=lambda it: (-len(it[1]), it[0])):
        if len(group) < 10:
            continue
        common_gaps.append({'rounded_gap_ms': ms, **describe(group),
                            'top_human_finger_pairs': [
                                {'pair': pair, 'count': count}
                                for pair,count in Counter(x['human_finger_pair'] for x in group).most_common(3)]})
    by_window = []
    for w in sorted({int(p['start_s'] // 20) for p in items}):
        group = [p for p in items if int(p['start_s'] // 20) == w]
        by_window.append({'window_start_s': w*20,
                          'window_end_s': (w+1)*20, **describe(group)})
    human_fingers = Counter(r['human_finger'] for r in rows if r['human_index'] is not None)
    return {'format': FORMAT,
            'source_sha256': {'h20_event_csv': csv_digest, 'h15_actual_motor': motor_digest},
            'chart': CHART, 'chart_sha256': motor['chart_sha256'],
            'candidate_id': CANDIDATE,
            'input': {'reference_events': len(rows),
                      'human_matched_events': sum(r['human_index'] is not None for r in rows),
                      'comparable_consecutive_pairs': len(items),
                      'human_fingers_in_matched_events': dict(human_fingers),
                      'motor_verified_all_hit_exact_trajectory': True},
            'all_pairs': describe(items),
            'by_reference_interval_ms': by_bin,
            'most_frequent_rounded_intervals_ms': common_gaps,
            'by_chart_20s_window': by_window,
            'within_chart_blocked_diagnostic': heldout_windows(items),
            'limitations': [
                'ONE human first-sight play on ONE previously selected Train chart.',
                'Intervals are reference due-time gaps, not human reaction times.',
                'Measured hand transitions are observations, NOT preference labels.',
                'Repeated chart motifs mean pairs and time windows are not statistically independent.',
                'No independent chart or player is evaluated here.',
                'Unpaired keydowns are excluded from transition statistics, not classified as misses.',
                'Suspected hardware chattering is not filtered from the original log.',
                'The DMD run uses a frozen privileged-oracle schedule, not HUD-only policy.',
                'No selector, Motor, timing, model weights, or FINAL/Validation holdout changed.',
                'Human-likeness and physical effort remain unverified.',
            ],
            'human_likeness_verified': False,
            'human_preference_observed': False,
            'hud_only_student_verified': False,
            'training_performed': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--h20-events-csv', required=True, type=Path)
    p.add_argument('--h15-motor-json', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error('output exists; refusing to overwrite')
    csv_bytes = args.h20_events_csv.read_bytes()
    motor_bytes = args.h15_motor_json.read_bytes()
    rows = read_events(args.h20_events_csv)
    motor = json.loads(motor_bytes)
    result = analyze(rows, motor, csv_digest=sha256(csv_bytes).hexdigest(),
                     motor_digest=sha256(motor_bytes).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'all_pairs': result['all_pairs'],
                      'blocked': result['within_chart_blocked_diagnostic'],
                      'saved': str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
