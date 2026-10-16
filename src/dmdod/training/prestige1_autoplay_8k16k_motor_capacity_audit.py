from __future__ import annotations

"""Stage T: privileged Motor-only N-key actuator-count comparison.

This is expressly NOT a HUD-only or Student controller. Known future note
schedules are granted to the existing teacher to isolate motor throughput.
16 independent simulated actuators are NOT sixteen human fingers or the
user's real-world 16K fingering technique.
"""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.motor.capacity import (
    CenterFirstNKeyTeacher, NKeyRealChartMotorEnv,
    _unresolved_tail, calibrate_n_key_press_lead,
)
from dmdod.motor.keyboard import KeyEvent
from dmdod.motor.n_key import n_key_names, n_key_tiers

from . import prestige1 as p
from . import prestige1_transfer as transfer

VERSION = 'prestige1-stage-t-8k16k-oracle-motor-capacity-v1'
REPORT = 'stage_t_motor_only_8k_vs_16k_capacity_readonly.json'
CONTROL_DT_S = .010
PHYSICS_DT_S = .001
CASES = ((1200., 144), (2000., 144), (4000., 144), (4000., 576))
PROTOCOLS = ((8, 8), (16, 8), (16, 16))


@dataclass(frozen=True)
class ComparisonProtocol:
    physical_keys: int
    active_keys: int

    def __post_init__(self):
        if self.physical_keys not in (8, 16) or self.active_keys not in (8, 16):
            raise ValueError('Unsupported predeclared key configuration')
        if self.active_keys > self.physical_keys:
            raise ValueError('Active keys exceed physical keys')

    @property
    def name(self) -> str:
        return f'{self.physical_keys}k_motor_{self.active_keys}k_allowed'

    @property
    def allowed(self) -> tuple[str, ...]:
        inner_tiers = n_key_tiers(self.physical_keys)[:self.active_keys // 2]
        return tuple(key for pair in inner_tiers for key in pair)


class RestrictedCenterFirstTeacher(CenterFirstNKeyTeacher):
    """Existing teacher's selection and release logic; restrict available tiers."""

    def __init__(self, protocol: ComparisonProtocol):
        super().__init__(protocol.physical_keys)
        self.protocol = protocol
        self.allowed_tiers = self.tiers[:protocol.active_keys // 2]

    def _choose_key(self, observation, occupied: set[str]) -> str | None:
        for index, pair in enumerate(self.allowed_tiers):
            last = self._last_by_tier[index]
            order = (pair[1], pair[0]) if last == pair[0] else pair
            for key in order:
                if key not in occupied and not observation.pressed(key):
                    self._last_by_tier[index] = key
                    return key
        return None


def make_fixture(bpm: float, notes: int):
    if (bpm, notes) not in CASES:
        raise ValueError('Unregistered fixed BPM and note-count fixture')
    raw = json.dumps({
        'angleData': [0.] * notes,
        'settings': {'bpm': bpm, 'pitch': 100., 'countdownTicks': 0},
        'actions': [],
    }, separators=(',', ':'), sort_keys=True)
    chart = build_playable_segment(compile_adofai(parse_adofai_text(raw)))
    assert len(chart.targets) == notes
    return chart, sha256(raw.encode('utf-8')).hexdigest()


def run_case(bpm: float, notes: int, protocol: ComparisonProtocol,
             *, lead_s: float = .044) -> dict:
    if (bpm, notes) not in CASES:
        raise ValueError('Not a predeclared fixture')
    if (protocol.physical_keys, protocol.active_keys) not in PROTOCOLS:
        raise ValueError('Not a predeclared protocol')
    if abs(lead_s - .044) > 1e-12:
        raise ValueError('Only the fixed 44ms cold-lead comparison is allowed')
    chart, fixture_hash = make_fixture(bpm, notes)
    env = NKeyRealChartMotorEnv(chart, key_count=protocol.physical_keys,
                                control_dt_s=CONTROL_DT_S,
                                physics_dt_s=PHYSICS_DT_S)
    teacher = RestrictedCenterFirstTeacher(protocol)
    obs = env.reset()
    allowed = set(protocol.allowed)
    down_counts: Counter[str] = Counter()
    last_down: dict[str, float] = {}
    repeat_intervals_ms: list[float] = []
    action_counts: Counter[str] = Counter()
    first_keydown_events: list[dict] = []
    max_active_positive = max_reserved = 0
    blocked_unique: set[object] = set()
    peak_hand_fatigue = 0.
    max_frames = int((chart.duration_s + 2.) / CONTROL_DT_S) + 200
    for tick in range(max_frames):
        # ORACLE inputs are deliberately allowed in this MOTOR-ONLY probe.
        now = env.privileged_episode_time_s()
        future = _unresolved_tail(env)
        action = teacher.pipeline_action(obs.motor, now_s=now,
                                         targets=future, lead_s=lead_s)
        positive = 0
        for key, command in zip(env.key_names, action.values):
            if abs(command) > 1e-12 and key not in allowed:
                raise RuntimeError('Restricted teacher activated forbidden finger')
            if command > 0:
                action_counts[key] += 1
                positive += 1
        max_active_positive = max(max_active_positive, positive)
        max_reserved = max(max_reserved, len(teacher._reservations))
        reserved = {token for token, _ in teacher._reservations.values()}
        for token, at in future:
            if at <= now + lead_s + 1e-12 and token not in reserved:
                blocked_unique.add(token)
        result = env.step(action)
        # Measured physical KeyDown, independent of evaluator Hit credit.
        for event in env.motor.last_transition.evaluator_events:
            if event.event is not KeyEvent.DOWN:
                continue
            if event.key not in allowed:
                raise RuntimeError('Forbidden key produced a physical KeyDown')
            down_counts[event.key] += 1
            previous = last_down.get(event.key)
            if previous is not None:
                repeat_intervals_ms.append((event.time_s - previous) * 1000.)
            last_down[event.key] = event.time_s
            if len(first_keydown_events) < 16:
                first_keydown_events.append({
                    'key': event.key, 'ms': round(event.time_s * 1000., 6),
                })
        diag = env.motor.diagnostics()
        peak_hand_fatigue = max(peak_hand_fatigue,
                                diag.left_hand_fatigue, diag.right_hand_fatigue)
        obs = result.observation
        if result.done:
            break
    else:
        raise RuntimeError('Capacity episode exceeded time bound')
    stats = env.stats
    if stats.hits + stats.misses != notes:
        raise RuntimeError('Not all physical notes resolved')
    if env.physical_keydowns != sum(down_counts.values()):
        raise RuntimeError('Count of physical KeyDown events disagrees with evaluator')
    return {
        'fixture_bpm': bpm, 'fixture_notes': notes,
        'fixture_sha256': fixture_hash,
        'protocol': protocol.name,
        'physical_key_count': protocol.physical_keys,
        'allowed_key_count': protocol.active_keys,
        'allowed_key_names': sorted(allowed),
        'hit': stats.hits, 'miss': stats.misses,
        'x_accuracy_percent': stats.x_accuracy_percent,
        'physical_keydowns': env.physical_keydowns,
        'max_simultaneous_positive_action_keys': max_active_positive,
        'max_reserved_keys': max_reserved,
        'unique_due_targets_blocked_at_some_point': len(blocked_unique),
        'fingers_with_actual_downs': len(down_counts),
        'physical_downs_per_key': dict(sorted(down_counts.items())),
        'positive_action_frames_per_key': dict(sorted(action_counts.items())),
        'minimum_same_key_down_gap_ms': min(repeat_intervals_ms) if repeat_intervals_ms else None,
        'mean_same_key_down_gap_ms': sum(repeat_intervals_ms) / len(repeat_intervals_ms)
            if repeat_intervals_ms else None,
        'peak_shared_hand_fatigue': peak_hand_fatigue,
        'first_keydowns': first_keydown_events,
        'control_frames': tick + 1,
        'no_motor_reset_between_notes': True,
        'oracle_target_schedule_used_by_teacher': True,
        'student_or_hud_model_executed': False,
    }


def run_audit(config: p.PrestigeConfig) -> dict:
    if (p.KEY_COUNT != 8 or p.CONTROL_DT != CONTROL_DT_S
            or p.PHYSICS_DT != PHYSICS_DT_S):
        raise RuntimeError('Frozen Prestige I control contract changed')
    path = transfer.output_dir(config) / REPORT
    if path.exists():
        raise FileExistsError('Refuse overwrite of original Motor-only 8k/16k diagnostic')
    calibrations = {}
    for keys in (8, 16):
        c = calibrate_n_key_press_lead(keys, control_dt_s=CONTROL_DT_S,
                                       physics_dt_s=PHYSICS_DT_S)
        calibrations[str(keys)] = {
            'lead_ms': c.lead_s * 1000.,
            'individual_cold_keydown_ms': {k: v * 1000. for k, v in c.key_latencies_s},
        }
        if abs(c.lead_s - .044) > 1e-9:
            raise RuntimeError('Cold-lead calibration diverged from fixed 44ms protocol')
    cases = []
    for bpm, notes in CASES:
        group = [run_case(bpm, notes, ComparisonProtocol(keys, active))
                 for keys, active in PROTOCOLS]
        if len({r['fixture_sha256'] for r in group}) != 1:
            raise RuntimeError('Compared different ADOFAI fixture bytes')
        cases.append({'bpm': bpm, 'notes': notes, 'protocol_results': group,
                      'all16_minus_8key_hit': group[2]['hit'] - group[0]['hit'],
                      'all16_minus_16center8_hit': group[2]['hit'] - group[1]['hit']})
        print(f'Motor-only {bpm:>6.0f} BPM {notes:>3} notes: '
              f'8k={group[0]["hit"]}, 16k-center8={group[1]["hit"]}, '
              f'16k-all={group[2]["hit"]} | X '
              f'{group[0]["x_accuracy_percent"]:.2f}/'
              f'{group[1]["x_accuracy_percent"]:.2f}/'
              f'{group[2]["x_accuracy_percent"]:.2f}', flush=True)
    report = {
        'format': VERSION, 'cases': cases, 'calibration': calibrations,
        'motor_only_oracle_teacher': True, 'hud_or_student_used': False,
        'not_a_human_16key_fingering_model': True,
        'policy_control_dt_ms': 10., 'physics_dt_ms': 1.,
        'lead_ms_for_all': 44.,
        'fixed_8k_student_untouched': True, 'stage_a_fail_preserved': True,
        'original_geometry_hybrid_untouched': True,
        'no_training': True, 'no_checkpoint_loaded_or_modified': True,
        'validation_evaluated': False, 'final_evaluated': False,
        'limitations': [
            'Exact future target timestamps and floor indices are given to the oracle teacher; not HUD-only inference',
            '16-key NKeyBody creates 16 independent finger actuators (eight per hand), not real human 16K technique',
            'Same frozen per-hand force budget but coordination coefficients change with number of model fingers',
            '16-key center-8 is a geometry control, not an exact equivalent to the 8-finger body',
            'Simple center-first teacher; no optimal trajectory or physiological fatigue claim',
            'Deterministic straight 180-degree floors only; no special events or realistic patterns',
            'Higher physical Hit count here is not proof that a 16K HUD Student or human can play the chart',
            'All results fixed before evaluation; no learning, Validation, FINAL, or parameter search',
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    p._no_replace_json(path, report)
    return report
