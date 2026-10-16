from __future__ import annotations

"""Read-only Stage T: keep only the safe visible prefix of a Motor preview.

Never use an evaluator timestamp, privileged floor index, or absolute chart
clock to select an action. Baseline Stage T remains completely unchanged.
"""

from collections import Counter
from dataclasses import dataclass
from math import atan2, isfinite, pi

from dmdod.envs.n_key import (
    DiagnosticHudNKeyRealChartMotorEnv,
    encode_n_key_hud_real_chart_observation,
)
from dmdod.motor.capacity import calibrate_n_key_press_lead
from dmdod.motor.n_key import NKeyAction

from . import prestige1 as p
from . import prestige1_autoplay_countdown as cd
from . import prestige1_autoplay_future_events_audit as future
from . import prestige1_autoplay_persistent_motor_audit as baseline
from . import prestige1_transfer as transfer

FORMAT = 'prestige1-stage-t-visible-prefix-motor-audit-v1'
REPORT = 'stage_timing_visible_prefix_motor_comparison_readonly.json'
FIXTURES = ((1200., 18), (2000., 18), (4000., 18),
            (1200., 72), (2000., 72), (4000., 72),
            (1200., 144), (2000., 144), (4000., 144))


@dataclass(frozen=True)
class PrefixForecast:
    floor_offsets: tuple[int, ...]
    times_s: tuple[float, ...]
    status: str
    # This marker is meaningful even when there are zero safe events. A
    # prefix never asserts absence of additional events beyond the visible HUD.
    stop_reason: str

    @property
    def eligible(self) -> bool:
        return self.status in {'eligible_complete', 'visible_prefix'}


def _hypothesis(hud: tuple[float, ...], sign: int,
                bpm: float, horizon: float) -> PrefixForecast:
    velocity = pi * bpm / 60.
    orbit = atan2(hud[cd.MOTOR_FEATURES], hud[cd.MOTOR_FEATURES + 1])
    events: list[float] = []
    offsets: list[int] = []
    elapsed = 0.

    def stopped(reason: str) -> PrefixForecast:
        if events:
            return PrefixForecast(tuple(offsets), tuple(events),
                                  'visible_prefix', reason)
        return PrefixForecast((), (), reason, reason)

    for slot in range(cd.CURRENT_SLOT, cd.CURRENT_SLOT + cd.FLOORS.ahead_floors + 1):
        floor = cd._visible_slot(hud, slot)
        if floor[-1] < .5:
            return stopped('insufficient_lookahead')
        offset = slot - cd.CURRENT_SLOT
        if floor[6] > .5 or floor[7] > .5:
            return stopped('midspin_or_multiplanet')
        markers = [j for j in range(cd.SKIP_MARKER_FROM, cd.SKIP_MARKER_TO)
                   if floor[j] > .5]
        if any(j != cd.SKIP_MARKER_FROM for j in markers) or (offset > 0 and markers):
            return stopped('special_event')
        entry = atan2(floor[2], floor[3])
        exit_angle = atan2(floor[4], floor[5])
        span = (sign * (exit_angle-entry)) % (2*pi)
        if span < future.RAD_EPS or 2*pi-span < future.RAD_EPS:
            return stopped('full_turn_or_zero_arc')
        if offset == 0:
            progress = (sign * (orbit-entry)) % (2*pi)
            if progress < future.RAD_EPS or 2*pi-progress < future.RAD_EPS:
                progress = 0.
            if progress > span + future.RAD_EPS:
                return PrefixForecast((), (), 'incompatible_direction',
                                      'incompatible_direction')
            if span-progress < future.RAD_EPS:
                return PrefixForecast((), (), 'boundary_ambiguous',
                                      'boundary_ambiguous')
            remaining = max(0., span-progress)
        else:
            remaining = span
        until = remaining/velocity
        if elapsed+until > horizon+future.TIME_EPS_S:
            return PrefixForecast(tuple(offsets), tuple(events),
                                  'eligible_complete', 'horizon_reached')
        following = (cd._visible_slot(hud, slot+1)
                     if slot+1 < cd.FLOORS.floor_slots else ())
        if not following or following[-1] < .5:
            return stopped('insufficient_lookahead')
        if any(following[j] > .5 for j in range(6, cd.SKIP_MARKER_TO)):
            return stopped('special_future_floor')
        elapsed += until
        if elapsed <= future.TIME_EPS_S:
            return PrefixForecast((), (), 'boundary_ambiguous',
                                  'boundary_ambiguous')
        events.append(elapsed)
        offsets.append(offset+1)
        if elapsed >= horizon-future.TIME_EPS_S:
            return PrefixForecast(tuple(offsets), tuple(events),
                                  'eligible_complete', 'horizon_reached')
    return stopped('insufficient_lookahead')


def forecast_visible_prefix(hud: tuple[float, ...],
                            horizon: float = baseline.PREVIEW) -> PrefixForecast:
    """Conservative per-frame event forecast, no reliance on future-oracle data.

    Partial hypotheses are accepted only if their whole known event sequence
    agrees across all geometrically possible directions. In particular, a
    shortage of future floor data must never eliminate the opposite direction.
    """
    if not isfinite(horizon) or horizon <= 0.:
        raise ValueError('Invalid forecast horizon')
    if len(hud) != p.INPUT_DIM or not all(isfinite(v) for v in hud):
        return PrefixForecast((), (), 'invalid_hud', 'invalid_hud')
    try:
        bpm = 120.*(2.**hud[cd.HUD_OFFSET])
    except OverflowError:
        return PrefixForecast((), (), 'invalid_bpm', 'invalid_bpm')
    if not isfinite(bpm) or bpm <= 0.:
        return PrefixForecast((), (), 'invalid_bpm', 'invalid_bpm')
    if max(abs(v) for v in hud[cd.MOTOR_FEATURES:cd.MOTOR_FEATURES+2]) < .5:
        return PrefixForecast((), (), 'invalid_orbit', 'invalid_orbit')
    cw = _hypothesis(hud, 1, bpm, horizon)
    ccw = _hypothesis(hud, -1, bpm, horizon)
    viable = [f for f in (cw, ccw) if f.eligible]
    if len(viable) == 1:
        only, other = (cw, ccw) if cw.eligible else (ccw, cw)
        if other.status == 'incompatible_direction':
            return only
        return PrefixForecast((), (), 'direction_not_excluded', other.status)
    if len(viable) == 2:
        if (cw.floor_offsets == ccw.floor_offsets
                and len(cw.times_s) == len(ccw.times_s)
                and all(abs(a-b) <= future.TIME_EPS_S
                        for a,b in zip(cw.times_s,ccw.times_s))):
            if cw.status == ccw.status == 'eligible_complete':
                return cw
            if cw.times_s:
                return PrefixForecast(cw.floor_offsets,cw.times_s,
                                      'visible_prefix', 'direction_agreed_prefix')
            return PrefixForecast((), (), 'insufficient_lookahead',
                                  'no_known_future_event')
        return PrefixForecast((), (), 'ambiguous_direction',
                              'ambiguous_direction')
    # Never assign an action when neither direction supplies a defensible
    # event sequence. The original constraints still apply at current floor.
    reasons = {cw.status, ccw.status}
    for reason in ('boundary_ambiguous','special_event','midspin_or_multiplanet',
                   'full_turn_or_zero_arc','special_future_floor',
                   'insufficient_lookahead'):
        if reason in reasons:
            return PrefixForecast((), (), reason, reason)
    return PrefixForecast((), (), 'no_consistent_direction',
                          'no_consistent_direction')


class PrefixPersistentMotorPlanner(baseline.PersistentMotorPlanner):
    """Same frozen scheduling algorithm; only its HUD forecast is replaced."""

    def __init__(self, calibration):
        super().__init__(calibration)
        self.partial_frames = 0
        self.partial_new_commitments = 0
        self.statuses: Counter[str] = Counter()

    def action(self, hud: tuple[float, ...]) -> NKeyAction:
        now = self.frame * baseline.DT
        self.frame += 1
        estimate = forecast_visible_prefix(hud, self.preview)
        self.statuses[estimate.status] += 1
        if estimate.eligible:
            self.forecast_frames += 1
            if estimate.status == 'visible_prefix':
                self.partial_frames += 1
            for delay in estimate.times_s:
                due = now + delay
                token = self.token(due)
                if token in self.commitments or due <= now + 1e-7:
                    continue
                chosen = None
                for offset in range(len(self.names)):
                    key = self.names[(self.next_key+offset) % len(self.names)]
                    index = self.names.index(key)
                    if (key not in self.reserved_keys
                            and hud[2*p.KEY_COUNT+index] < .5):
                        chosen = key
                        self.next_key = (index+1) % len(self.names)
                        break
                if chosen is None:
                    continue
                self.commitments[token] = baseline.Commitment(due, chosen)
                self.reserved_keys.add(chosen)
                self.planned_unique += 1
                if estimate.status == 'visible_prefix':
                    self.partial_new_commitments += 1
                if due-now < self.latency[chosen]-1e-7:
                    self.deadline_late_at_discovery += 1
        else:
            self.abstain_frames += 1

        values = [0.]*len(self.names)
        for token, item in tuple(sorted(self.commitments.items())):
            index = self.names.index(item.key)
            pressed = hud[2*p.KEY_COUNT+index] > .5
            if pressed:
                item.observed_down = True
            if item.observed_down:
                values[index] = -1.
            elif now+self.latency[item.key] >= item.predicted_s-1e-7:
                values[index] = 1.
                item.commanded = True
            if item.observed_down and not pressed:
                self.reserved_keys.discard(item.key)
                del self.commitments[token]
            elif now > item.predicted_s+.075:
                self.reserved_keys.discard(item.key)
                del self.commitments[token]
        return NKeyAction(tuple(values))


def _run(bpm: float, floors: int, calibration, *, partial: bool) -> dict:
    segment = baseline._fixture(bpm, floors=floors)
    planner = (PrefixPersistentMotorPlanner(calibration) if partial else
               baseline.PersistentMotorPlanner(calibration))
    env = DiagnosticHudNKeyRealChartMotorEnv(segment, key_count=p.KEY_COUNT,
        control_dt_s=baseline.DT, physics_dt_s=baseline.PHYSICS)
    observation = env.reset()
    statuses: Counter[str] = Counter()
    steps = int((segment.duration_s+.5)/baseline.DT)+3
    for _ in range(steps):
        hud = tuple(encode_n_key_hud_real_chart_observation(observation))
        if not partial:
            statuses[baseline.forecast_visible(hud).status] += 1
        action = planner.action(hud)
        result = env.step(action)
        observation = result.observation
        if result.done:
            break
    else:
        raise RuntimeError('Motor experiment did not finish in bounded steps')

    # Diagnostics below are evaluator-only; they never influence the commands.
    truth = tuple(target.episode_time_s for target in segment.targets)
    return {
        'bpm': bpm, 'targets': floors, 'variant': 'visible_prefix' if partial else 'full_horizon',
        'hits': env.stats.hits, 'misses': env.stats.misses,
        'x_accuracy_percent': env.stats.x_accuracy_percent,
        'overloaded': env.stats.overloaded,
        'planned_unique': planner.planned_unique,
        'late_at_discovery': planner.deadline_late_at_discovery,
        'forecast_frames': planner.forecast_frames,
        'abstain_frames': planner.abstain_frames,
        'visible_prefix_frames': planner.partial_frames if partial else 0,
        'visible_prefix_new_commitments': planner.partial_new_commitments if partial else 0,
        'forecast_statuses': dict(sorted((planner.statuses if partial else statuses).items())),
        'last_targets_hit': [bool(env._used[i]) for i in range(max(0,len(truth)-6),len(truth))],
        'motor_resets_after_start': 0,
    }


def compare_case(bpm: float, floors: int, calibration) -> dict:
    if (bpm, floors) not in FIXTURES:
        raise ValueError('Only fixed deterministic comparison fixtures are allowed')
    original = _run(bpm, floors, calibration, partial=False)
    revised = _run(bpm, floors, calibration, partial=True)
    return {
        'bpm': bpm, 'targets': floors,
        'original': original, 'visible_prefix': revised,
        'hit_delta': revised['hits']-original['hits'],
        'miss_delta': revised['misses']-original['misses'],
        'x_accuracy_delta': revised['x_accuracy_percent']-original['x_accuracy_percent'],
    }


def run_audit(config: p.PrestigeConfig) -> dict:
    if p.KEY_COUNT!=8 or abs(p.CONTROL_DT-.010)>1e-12 or abs(p.PHYSICS_DT-.001)>1e-12:
        raise RuntimeError('Frozen Stage T Motor contract changed')
    root=transfer.output_dir(config)
    path=root/REPORT
    if path.exists():
        raise FileExistsError('Refuse to overwrite earlier visible-prefix report')
    calibration=calibrate_n_key_press_lead(8,control_dt_s=.010,physics_dt_s=.001)
    cases=[compare_case(bpm,floors,calibration) for bpm,floors in FIXTURES]
    for r in cases:
        print(f"visible-prefix {r['bpm']:>6.0f} BPM {r['targets']:>3}: "
              f"baseline={r['original']['hits']}/{r['targets']} "
              f"prefix={r['visible_prefix']['hits']}/{r['targets']} "
              f"delta={r['hit_delta']:+d} "
              f"new-prefix-plans={r['visible_prefix']['visible_prefix_new_commitments']}",flush=True)
    report={
        'format': FORMAT, 'cases': cases,
        'prediction_horizon_ms': baseline.PREVIEW*1000.,
        'policy_control_dt_ms': 10., 'physical_dt_ms': 1.,
        'fixed_fixtures': True, 'no_training': True, 'no_checkpoint': True,
        'stage_a_fail_preserved': True, 'geometry_hybrid_untouched': True,
        'validation_evaluated': False, 'final_evaluated': False,
        'limitations': [
            'Partial forecasts assert only known safe events, never full horizon coverage',
            'Only ordinary floor geometry, fixed BPM and deterministic 18/72/144 note charts',
            'Original persistent motor scheduler unchanged; subclass replaces only forecaster',
            'One simulator reset per episode, 10ms action hold and 1ms physics',
            'Existing physical timing/misalignment problems remain out of scope',
            'Posthoc simulator Hit/Miss may credit a different target than planned',
            'No Student, checkpoints, Validation, FINAL or frozen geometry changes',
        ],
    }
    root.mkdir(parents=True,exist_ok=True)
    p._no_replace_json(path,report)
    return report
