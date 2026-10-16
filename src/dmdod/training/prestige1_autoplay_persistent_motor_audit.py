from __future__ import annotations

"""Minimal Stage T persistent 10ms-action / 1ms-physics motor experiment.

A provisional CPU-only baseline, not a trained Student or an oracle teacher.
Actions come only from visible HUD, prior controller state, and a fixed physical
cold-finger calibration. Evaluator target times are read posthoc only.
"""

from dataclasses import dataclass
from math import atan2, isfinite, pi
from pathlib import Path
import json

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.n_key import DiagnosticHudNKeyRealChartMotorEnv, encode_n_key_hud_real_chart_observation
from dmdod.motor.capacity import calibrate_n_key_press_lead, NKeyLeadCalibration
from dmdod.motor.n_key import NKeyAction, n_key_names

from . import prestige1 as p
from . import prestige1_autoplay_countdown as cd
from . import prestige1_autoplay_future_events_audit as future
from . import prestige1_transfer as transfer

FORMAT = "prestige1-stage-t-persistent-10ms-8key-readonly-v1"
REPORT = "stage_timing_persistent_motor_minimal_readonly.json"
DT = .010
PHYSICS = .001
PREVIEW = .070
FIXTURE_BPM = (600., 900., 1200., 2000., 4000., 20000.)


def _forecast_one(hud: tuple[float, ...], sign: int, bpm: float, horizon: float) -> future.Forecast:
    """Generalization of the frozen 10ms geometry baseline, with explicit horizon."""
    velocity = pi * bpm / 60.
    orbit = atan2(hud[cd.MOTOR_FEATURES], hud[cd.MOTOR_FEATURES + 1])
    events: list[float] = []
    offsets: list[int] = []
    elapsed = 0.
    for slot in range(cd.CURRENT_SLOT, cd.CURRENT_SLOT + cd.FLOORS.ahead_floors + 1):
        floor = cd._visible_slot(hud, slot)
        if floor[-1] < .5:
            return future.Forecast((), (), "insufficient_lookahead")
        offset = slot - cd.CURRENT_SLOT
        if floor[6] > .5 or floor[7] > .5:
            return future.Forecast((), (), "midspin_or_multiplanet")
        markers = [j for j in range(cd.SKIP_MARKER_FROM, cd.SKIP_MARKER_TO) if floor[j] > .5]
        if any(j != cd.SKIP_MARKER_FROM for j in markers) or (offset > 0 and markers):
            return future.Forecast((), (), "special_event")
        entry = atan2(floor[2], floor[3])
        end = atan2(floor[4], floor[5])
        span = (sign * (end - entry)) % (2 * pi)
        if span < future.RAD_EPS or 2 * pi - span < future.RAD_EPS:
            return future.Forecast((), (), "full_turn_or_zero_arc")
        if offset == 0:
            progress = (sign * (orbit - entry)) % (2 * pi)
            if progress < future.RAD_EPS or 2 * pi - progress < future.RAD_EPS:
                progress = 0.
            if progress > span + future.RAD_EPS:
                return future.Forecast((), (), "incompatible_direction")
            if span - progress < future.RAD_EPS:
                return future.Forecast((), (), "boundary_ambiguous")
            remaining = max(0., span - progress)
        else:
            remaining = span
        until = remaining / velocity
        if elapsed + until > horizon + future.TIME_EPS_S:
            return future.Forecast(tuple(offsets), tuple(events), "eligible")
        following = cd._visible_slot(hud, slot + 1) if slot + 1 < cd.FLOORS.floor_slots else ()
        if not following or following[-1] < .5:
            return future.Forecast((), (), "insufficient_lookahead")
        if any(following[j] > .5 for j in range(6, cd.SKIP_MARKER_TO)):
            return future.Forecast((), (), "special_future_floor")
        elapsed += until
        if elapsed <= future.TIME_EPS_S:
            return future.Forecast((), (), "boundary_ambiguous")
        events.append(elapsed)
        offsets.append(offset + 1)
        if elapsed >= horizon - future.TIME_EPS_S:
            return future.Forecast(tuple(offsets), tuple(events), "eligible")
    return future.Forecast((), (), "insufficient_lookahead")


def forecast_visible(hud: tuple[float, ...], horizon: float = PREVIEW) -> future.Forecast:
    """Single-frame causal HUD forecast, rejecting direction ambiguity."""
    if not isfinite(horizon) or horizon <= 0.:
        raise ValueError("Invalid horizon")
    if len(hud) != p.INPUT_DIM or not all(isfinite(x) for x in hud):
        return future.Forecast((), (), "invalid_hud")
    try:
        bpm = 120. * (2. ** hud[cd.HUD_OFFSET])
    except OverflowError:
        return future.Forecast((), (), "invalid_bpm")
    if not isfinite(bpm) or bpm <= 0.:
        return future.Forecast((), (), "invalid_bpm")
    if max(abs(v) for v in hud[cd.MOTOR_FEATURES:cd.MOTOR_FEATURES + 2]) < .5:
        return future.Forecast((), (), "invalid_orbit")
    a, b = (_forecast_one(hud, s, bpm, horizon) for s in (1, -1))
    viable = [x for x in (a, b) if x.eligible]
    if len(viable) == 1:
        other = b if a.eligible else a
        if other.status == "incompatible_direction":
            return viable[0]
        return future.Forecast((), (), other.status)
    if len(viable) == 2:
        if (a.floor_offsets == b.floor_offsets and len(a.times_s) == len(b.times_s)
                and all(abs(x-y) < future.TIME_EPS_S for x, y in zip(a.times_s, b.times_s))):
            return a
        return future.Forecast((), (), "ambiguous_direction")
    for reason in ("special_future_floor", "special_event", "midspin_or_multiplanet",
                   "boundary_ambiguous", "full_turn_or_zero_arc", "insufficient_lookahead"):
        if reason in (a.status, b.status):
            return future.Forecast((), (), reason)
    return future.Forecast((), (), "no_consistent_direction")


@dataclass
class Commitment:
    predicted_s: float
    key: str
    commanded: bool = False
    observed_down: bool = False


class PersistentMotorPlanner:
    """One physical Motor / one 10ms command each tick; no oracle inputs."""

    def __init__(self, calibration: NKeyLeadCalibration, *, preview: float = PREVIEW):
        if calibration.key_count != p.KEY_COUNT:
            raise ValueError("Incorrect motor key count")
        self.names = n_key_names(p.KEY_COUNT)
        self.latency = dict(calibration.key_latencies_s)
        self.preview = preview
        self.commitments: dict[int, Commitment] = {}
        self.reserved_keys: set[str] = set()
        self.next_key = 0
        self.frame = 0
        self.abstain_frames = 0
        self.forecast_frames = 0
        self.planned_unique = 0
        self.deadline_late_at_discovery = 0

    @staticmethod
    def token(seconds: float) -> int:
        # All chart-schedule timestamps remain relative to the policy's local
        # monotonic control tick. 100us token resolution, no evaluator clock.
        return round(seconds * 10000)

    def action(self, hud: tuple[float, ...]) -> NKeyAction:
        now = self.frame * DT
        self.frame += 1
        estimate = forecast_visible(hud, self.preview)
        if estimate.eligible:
            self.forecast_frames += 1
            for delay in estimate.times_s:
                due = now + delay
                token = self.token(due)
                if token in self.commitments or due <= now + 1e-7:
                    continue
                # One distinct finger per active commitment. Persist ownership
                # across successive re-observations of the same target.
                chosen = None
                for offset in range(len(self.names)):
                    key = self.names[(self.next_key + offset) % len(self.names)]
                    i = self.names.index(key)
                    if key not in self.reserved_keys and hud[2*p.KEY_COUNT+i] < .5:
                        chosen = key
                        self.next_key = (i + 1) % len(self.names)
                        break
                if chosen is None:
                    continue
                self.commitments[token] = Commitment(due, chosen)
                self.reserved_keys.add(chosen)
                self.planned_unique += 1
                if due - now < self.latency[chosen] - 1e-7:
                    self.deadline_late_at_discovery += 1
        else:
            self.abstain_frames += 1

        values = [0.] * len(self.names)
        for token, item in tuple(sorted(self.commitments.items())):
            index = self.names.index(item.key)
            pressed = hud[2*p.KEY_COUNT+index] > .5
            if pressed:
                item.observed_down = True
            if item.observed_down:
                values[index] = -1.
            elif now + self.latency[item.key] >= item.predicted_s - 1e-7:
                values[index] = 1.
                item.commanded = True
            if item.observed_down and not pressed:
                self.reserved_keys.discard(item.key)
                del self.commitments[token]
            elif now > item.predicted_s + .075:
                self.reserved_keys.discard(item.key)
                del self.commitments[token]
        # All physical commands persist exactly the full 10ms next step.
        return NKeyAction(tuple(values))


def _fixture(bpm: float, *, floors: int = 18):
    if bpm not in FIXTURE_BPM:
        raise ValueError("Fixed diagnostic fixtures only")
    content = json.dumps({"angleData": [0.] * floors,
                          "settings": {"bpm": bpm, "pitch": 100., "countdownTicks": 0},
                          "actions": []})
    compiled = compile_adofai(parse_adofai_text(content))
    return build_playable_segment(compiled)


def simulate(bpm: float) -> dict:
    segment = _fixture(bpm)
    calibration = calibrate_n_key_press_lead(p.KEY_COUNT, control_dt_s=DT, physics_dt_s=PHYSICS)
    planner = PersistentMotorPlanner(calibration)
    env = DiagnosticHudNKeyRealChartMotorEnv(segment, key_count=p.KEY_COUNT,
        control_dt_s=DT, physics_dt_s=PHYSICS)
    obs = env.reset()
    max_steps = int((segment.duration_s + .5)/DT)+3
    actions = []
    physical_events = []
    for _ in range(max_steps):
        action = planner.action(tuple(encode_n_key_hud_real_chart_observation(obs)))
        actions.append(action.as_tuple())
        result = env.step(action)
        # Diagnostic evaluator-side: never passed to planner.
        for evt in env.motor.last_transition.evaluator_events:
            if evt.event.value == "down":
                physical_events.append((float(evt.time_s), evt.key))
        obs = result.observation
        if result.done:
            break
    else:
        raise RuntimeError("Persistent motor experiment exceeded bounded horizon")
    true_times = tuple(t.episode_time_s for t in segment.targets)
    # Evaluator-only nearest distinct target accounting; comparison is NOT
    # used to adjust any plans or Motor actions.
    unmatched = list(true_times)
    signed_errors = []
    for time_s, key in physical_events:
        if not unmatched:
            break
        closest = min(range(len(unmatched)), key=lambda i: abs(unmatched[i]-time_s))
        signed_errors.append((time_s-unmatched.pop(closest))*1000.)
    return {
        "bpm": bpm,
        "control_frames": len(actions),
        "targets": len(true_times),
        "hud_forecast_frames": planner.forecast_frames,
        "hud_abstain_frames": planner.abstain_frames,
        "unique_commitments": planner.planned_unique,
        "late_at_discovery": planner.deadline_late_at_discovery,
        "physical_keydowns": len(physical_events),
        "actual_keydown_times_ms": [round(t*1000., 6) for t, _ in physical_events],
        "signed_nearest_target_errors_ms": signed_errors,
        "evaluator_hits": env.stats.hits,
        "evaluator_misses": env.stats.misses,
        "evaluator_overloaded": env.stats.overloaded,
        "continuous_10ms_policy_1ms_physics": True,
        "motor_resets_after_start": 0,
    }


def run_audit(config: p.PrestigeConfig) -> dict:
    if abs(p.CONTROL_DT-DT)>1e-12 or abs(p.PHYSICS_DT-PHYSICS)>1e-12:
        raise RuntimeError("Frozen clock contract changed")
    output = transfer.output_dir(config) / REPORT
    if output.exists():
        raise FileExistsError("Refuse overwrite of persistent Motor diagnostic")
    cases = [simulate(bpm) for bpm in FIXTURE_BPM]
    for item in cases:
        print(f"persistent {item['bpm']:>7.0f} BPM: "
              f"planned={item['unique_commitments']} "
              f"downs={item['physical_keydowns']} "
              f"hits={item['evaluator_hits']} "
              f"misses={item['evaluator_misses']} "
              f"abstentions={item['hud_abstain_frames']}", flush=True)
    report = {
        "format": FORMAT,
        "cases": cases,
        "no_training": True,
        "no_checkpoint_loaded": True,
        "original_stage_a_fail_preserved": True,
        "validation_evaluated": False,
        "final_evaluated": False,
        "original_geometry_hybrid_untouched": True,
        "notes": [
            "Minimal deterministic straight-floor fixtures; this is not a general scheduler",
            "263D HUD-only causal lookahead and local policy-clock deadlines",
            "Repeated 10ms held Motor actions and 1ms physical substeps; no Motor reset between notes",
            "No privileged timestamp or floor index reaches the planner",
            "Initial physical fingers at rest; no learned anticipation before the first HUD",
            "No demonstration of end-to-end 20,000 BPM playability",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    p._no_replace_json(output, report)
    return report
