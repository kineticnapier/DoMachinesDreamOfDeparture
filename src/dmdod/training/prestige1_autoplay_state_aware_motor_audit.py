from __future__ import annotations

"""Read-only Stage T experiment: own-action shadow Motor for predictive key arming.

The physical shadow is initialized from rest and advanced solely with the
controller's own previous actions. It receives no evaluator or hidden live
Motor state. The current 263D HUD and persistent plans are the only external
inputs. The original persistent/prefix planners and Motor are unchanged.
"""

from copy import deepcopy
from math import ceil

from dmdod.envs.n_key import (
    DiagnosticHudNKeyRealChartMotorEnv,
    encode_n_key_hud_real_chart_observation,
)
from dmdod.motor.capacity import calibrate_n_key_press_lead
from dmdod.motor.keyboard import KeyEvent
from dmdod.motor.n_key import NKeyAction, NKeyMotorEnv

from . import prestige1 as p
from . import prestige1_autoplay_persistent_motor_audit as baseline
from . import prestige1_autoplay_visible_prefix_audit as prefix
from . import prestige1_transfer as transfer

FORMAT = 'prestige1-stage-t-shadow-dynamics-motor-arm-readonly-v1'
REPORT = 'stage_timing_shadow_dynamics_motor_arming_readonly.json'
FIXTURES = ((1200., 18), (2000., 18), (4000., 18),
            (1200., 72), (2000., 72), (4000., 72),
            (1200., 144), (2000., 144), (4000., 144))
PREDICTION_HORIZON_S = .090
# This is a fixed sample-quantization margin (half of one frozen 10ms command),
# not a tuned search parameter.
COMMAND_QUANTIZATION_MARGIN_S = baseline.DT / 2


class ShadowDynamicsPlanner(prefix.PrefixPersistentMotorPlanner):
    """Predict KeyDown by replaying the physical Motor from own-action history.

    No simulator env/motor reference accepted; shadow initialized independently.
    """

    def __init__(self, calibration):
        super().__init__(calibration)
        self.shadow = NKeyMotorEnv(p.KEY_COUNT, control_dt_s=baseline.DT,
                                   physics_dt_s=baseline.PHYSICS)
        self.shadow.reset()
        self.shadow_disagreements = 0
        self.rollout_count = 0
        self.arm_decisions = 0
        self.wait_decisions = 0
        self.no_predicted_down = 0
        self.predicted_keydowns: list[dict] = []

    def _validate_hud(self, hud: tuple[float,...]) -> None:
        obs = self.shadow.observe()
        for a,b in zip(obs.positions_m, hud[:p.KEY_COUNT]):
            if abs(a/0.006-b)>1e-5:
                self.shadow_disagreements+=1
                raise RuntimeError('Independent shadow Motor position diverged from visible HUD')
        for a,b in zip(obs.velocities_m_s, hud[p.KEY_COUNT:2*p.KEY_COUNT]):
            if abs(a-b)>1e-5:
                self.shadow_disagreements+=1
                raise RuntimeError('Independent shadow Motor velocity diverged from visible HUD')
        for k,v in enumerate(obs.pressed_flags):
            if bool(v) != bool(hud[2*p.KEY_COUNT+k]>.5):
                self.shadow_disagreements+=1
                raise RuntimeError('Independent shadow Motor pressed flag diverged from visible HUD')

    def _down_if_started_now(self, index: int, other_actions: tuple[float,...]) -> float | None:
        """10ms-held +1 key command through exact coupled physical simulation.

        Keep other keys' candidate commands fixed. Return seconds until *first*
        KeyDown, or None if not observed within a fixed bounded 90ms horizon.
        """
        trial = deepcopy(self.shadow)
        values = list(other_actions)
        values[index] = 1.
        for _ in range(ceil(PREDICTION_HORIZON_S/baseline.DT)):
            transition = trial.step(NKeyAction(tuple(values)))
            self.rollout_count += 1
            for event in transition.evaluator_events:
                if event.event is KeyEvent.DOWN and event.key==self.names[index]:
                    return event.time_s - self.shadow.time_s
        return None

    def action(self, hud:tuple[float,...]) -> NKeyAction:
        self._validate_hud(hud)
        now = self.frame*baseline.DT
        self.frame+=1
        forecast = prefix.forecast_visible_prefix(hud,self.preview)
        self.statuses[forecast.status]+=1
        if forecast.eligible:
            self.forecast_frames+=1
            if forecast.status=='visible_prefix':
                self.partial_frames+=1
            for delay in forecast.times_s:
                due=now+delay
                token=self.token(due)
                if token in self.commitments or due<=now+1e-7:
                    continue
                chosen=None
                for offset in range(len(self.names)):
                    key=self.names[(self.next_key+offset)%len(self.names)]
                    index=self.names.index(key)
                    if key not in self.reserved_keys and hud[2*p.KEY_COUNT+index]<.5:
                        chosen=key
                        self.next_key=(index+1)%len(self.names)
                        break
                if chosen is None:
                    continue
                self.commitments[token]=baseline.Commitment(due,chosen)
                self.reserved_keys.add(chosen)
                self.planned_unique+=1
                if forecast.status=='visible_prefix':
                    self.partial_new_commitments+=1
                if due-now<self.latency[chosen]-1e-7:
                    self.deadline_late_at_discovery+=1
        else:
            self.abstain_frames+=1

        values=[0.]*len(self.names)
        for token,item in tuple(sorted(self.commitments.items())):
            index=self.names.index(item.key)
            pressed=hud[2*p.KEY_COUNT+index]>.5
            if pressed:
                item.observed_down=True
            if item.observed_down:
                values[index]=-1.
            elif item.commanded:
                values[index]=1.
            else:
                estimate=self._down_if_started_now(index,tuple(values))
                if estimate is None:
                    self.no_predicted_down+=1
                    values[index]=1.
                elif now+estimate>=item.predicted_s-COMMAND_QUANTIZATION_MARGIN_S:
                    values[index]=1.
                    self.arm_decisions+=1
                    self.predicted_keydowns.append({
                        'intended_ms':item.predicted_s*1000,
                        'estimated_lead_ms':estimate*1000,
                        'command_start_ms':now*1000,
                        'key':item.key,
                    })
                else:
                    self.wait_decisions+=1
                if values[index]>0:
                    item.commanded=True
            if item.observed_down and not pressed:
                self.reserved_keys.discard(item.key)
                del self.commitments[token]
            elif now>item.predicted_s+.075:
                self.reserved_keys.discard(item.key)
                del self.commitments[token]
        action=NKeyAction(tuple(values))
        self.shadow.step(action)
        return action


def simulate_case(bpm:float, notes:int, calibration, *, shadow:bool) -> dict:
    if (bpm,notes) not in FIXTURES:
        raise ValueError('Unregistered fixture')
    segment=baseline._fixture(bpm,floors=notes)
    planner=(ShadowDynamicsPlanner(calibration) if shadow else
             prefix.PrefixPersistentMotorPlanner(calibration))
    env=DiagnosticHudNKeyRealChartMotorEnv(segment,key_count=p.KEY_COUNT,
                                           control_dt_s=.010,physics_dt_s=.001)
    obs=env.reset()
    armed_trajectories: list[dict] = []
    physical_keydowns_ms: list[dict] = []
    for tick in range(int((segment.duration_s+.5)/.01)+3):
        hud=tuple(encode_n_key_hud_real_chart_observation(obs))
        action=planner.action(hud)
        if shadow:
            # Read the planner's own logged intent after it has chosen action.
            for predicted in planner.predicted_keydowns[len(armed_trajectories):]:
                armed_trajectories.append({**predicted, 'actual_ms':None})
        result=env.step(action)
        for evt in env.motor.last_transition.evaluator_events:
            if evt.event is KeyEvent.DOWN:
                physical_keydowns_ms.append({'key':evt.key,'ms':evt.time_s*1000.})
                if shadow:
                    candidates=[r for r in armed_trajectories if r['key']==evt.key
                                and r['actual_ms'] is None and
                                r['command_start_ms']<=evt.time_s*1000.+1e-6]
                    if candidates:
                        candidates[0]['actual_ms']=evt.time_s*1000.
        obs=result.observation
        if result.done:
            break
    else:
        raise RuntimeError('Episode time bound exceeded')
    predicted_errors=[r['actual_ms']-(r['command_start_ms']+r['estimated_lead_ms'])
                      for r in armed_trajectories if r['actual_ms'] is not None]
    intended_errors=[r['actual_ms']-r['intended_ms']
                     for r in armed_trajectories if r['actual_ms'] is not None]
    return dict(bpm=bpm,notes=notes,variant='shadow_trajectory' if shadow else 'visible_prefix_baseline',
                hits=env.stats.hits, misses=env.stats.misses,
                x_accuracy=env.stats.x_accuracy_percent,
                overloaded=env.stats.overloaded,
                physical_keydowns=env.physical_keydowns,
                unique_plans=planner.planned_unique,
                shadow_rollout_count=planner.rollout_count if shadow else 0,
                shadow_position_disagreements=planner.shadow_disagreements if shadow else None,
                arms=planner.arm_decisions if shadow else 0,
                waits=planner.wait_decisions if shadow else 0,
                no_predicted_down=planner.no_predicted_down if shadow else 0,
                matched_armed_keydowns=len(predicted_errors),
                predicted_vs_actual_keydown_mae_ms=(sum(abs(e) for e in predicted_errors)/len(predicted_errors)
                    if predicted_errors else None),
                intended_vs_actual_keydown_mae_ms=(sum(abs(e) for e in intended_errors)/len(intended_errors)
                    if intended_errors else None),
                keydown_count=len(physical_keydowns_ms),
                first_prediction_errors_ms=[round(e,3) for e in predicted_errors[:12]],
                planned_arming_examples=armed_trajectories[:12])


def compare_case(bpm:float, notes:int, calibration)->dict:
    if (bpm,notes) not in FIXTURES:
        raise ValueError('Only fixed deterministic chart conditions are permitted')
    a=simulate_case(bpm,notes,calibration,shadow=False)
    b=simulate_case(bpm,notes,calibration,shadow=True)
    return {'bpm':bpm,'notes':notes,'baseline':a,'shadow':b,
            'hit_delta':b['hits']-a['hits'],
            'x_accuracy_delta':b['x_accuracy']-a['x_accuracy']}


def run_audit(config:p.PrestigeConfig)->dict:
    if p.KEY_COUNT!=8 or p.CONTROL_DT!=.010 or p.PHYSICS_DT!=.001:
        raise RuntimeError('Frozen 8-Key model clock changed')
    output=transfer.output_dir(config)/REPORT
    if output.exists():
        raise FileExistsError('Refuse to overwrite previous shadow dynamics report')
    calibration=calibrate_n_key_press_lead(8,control_dt_s=.010,physics_dt_s=.001)
    results=[compare_case(bpm,n,calibration) for bpm,n in FIXTURES]
    for r in results:
        b,s=r['baseline'],r['shadow']
        print(f"shadow-trajectory {r['bpm']:>6.0f} BPM {r['notes']:>3}: "
              f"Hit {b['hits']}->{s['hits']}, X {b['x_accuracy']:.2f}->{s['x_accuracy']:.2f}, "
              f"arm-model MAE {s['predicted_vs_actual_keydown_mae_ms']}",flush=True)
    report={
        'format':FORMAT,'cases':results,
        'no_training':True,'no_student_checkpoint':True,
        'stage_a_fail_preserved':True,'male_cns_4096_untouched':True,
        'original_geometry_hybrid_untouched':True,
        'validation_evaluated':False,'final_evaluated':False,
        'motor_clock_ms':10.,'physical_clock_ms':1.,
        'model_based_prediction_horizon_ms':PREDICTION_HORIZON_S*1000.,
        'command_quantization_margin_ms':COMMAND_QUANTIZATION_MARGIN_S*1000.,
        'fixture_key_count':8,
        'limitations':[
            'A new isolated planner only: existing persistent/visible-prefix controllers unchanged',
            'Shadow physical state inferred solely from own prior actions; HUD verifies visible position/velocity',
            'Motor dynamics are known exactly in this simulator; no unknown real-world-model robustness evaluated',
            'Candidate one-finger predicted onset assumes other keys hold provisional commands',
            'Single-finger predictive arming is not joint multi-key trajectory optimization',
            'Current 8-key fixed Student HUD 263D; user ordinarily uses 16-key fingering near 4000 BPM',
            'No inference from evaluator target timestamps or motor hidden live state',
            'A/B evaluation fixed before testing; regressions retained in report',
            'No Motor training, Validation, FINAL, or checkpoint mutation',
        ],
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    p._no_replace_json(output,report)
    return report
