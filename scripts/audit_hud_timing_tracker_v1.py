"""HUD-only stateful Timing tracker, scored against oracle *after* inference.

Offline Stage T research. Never touches Student training, Stage A gate, or
Motor. Output creation is exclusive: no prior diagnostic is overwritten.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
import json
from pathlib import Path

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.n_key import DiagnosticHudNKeyRealChartMotorEnv, encode_n_key_hud_real_chart_observation
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG as FLOORS
from dmdod.motor.n_key import NKeyAction
from dmdod.planning.hud_timing_tracker_v1 import CausalTimingTracker


def audit(chart: Path, *, end_s: float) -> dict:
    if not end_s > 0:
        raise ValueError("end_s must be positive")
    raw = chart.read_bytes()
    segment = build_playable_segment(
        compile_adofai(parse_adofai_text(raw.decode("utf-8-sig"))), end_s=end_s)
    env = DiagnosticHudNKeyRealChartMotorEnv(
        segment, key_count=8, control_dt_s=.010, physics_dt_s=.001,
        behind_floors=FLOORS.behind_floors, ahead_floors=FLOORS.ahead_floors, tail_s=.04)
    obs = env.reset()
    idle = NKeyAction((0.,)*8)
    hud_frames = []
    for _ in range(int((end_s+.5)/.010)+10):
        hud_frames.append(tuple(encode_n_key_hud_real_chart_observation(obs)))
        transition = env.step(idle)
        obs = transition.observation
        if transition.done:
            break
    else:
        raise RuntimeError("HUD capture exceeded the bounded episode")
    tracker = CausalTimingTracker()
    results = [tracker.observe(i, hud) for i, hud in enumerate(hud_frames)]
    # Oracle target list is accessed ONLY AFTER all HUD predictions.
    truth = tuple(float(t.episode_time_s) for t in segment.targets)
    all_elapsed = [event for update in results for event in update.elapsed]
    # The end_s-clipped chart target list excludes real future floors.
    # Do not mark an out-of-scope prediction as a false positive.
    elapsed = [event for event in all_elapsed if event.due_s <= end_s + 1e-8]
    out_of_scope = len(all_elapsed) - len(elapsed)
    exact = []
    unmatched = []
    for event in elapsed:
        k = bisect_left(truth, event.due_s)
        candidates = truth[max(0,k-1):min(len(truth),k+1)]
        err = min((abs(t-event.due_s) for t in candidates), default=float("inf"))
        if err <= .00005:
            exact.append(event)
        else:
            unmatched.append({"id":event.local_id,"due_s":event.due_s,
                              "closest_error_ms":err*1000})
    return {
        "format":"dmdod-hud-timing-tracker-v1-conditional-audit",
        "no_oracle_predictor_inputs": True,
        "no_training": True,
        "horizon_ms":30,
        "control_ms":10,
        "physics_ms":1,
        "frames":len(results),
        "true_events":len(truth),
        "statuses":dict(Counter(r.status for r in results)),
        "eligible_frames":sum(r.status=="eligible" for r in results),
        "new_ids":sum(len(r.new_ids) for r in results),
        "reused_ids":sum(len(r.reused_ids) for r in results),
        "elapsed_events":len(elapsed),
        "out_of_scope_events":out_of_scope,
        "correct_elapsed_events":len(exact),
        "unmatched_elapsed_events":len(unmatched),
        "examples":unmatched[:10],
        "limitations":"Elapsed means scheduled-time passed, not a physical hit. "
                      "Only visible ordinary geometry; partial coverage; no Student/Stage A adoption.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chart",type=Path)
    ap.add_argument("--end-s",type=float,default=65.)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():
        ap.error("Refusing to overwrite previous diagnostic")
    result=audit(args.chart,end_s=args.end_s)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
