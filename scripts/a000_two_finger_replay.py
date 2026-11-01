#!/usr/bin/env python3
"""Research A / Experiment 000: deterministic two-finger Motor replay.

This is a chart-free and teacher-free infrastructure check.  It does not train
a policy or establish that alternation has emerged.

Run from the repository root:
    python scripts/a000_two_finger_replay.py --seed 123 --actions 320

Only policy-independent fixed actions and physical Motor observations are used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import random

from dmdod.motor.env import MotorAction, MotorEnv


def deterministic_actions(seed: int, count: int) -> tuple[MotorAction, ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    # Deterministic controls, NOT an expert/teacher or fingering recommendation.
    # Mix pressing, releasing, and off periods to exercise physical state.
    choices = ((1., 0.), (-1., 0.), (0., 1.), (0., -1.),
               (1., 1.), (-1., -1.), (0., 0.), (1., -1.), (-1., 1.))
    controls = []
    while len(controls) < count:
        left, right = choices[rng.randrange(len(choices))]
        controls.extend([MotorAction(left, right)] * rng.randrange(2, 12))
    return tuple(controls[:count])


def replay(*, seed: int = 123, count: int = 320, same_hand: bool = True) -> dict:
    env = MotorEnv(same_hand=same_hand, control_dt_s=0.010)
    initial = env.reset()
    actions = deterministic_actions(seed, count)
    output = {
        "initial_observation": asdict(initial),
        "motor_configuration": asdict(env.sim.body.config),
        "control_dt_s": env.control_dt_s,
        "physics_substeps": env.physics_substeps,
        "actions": [asdict(action) for action in actions],
        "transitions": [],
    }
    downs = 0
    ups = 0
    for action in actions:
        tr = env.step(action)
        events = [
            {"time_s": e.time_s, "key": e.key, "event": e.event.value}
            for e in tr.evaluator_events
        ]
        downs += sum(e["event"] == "down" for e in events)
        ups += sum(e["event"] == "up" for e in events)
        output["transitions"].append({
            "observation": asdict(tr.observation),
            "events": events,
            "diagnostics": asdict(tr.diagnostics),
        })
    serialized = json.dumps(
        output, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    action_data = json.dumps(
        output["actions"], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema": "research-a-experiment-000-replay-v1",
        "seed": seed,
        "actions": len(actions),
        "same_hand": same_hand,
        "control_dt_s": env.control_dt_s,
        "physics_dt_s": env.sim.config.dt_s,
        "keydowns": downs,
        "keyups": ups,
        "action_sha256": hashlib.sha256(action_data).hexdigest(),
        "trace_sha256": hashlib.sha256(serialized).hexdigest(),
        "trace": output,
    }


def summary(replay_result: dict) -> dict:
    return {k: v for k, v in replay_result.items() if k != "trace"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--actions", type=int, default=320)
    parser.add_argument("--different-hands", action="store_true")
    args = parser.parse_args()
    result = replay(
        seed=args.seed, count=args.actions,
        same_hand=not args.different_hands,
    )
    print(json.dumps(summary(result), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
