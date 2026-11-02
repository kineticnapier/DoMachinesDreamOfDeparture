#!/usr/bin/env python3
"""D21: trace a single archived D19 case from policy action to physical KeyDown.

Evaluation-only diagnostics; no changes to Motor, reward, PPO, or policy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

import torch

import a002v2_l0_low_speed_ppo as l0
import a002v2_l0_d12_frozen_ppo_cpu as d12
import a002v2_l0_d13_frozen_gradient_stage as d13
import a002v2_l0_d15_first_ppo_clipping as d15
import a002v2_l0_d18_frozen_policy_probability as d18
import a002v2_l0_d20_first_keydown_divergence as d20

SCHEMA = "research-a-v2-l0-d21-single-case-command-latency-v1"
PROTOCOL = "docs/research-a-v2-l0-d21-command-latency-protocol.md"
CASE = 4
REPETITION = 3
SEED = 320403
INTERVAL_MS = 240
START_S = 1.075
MAX_STEPS = 800


def require(ok, message):
    if not ok:
        raise ValueError(message)


def trace_one(model):
    """Use the original sampling and physical step; observe without intervening."""
    env = l0.LoggedLowSpeedEnv(arm="a0", interval_ms=INTERVAL_MS,
                               start_s=START_S)
    observation = env.reset()
    rng = torch.Generator(device="cpu")
    rng.manual_seed(SEED)
    model.eval()
    controls = [0] * 4
    steps = []
    physical = []
    entropy_total = 0.0
    left_command_held = False
    last_left_onset_s = None
    with torch.no_grad():
        for index in range(MAX_STEPS):
            start_time = float(env.motor.sim.time_s)
            require(math.isfinite(start_time), "nonfinite control time")
            require(len(observation) == 8 and
                    all(math.isfinite(float(x)) for x in observation),
                    "invalid observation")
            logits, _ = model(torch.tensor(observation, dtype=torch.float32))
            probs = torch.softmax(logits, dim=-1)
            require(probs.shape == (4,) and bool(torch.isfinite(probs).all()),
                    "invalid policy probabilities")
            entropy_total += float(
                -(probs * torch.log(probs.clamp_min(1e-12))).sum().item()
            )
            action = int(torch.multinomial(probs, 1, generator=rng).item())
            controls[action] += 1
            left_held = action in (2, 3)
            if left_held and not left_command_held:
                last_left_onset_s = start_time
            left_command_held = left_held
            pre_state = asdict(env.motor.observe())
            pre_diagnostics = asdict(env.motor.diagnostics())
            before_events = len(env.down_trace)
            next_observation, reward, finished = env.step(action)
            emitted = env.down_trace[before_events:]
            new_events = []
            for event in emitted:
                require(event["key"] in ("left", "right"),
                        "unexpected physical key")
                onset = last_left_onset_s if event["key"] == "left" else None
                latency = (1000.0 * (event["time_s"] - onset)
                           if onset is not None else None)
                require(latency is None or latency >= -1e-6,
                        "physical KeyDown precedes command onset")
                record = {
                    "event_index": len(physical),
                    "source_step": index,
                    "source_step_time_s": start_time,
                    "pre_observation": list(observation),
                    "selected_action": action,
                    "last_left_command_onset_s": onset,
                    "command_to_keydown_ms": latency,
                    "event": dict(event),
                }
                physical.append(record)
                new_events.append(len(physical) - 1)
            steps.append({
                "step": index,
                "time_s": start_time,
                "observation": list(observation),
                "action_probabilities": [float(p) for p in probs.tolist()],
                "action": action,
                "left_command_held": left_held,
                "left_command_onset_s": (start_time if left_held and
                                         (index == 0 or not steps[-1]["left_command_held"])
                                         else None),
                "pre_motor": pre_state,
                "pre_diagnostics": pre_diagnostics,
                "reward": float(reward),
                "emitted_keydown_indices": new_events,
                "end_time_s": float(env.motor.sim.time_s),
            })
            observation = next_observation
            if finished:
                break
        else:
            raise ValueError("episode exceeded maximum control steps")
    result = env.result()
    row = {
        "case_index": CASE, "repetition": REPETITION,
        "arm": "a0", "interval_ms": INTERVAL_MS, "start_s": START_S,
        "mode": "sampled", "sampling_seed": SEED,
        **asdict(result),
        "physical_left": sum(e["key"] == "left" for e in env.down_trace),
        "physical_right": sum(e["key"] == "right" for e in env.down_trace),
        "controls": controls, "steps": len(steps),
        "mean_action_entropy": entropy_total / len(steps),
        "physical_trace": list(env.down_trace),
        "matched_keys": "".join("L" if x == "left" else "R"
                                for x in result.matched_keys),
    }
    require(len(physical) == result.physical_keydowns and
            len(physical) == len(row["physical_trace"]),
            "physical event count mismatch")
    require(all(p["event"] == row["physical_trace"][i]
                for i, p in enumerate(physical)), "physical event mapping mismatch")
    require(sum(controls) == len(steps), "action accounting mismatch")
    return {"row": row, "steps": steps, "physical_keydowns": physical}


def archived_pair(data):
    # D20 validates all 45 archived pairs and the known D19 aggregate.
    d20.analyze(data)
    require(data["schema"] == d20.D19_SCHEMA and
            data["classification"] == "valid_paired_motor_evaluation",
            "D19 source invalid")
    pair = {}
    for side in ("before", "after"):
        rows = data["sets"]["sampled"][side]["rows"]
        matches = [r for r in rows if r["case_index"] == CASE and
                   r["repetition"] == REPETITION]
        require(len(matches) == 1, "D19 case/repetition missing")
        row = matches[0]
        require(row["sampling_seed"] == SEED and
                row["interval_ms"] == INTERVAL_MS and
                row["start_s"] == START_S and row["targets"] == 16,
                "D19 target condition mismatch")
        pair[side] = row
    require((pair["before"]["hits"], pair["before"]["extra_keydowns"],
             pair["before"]["physical_keydowns"]) == (1, 1, 2),
            "D19 before known counts mismatch")
    require((pair["after"]["hits"], pair["after"]["extra_keydowns"],
             pair["after"]["physical_keydowns"]) == (0, 3, 3),
            "D19 after known counts mismatch")
    require(pair["before"]["physical_trace"][0]["time_s"] == 3.718 and
            pair["after"]["physical_trace"][0]["time_s"] == 3.689,
            "D19 known first physical event mismatch")
    return pair


def first_step_difference(before, after, field):
    require(len(before) == len(after), "control step counts differ")
    for b, a in zip(before, after):
        if b[field] != a[field]:
            return {"step": b["step"], "time_s": b["time_s"],
                    "before": {"action": b["action"],
                               "observation": b["observation"],
                               "probabilities": b["action_probabilities"]},
                    "after": {"action": a["action"],
                              "observation": a["observation"],
                              "probabilities": a["action_probabilities"]}}
    return None


def compare_traces(before, after):
    bs, az = before["steps"], after["steps"]
    first_action = first_step_difference(bs, az, "action")
    first_observation = first_step_difference(bs, az, "observation")
    first_physical = d20.first_difference(
        before["row"]["physical_trace"], after["row"]["physical_trace"])
    b_events = before["physical_keydowns"]
    a_events = after["physical_keydowns"]
    first_index = first_physical["event_index"] if first_physical else None
    mapped = ({"before": b_events[first_index] if first_index < len(b_events) else None,
               "after": a_events[first_index] if first_index < len(a_events) else None}
              if first_index is not None else None)
    return {
        "first_action_difference": first_action,
        "first_observation_difference": first_observation,
        "first_physical_difference": first_physical,
        "first_physical_command_mapping": mapped,
        "first_action_before_first_physical":
            (first_action["time_s"] <= first_physical["first_divergence_time_s"]
             if first_action and first_physical else None),
        "metrics": {
            side: {key: trace["row"][key] for key in
                   ("hits", "extra_keydowns", "physical_keydowns", "steps")}
            for side, trace in (("before", before), ("after", after))
        },
        "hit_delta": after["row"]["hits"] - before["row"]["hits"],
        "extra_delta": after["row"]["extra_keydowns"] - before["row"]["extra_keydowns"],
        "keydown_delta": (after["row"]["physical_keydowns"] -
                          before["row"]["physical_keydowns"]),
    }


def run(d8_dir, d12_dir, d19_json):
    d12.require_dispatch("native")
    require(os.environ.get("ATEN_CPU_CAPABILITY") is None,
            "native CPU path required")
    torch.set_num_threads(1)
    require(torch.__version__ == "2.5.1+cpu" and
            torch.get_num_threads() == 1, "pinned CPU runtime required")
    data = json.loads(Path(d19_json).read_text(encoding="utf-8"))
    source_rows = archived_pair(data)
    source = d15.load_source(d12_dir)
    capture, native, default, comparison = d13.validate_archived(source)
    model, initial_sha = d12.weights_digest(
        Path(d8_dir) / "native-initial-state.pt",
        Path(d8_dir) / "native.json", Path(d8_dir) / "default.json")
    require(initial_sha == capture["initial_state_sha256"],
            "D8 initial weights SHA mismatch")
    batch = torch.load(Path(d12_dir) / "frozen-batch.pt",
                       map_location="cpu", weights_only=True)
    require(d12.batch_digest(batch) == capture["batch_sha256"],
            "D12 frozen batch SHA mismatch")
    require(d13.compare_parameters(native["weights_after"],
                                   default["weights_after"])["different_elements"]
            == comparison["weights"]["different_elements"] == 10,
            "D12 updated weights mismatch")
    before_bits = d18.weights_bits(model)
    updated = d18.restore_updated(model, native["weights_after"])
    after_bits = d18.weights_bits(updated)
    before = trace_one(model)
    negative = trace_one(model)
    require(before == negative, "negative control full-step replay mismatch")
    after = trace_one(updated)
    for side, replay in (("before", before), ("after", after)):
        for key in ("case_index", "repetition", "arm", "interval_ms",
                    "start_s", "mode", "sampling_seed", "targets",
                    "hits", "extra_keydowns", "physical_keydowns",
                    "physical_left", "physical_right", "controls", "steps",
                    "physical_trace", "matched_keys"):
            require(replay["row"][key] == source_rows[side][key],
                    "D19 archived row mismatch: " + side + "/" + key)
    require(d18.weights_bits(model) == before_bits and
            d18.weights_bits(updated) == after_bits,
            "policy mutated during evaluation")
    comparison_result = compare_traces(before, after)
    return {
        "schema": SCHEMA, "protocol": PROTOCOL,
        "classification": "valid_single_case_command_latency",
        "source_d19_run": d20.SOURCE_RUN,
        "source_d19_artifact": d20.SOURCE_ARTIFACT,
        "source_d19_commit": "08d8de523ffae10513645dfe95566060b4e6bf0d",
        "initial_state_sha256": initial_sha,
        "batch_sha256": capture["batch_sha256"],
        "runtime": d12.runtime(),
        "arm": "a0", "training_seed": 0, "case_index": CASE,
        "repetition": REPETITION, "sampling_seed": SEED,
        "interval_ms": INTERVAL_MS, "start_s": START_S,
        "negative_control_exact": True,
        "comparison": comparison_result,
        "new_training_episodes": 0, "new_ppo_updates": 0,
        "new_adam_steps": 0, "new_motor_episodes": 3,
        "new_physical_keydowns": (before["row"]["physical_keydowns"] * 2 +
                                  after["row"]["physical_keydowns"]),
        "a1_evaluated": False, "a3_evaluated": False,
        "intervention_performed": False,
        "teacher_free_fingering_acquired": False,
        "human_equivalence_verified": False,
        "single_finger_physical_impossibility_verified": False,
        "formal_study_002_approved": False,
        "limitation": ("Association is a control-step timing observation, not "
                       "proof of which command caused a later physical KeyDown."),
    }, {"before": before, "after": after, "negative_control": negative}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d8-dir", type=Path, required=True)
    parser.add_argument("--d12-dir", type=Path, required=True)
    parser.add_argument("--d19-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trace-out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.trace_out.exists():
        raise FileExistsError("refusing to overwrite D21 output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        result, traces = run(args.d8_dir, args.d12_dir, args.d19_json)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError,
            json.JSONDecodeError) as exc:
        failure = {"schema": SCHEMA, "protocol": PROTOCOL,
                   "classification": "invalid_single_case_command_latency",
                   "error": str(exc), "new_training_episodes": 0,
                   "new_ppo_updates": 0, "new_adam_steps": 0}
        args.out.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False), flush=True)
        raise SystemExit(1)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    args.trace_out.write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print("D21_RESULT", json.dumps({
        "classification": result["classification"],
        "comparison": result["comparison"],
        "new_physical_keydowns": result["new_physical_keydowns"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
