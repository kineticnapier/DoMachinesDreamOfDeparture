#!/usr/bin/env python3
"""D22: single-step causal intervention on A0 left command at 3.620s."""
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
import a002v2_l0_d21_command_latency as d21

SCHEMA = "research-a-v2-l0-d22-single-step-intervention-v1"
PROTOCOL = "docs/research-a-v2-l0-d22-single-step-intervention-protocol.md"
INTERVENTION_STEP = 362
CASE, REPETITION, SEED = d21.CASE, d21.REPETITION, d21.SEED
INTERVAL_MS, START_S, MAX_STEPS = d21.INTERVAL_MS, d21.START_S, d21.MAX_STEPS
TARGET11_S = 3.715
require = d21.require

def trace_one(model, *, intervention_action=None, expected_sampled_action=None):
    """Draw the original action exactly once, then optionally override step362."""
    env = l0.LoggedLowSpeedEnv(arm="a0", interval_ms=INTERVAL_MS,
                               start_s=START_S)
    observation = env.reset()
    rng = torch.Generator(device="cpu")
    rng.manual_seed(SEED)
    model.eval()
    controls = [0] * 4
    sampled_controls = [0] * 4
    intervention_count = 0
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
            sampled_action = int(torch.multinomial(probs, 1, generator=rng).item())
            sampled_controls[sampled_action] += 1
            action = sampled_action
            overridden = index == INTERVENTION_STEP and intervention_action is not None
            if overridden:
                require(expected_sampled_action is None or
                        sampled_action == expected_sampled_action,
                        "unexpected sampled action at intervention step")
                require(intervention_action in (0, 2) and
                        intervention_action != sampled_action,
                        "invalid/no-op intervention")
                action = intervention_action
                intervention_count += 1
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
                "sampled_action": sampled_action,
                "intervention_applied": overridden,
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
        "controls": controls, "sampled_controls": sampled_controls,
        "steps": len(steps),
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
    require(sum(controls) == sum(sampled_controls) == len(steps),
            "action accounting mismatch")
    require(intervention_count == (1 if intervention_action is not None else 0),
            "intervention step missing or repeated")
    return {"row": row, "steps": steps, "physical_keydowns": physical}



def strip_extra(trace):
    """Recover exact D21 structure from the D22 diagnostic extension."""
    return {
        "row": {k: v for k, v in trace["row"].items()
                if k != "sampled_controls"},
        "steps": [{k: v for k, v in s.items()
                   if k not in ("sampled_action", "intervention_applied")}
                  for s in trace["steps"]],
        "physical_keydowns": trace["physical_keydowns"],
    }


def summarize(trace):
    row = trace["row"]
    events = trace["physical_keydowns"]
    first = events[0] if events else None
    return {
        "hits": row["hits"],
        "extra_keydowns": row["extra_keydowns"],
        "physical_keydowns": row["physical_keydowns"],
        "controls": row["controls"],
        "sampled_controls": row["sampled_controls"],
        "steps": row["steps"],
        "first_physical": (
            {"time_s": first["event"]["time_s"],
             "key": first["event"]["key"],
             "hit": first["event"]["hit"],
             "offset_from_target11_ms":
                 1000.0 * (first["event"]["time_s"] - TARGET11_S),
             "command_to_keydown_ms": first["command_to_keydown_ms"]}
            if first is not None else None),
        "intervention_count": sum(
            bool(s["intervention_applied"]) for s in trace["steps"]),
    }


def check_physical(trace):
    row = trace["row"]
    require(row["hits"] + row["extra_keydowns"] ==
            row["physical_keydowns"] == len(trace["physical_keydowns"]),
            "physical KeyDown accounting mismatch")
    require(row["physical_right"] == 0, "A0 produced right KeyDown")
    require(sum(row["controls"]) == sum(row["sampled_controls"]) ==
            row["steps"] == len(trace["steps"]), "action count mismatch")
    for i, step in enumerate(trace["steps"]):
        require(step["step"] == i and
                math.isfinite(step["time_s"]) and
                math.isfinite(step["end_time_s"]) and
                step["time_s"] < step["end_time_s"], "invalid step clock")
        if i:
            require(abs(step["time_s"] -
                        trace["steps"][i-1]["end_time_s"]) < 1e-8,
                    "noncontinuous control clock")
        require(all(math.isfinite(float(v)) for v in step["observation"]) and
                len(step["observation"]) == 8, "invalid observation")
        require(len(step["action_probabilities"]) == 4 and
                all(math.isfinite(v) for v in step["action_probabilities"]),
                "invalid probabilities")
    for i, item in enumerate(trace["physical_keydowns"]):
        require(item["event_index"] == i and
                item["event"] == row["physical_trace"][i],
                "physical trace mapping mismatch")
        step = trace["steps"][item["source_step"]]
        require(i in step["emitted_keydown_indices"] and
                step["time_s"] <= item["event"]["time_s"] <=
                step["end_time_s"] + 1e-8,
                "physical event outside source step")


def run(d8_dir, d12_dir, d19_json, d21_trace_json):
    d12.require_dispatch("native")
    require(os.environ.get("ATEN_CPU_CAPABILITY") is None,
            "native CPU path required")
    torch.set_num_threads(1)
    require(torch.__version__ == "2.5.1+cpu" and
            torch.get_num_threads() == 1, "pinned CPU runtime required")
    data = json.loads(Path(d19_json).read_text(encoding="utf-8"))
    archived_rows = d21.archived_pair(data)
    d21_traces = json.loads(Path(d21_trace_json).read_text(encoding="utf-8"))
    require(set(d21_traces) == {"before", "after", "negative_control"},
            "invalid D21 archived trace keys")
    require(d21_traces["before"] == d21_traces["negative_control"],
            "archived D21 negative control mismatch")

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
            "D12 weights mismatch")
    before_bits = d18.weights_bits(model)
    updated = d18.restore_updated(model, native["weights_after"])
    after_bits = d18.weights_bits(updated)

    before = trace_one(model)
    after = trace_one(updated)
    before_negative = trace_one(model)
    after_negative = trace_one(updated)
    require(before == before_negative and after == after_negative,
            "new negative control full-step mismatch")
    for side, replay in (("before", before), ("after", after)):
        check_physical(replay)
        require(strip_extra(replay) == d21_traces[side],
                "D21 full-step archived mismatch: " + side)
        for key, value in archived_rows[side].items():
            require(replay["row"].get(key) == value,
                    "D19 archived row mismatch: " + side + "/" + key)
    b_step, a_step = before["steps"][INTERVENTION_STEP], after["steps"][INTERVENTION_STEP]
    require(b_step["step"] == a_step["step"] == INTERVENTION_STEP and
            b_step["time_s"] == a_step["time_s"] == 3.62,
            "intervention time mismatch")
    require(b_step["observation"] == a_step["observation"] and
            b_step["action"] == 0 and a_step["action"] == 2,
            "D21 step362 baseline mismatch")

    before_left = trace_one(model, intervention_action=2,
                            expected_sampled_action=0)
    after_none = trace_one(updated, intervention_action=0,
                           expected_sampled_action=2)
    for side, baseline, intervention, applied in (
        ("before_to_left", before, before_left, 2),
        ("after_to_none", after, after_none, 0)):
        check_physical(intervention)
        changed = [s["step"] for s in intervention["steps"]
                   if s["intervention_applied"]]
        require(changed == [INTERVENTION_STEP], "intervention not exactly one step")
        require(intervention["steps"][INTERVENTION_STEP]["observation"] ==
                baseline["steps"][INTERVENTION_STEP]["observation"],
                "intervention altered pre-step observation")
        require(intervention["steps"][INTERVENTION_STEP]["sampled_action"] ==
                baseline["steps"][INTERVENTION_STEP]["sampled_action"] and
                intervention["steps"][INTERVENTION_STEP]["action"] == applied,
                "intervention action mismatch")
        for i in range(INTERVENTION_STEP):
            require(intervention["steps"][i] == baseline["steps"][i],
                    "pre-intervention divergence at step " + str(i))
        require(all(s["action"] == s["sampled_action"] for i, s in
                    enumerate(intervention["steps"]) if i != INTERVENTION_STEP),
                "unregistered action override")
    require(d18.weights_bits(model) == before_bits and
            d18.weights_bits(updated) == after_bits,
            "policy weights mutated")
    traces = {
        "before": before, "after": after,
        "before_negative": before_negative, "after_negative": after_negative,
        "before_to_left": before_left, "after_to_none": after_none,
    }
    metrics = {key: summarize(value) for key, value in traces.items()}
    first = {key: metrics[key]["first_physical"] for key in traces}
    support = (first["before_to_left"] is not None and
               first["before"] is not None and
               first["after_to_none"] is not None and
               first["after"] is not None and
               first["before_to_left"]["time_s"] < first["before"]["time_s"] and
               first["after_to_none"]["time_s"] > first["after"]["time_s"])
    result = {
        "schema": SCHEMA, "protocol": PROTOCOL,
        "classification": "valid_single_step_intervention",
        "source_d21_run": 41505006857,
        "source_d19_run": 41485667328,
        "initial_state_sha256": initial_sha,
        "batch_sha256": capture["batch_sha256"],
        "runtime": d12.runtime(),
        "arm": "a0", "case_index": CASE, "repetition": REPETITION,
        "sampling_seed": SEED, "intervention_step": INTERVENTION_STEP,
        "intervention_time_s": 3.62,
        "baseline_observation_equal": True,
        "negative_controls_exact": True,
        "metrics": metrics,
        "supports_local_command_contribution": bool(support),
        "first_keydown_shift_ms": {
            "before_to_left_minus_before": (
                1000 * (first["before_to_left"]["time_s"] -
                        first["before"]["time_s"])
                if first["before_to_left"] and first["before"] else None),
            "after_to_none_minus_after": (
                1000 * (first["after_to_none"]["time_s"] -
                        first["after"]["time_s"])
                if first["after_to_none"] and first["after"] else None),
        },
        "new_training_episodes": 0, "new_ppo_updates": 0,
        "new_adam_steps": 0, "new_motor_episodes": 6,
        "new_physical_keydowns": sum(
            row["physical_keydowns"] for row in metrics.values()),
        "a1_evaluated": False, "a3_evaluated": False,
        "intervention_performed": True,
        "teacher_free_fingering_acquired": False,
        "human_equivalence_verified": False,
        "single_finger_physical_impossibility_verified": False,
        "formal_study_002_approved": False,
        "limitation": ("Single-step intervention includes all downstream physical "
                       "and policy feedback; not a unique direct latency attribution."),
    }
    return result, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d8-dir", type=Path, required=True)
    parser.add_argument("--d12-dir", type=Path, required=True)
    parser.add_argument("--d19-json", type=Path, required=True)
    parser.add_argument("--d21-trace-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trace-out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.trace_out.exists():
        raise FileExistsError("refusing to overwrite D22 output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        result, traces = run(args.d8_dir, args.d12_dir,
                             args.d19_json, args.d21_trace_json)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError,
            json.JSONDecodeError) as exc:
        failure = {"schema": SCHEMA, "protocol": PROTOCOL,
                   "classification": "invalid_single_step_intervention",
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
    print("D22_RESULT", json.dumps({
        "classification": result["classification"],
        "supports_local_command_contribution":
            result["supports_local_command_contribution"],
        "metrics": result["metrics"],
        "first_keydown_shift_ms": result["first_keydown_shift_ms"],
        "new_physical_keydowns": result["new_physical_keydowns"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
