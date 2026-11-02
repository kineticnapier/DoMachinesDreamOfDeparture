#!/usr/bin/env python3
"""D23: read-only analysis of D22 archived physical Motor states."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SCHEMA = "research-a-v2-l0-d23-archived-motor-state-v1"
PROTOCOL = "docs/research-a-v2-l0-d23-archived-motor-state-protocol.md"
D22_SCHEMA = "research-a-v2-l0-d22-single-step-intervention-v1"
SOURCE_RUN = 41515190995
SOURCE_ARTIFACT = 13175252296
INTERVENTION_STEP = 362
SNAPSHOT_STEPS = (362, 363, 365, 368, 369, 371, 372)
ACTUATION_M = 0.0020
RESET_M = 0.0018
SERIES = (
    "before", "after", "before_to_left", "after_to_none",
    "before_negative", "after_negative",
)
FIELDS = (
    ("position_m", "pre_motor", "left_position_m"),
    ("velocity_m_s", "pre_motor", "left_velocity_m_s"),
    ("pressed", "pre_motor", "left_pressed"),
    ("activation", "pre_diagnostics", "left_activation"),
    ("finger_fatigue", "pre_diagnostics", "left_fatigue"),
    ("hand_fatigue", "pre_diagnostics", "hand_fatigue"),
    ("hand_coordination", "pre_diagnostics", "hand_coordination"),
)
EXPECTED_COUNTS = {
    "before": (1, 1, 2),
    "after": (0, 3, 3),
    "before_to_left": (0, 3, 3),
    "after_to_none": (1, 2, 3),
    "before_negative": (1, 1, 2),
    "after_negative": (0, 3, 3),
}
EXPECTED_FIRST = {
    "before": (3.718, True),
    "after": (3.689, False),
    "before_to_left": (3.689, False),
    "after_to_none": (3.718, True),
    "before_negative": (3.718, True),
    "after_negative": (3.689, False),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value, name):
    require(type(value) in (int, float) and math.isfinite(value),
            "nonfinite/non-numeric " + name)
    return float(value)


def validate_sources(result, traces):
    require(result["schema"] == D22_SCHEMA and
            result["classification"] == "valid_single_step_intervention" and
            result["supports_local_command_contribution"] is True,
            "D22 result classification mismatch")
    require(result["source_d21_run"] == 41505006857 and
            result["source_d19_run"] == 41485667328 and
            result["new_motor_episodes"] == 6 and
            result["new_physical_keydowns"] == 16 and
            result["intervention_step"] == INTERVENTION_STEP and
            result["arm"] == "a0" and
            result["case_index"] == 4 and
            result["repetition"] == 3 and
            result["sampling_seed"] == 320403,
            "D22 provenance mismatch")
    require(set(traces) == set(SERIES) and
            set(result["metrics"]) == set(SERIES),
            "D22 trace keys mismatch")
    require(traces["before"] == traces["before_negative"] and
            traces["after"] == traces["after_negative"],
            "D22 negative control mismatch")
    total_keydowns = 0
    for name in SERIES:
        trace = traces[name]
        row, steps, events = (trace["row"], trace["steps"],
                              trace["physical_keydowns"])
        counts = (row["hits"], row["extra_keydowns"],
                  row["physical_keydowns"])
        require(counts == EXPECTED_COUNTS[name], name + ": event count mismatch")
        require(len(steps) == row["steps"] == 480 and
                len(events) == len(row["physical_trace"]) == counts[2] and
                sum(row["controls"]) == sum(row["sampled_controls"]) == 480,
                name + ": incomplete 480-step trace")
        require((result["metrics"][name]["hits"],
                 result["metrics"][name]["extra_keydowns"],
                 result["metrics"][name]["physical_keydowns"]) == counts and
                result["metrics"][name]["controls"] == row["controls"] and
                result["metrics"][name]["sampled_controls"] == row["sampled_controls"],
                name + ": D22 result/trace mismatch")
        total_keydowns += counts[2]
        for index, step in enumerate(steps):
            require(step["step"] == index and
                    abs(finite_number(step["time_s"], "time") - index / 100) < 1e-7 and
                    abs(finite_number(step["end_time_s"], "end time") -
                        (index + 1) / 100) < 1e-7,
                    name + ": invalid/noncontinuous step clock")
            require(len(step["observation"]) == 8 and
                    all(math.isfinite(float(x)) for x in step["observation"]),
                    name + ": invalid observation")
            require(len(step["action_probabilities"]) == 4 and
                    all(math.isfinite(float(x)) for x in step["action_probabilities"]),
                    name + ": invalid action probabilities")
            require(step["action"] in range(4) and
                    step["sampled_action"] in range(4),
                    name + ": invalid action")
            for field_name, container, key in FIELDS:
                value = step[container][key]
                if field_name == "pressed":
                    require(type(value) is bool, name + ": invalid pressed state")
                else:
                    finite_number(value, name + "/" + field_name)
            require(abs(finite_number(step["pre_diagnostics"]["time_s"],
                                      "diagnostic clock") -
                        step["time_s"]) < 1e-7,
                    name + ": diagnostic clock mismatch")
        for index, event in enumerate(events):
            require(event["event_index"] == index and
                    event["event"] == row["physical_trace"][index],
                    name + ": physical event mapping mismatch")
            source = event["source_step"]
            require(0 <= source < len(steps) and
                    index in steps[source]["emitted_keydown_indices"],
                    name + ": missing event step")
            t = finite_number(event["event"]["time_s"], "keydown time")
            require(steps[source]["time_s"] - 1e-8 <= t <=
                    steps[source]["end_time_s"] + 1e-8,
                    name + ": KeyDown outside source step")
        first = events[0]["event"]
        require(abs(first["time_s"] - EXPECTED_FIRST[name][0]) < 1e-8 and
                first["hit"] is EXPECTED_FIRST[name][1] and
                first["key"] == "left",
                name + ": D22 first KeyDown mismatch")
        d22_first = result["metrics"][name]["first_physical"]
        require(abs(d22_first["time_s"] - first["time_s"]) < 1e-8 and
                d22_first["hit"] is first["hit"],
                name + ": D22 first event result mismatch")
        changed = [s["step"] for s in steps if s["intervention_applied"]]
        expected_changed = [INTERVENTION_STEP] if name in (
            "before_to_left", "after_to_none") else []
        require(changed == expected_changed and
                all(s["action"] == s["sampled_action"] for s in steps
                    if not s["intervention_applied"]),
                name + ": unexpected intervention")
    require(total_keydowns == result["new_physical_keydowns"] == 16,
            "total physical KeyDown mismatch")
    for baseline, changed, original, override in (
        ("before", "before_to_left", 0, 2),
        ("after", "after_to_none", 2, 0),
    ):
        b = traces[baseline]["steps"]
        c = traces[changed]["steps"]
        require(b[:INTERVENTION_STEP] == c[:INTERVENTION_STEP] and
                b[INTERVENTION_STEP]["observation"] ==
                c[INTERVENTION_STEP]["observation"] and
                b[INTERVENTION_STEP]["pre_motor"] ==
                c[INTERVENTION_STEP]["pre_motor"] and
                b[INTERVENTION_STEP]["pre_diagnostics"] ==
                c[INTERVENTION_STEP]["pre_diagnostics"] and
                b[INTERVENTION_STEP]["action"] ==
                c[INTERVENTION_STEP]["sampled_action"] == original and
                c[INTERVENTION_STEP]["action"] == override,
                baseline + ": intervention provenance mismatch")
    require(traces["before"]["steps"][INTERVENTION_STEP]["observation"] ==
            traces["after"]["steps"][INTERVENTION_STEP]["observation"],
            "D22 intervention pre-observation differs")


def snapshot(step):
    motor = step["pre_motor"]
    diag = step["pre_diagnostics"]
    pos = motor["left_position_m"]
    return {
        "step": step["step"],
        "time_s": step["time_s"],
        "sampled_action": step["sampled_action"],
        "executed_action": step["action"],
        "left_position_m": pos,
        "left_velocity_m_s": motor["left_velocity_m_s"],
        "left_pressed": motor["left_pressed"],
        "left_activation": diag["left_activation"],
        "left_fatigue": diag["left_fatigue"],
        "hand_fatigue": diag["hand_fatigue"],
        "hand_coordination": diag["hand_coordination"],
        "actuation_gap_m": ACTUATION_M - pos,
        "reset_gap_m": RESET_M - pos,
    }


def first_field_difference(baseline, changed):
    found = {}
    for field_name, container, key in FIELDS:
        first = next((i for i, (b, c) in enumerate(zip(baseline, changed))
                      if b[container][key] != c[container][key]), None)
        found[field_name] = (
            {"step": first, "time_s": baseline[first]["time_s"],
             "baseline": baseline[first][container][key],
             "intervention": changed[first][container][key]}
            if first is not None else None
        )
    return found


def analyze(result, traces):
    validate_sources(result, traces)
    snapshots = {
        name: {str(i): snapshot(traces[name]["steps"][i])
               for i in SNAPSHOT_STEPS}
        for name in SERIES
    }
    pairs = {}
    for baseline, changed, label in (
        ("before", "before_to_left", "before_to_left"),
        ("after", "after_to_none", "after_to_none"),
    ):
        b, c = traces[baseline], traces[changed]
        bs, cs = b["steps"], c["steps"]
        b_event, c_event = (b["physical_keydowns"][0],
                            c["physical_keydowns"][0])
        pairs[label] = {
            "first_state_differences": first_field_difference(bs, cs),
            "first_keydown": {
                "baseline_time_s": b_event["event"]["time_s"],
                "intervention_time_s": c_event["event"]["time_s"],
                "delta_ms": 1000.0 * (
                    c_event["event"]["time_s"] - b_event["event"]["time_s"]),
                "baseline_source_step": b_event["source_step"],
                "intervention_source_step": c_event["source_step"],
                "baseline_pre_event_step": snapshot(bs[b_event["source_step"]]),
                "intervention_pre_event_step": snapshot(cs[c_event["source_step"]]),
                "baseline_next_step": snapshot(bs[b_event["source_step"] + 1]),
                "intervention_next_step": snapshot(cs[c_event["source_step"] + 1]),
            },
            "hit_delta": c["row"]["hits"] - b["row"]["hits"],
            "extra_delta": (c["row"]["extra_keydowns"] -
                            b["row"]["extra_keydowns"]),
            "physical_keydown_delta": (c["row"]["physical_keydowns"] -
                                       b["row"]["physical_keydowns"]),
        }
        require(abs(pairs[label]["first_keydown"]["delta_ms"] -
                    result["first_keydown_shift_ms"][
                        "before_to_left_minus_before" if baseline == "before"
                        else "after_to_none_minus_after"]) < 1e-8,
                label + ": D22 KeyDown shift mismatch")
    early = (snapshots["before_to_left"]["368"]["actuation_gap_m"] <
             snapshots["before"]["368"]["actuation_gap_m"])
    reverse = (snapshots["after_to_none"]["368"]["actuation_gap_m"] >
               snapshots["after"]["368"]["actuation_gap_m"])
    return {
        "schema": SCHEMA, "protocol": PROTOCOL,
        "classification": "valid_archived_motor_state_trace",
        "source_run": SOURCE_RUN, "source_artifact": SOURCE_ARTIFACT,
        "intervention_step": INTERVENTION_STEP,
        "actuation_threshold_m": ACTUATION_M,
        "reset_threshold_m": RESET_M,
        "snapshot_steps": list(SNAPSHOT_STEPS),
        "snapshots": snapshots, "pairs": pairs,
        "negative_controls_exact": True,
        "early_trace_closer_to_threshold_at_step368": early,
        "reverse_trace_farther_from_threshold_at_step368": reverse,
        "archived_counts": {
            name: {"hits": traces[name]["row"]["hits"],
                   "extra_keydowns": traces[name]["row"]["extra_keydowns"],
                   "physical_keydowns": traces[name]["row"]["physical_keydowns"]}
            for name in SERIES
        },
        "new_training_episodes": 0, "new_ppo_updates": 0,
        "new_adam_steps": 0, "new_motor_episodes": 0,
        "new_physical_keydowns": 0,
        "a1_evaluated": False, "a3_evaluated": False,
        "new_intervention_performed": False,
        "teacher_free_fingering_acquired": False,
        "human_equivalence_verified": False,
        "single_finger_physical_impossibility_verified": False,
        "formal_study_002_approved": False,
        "limitation": ("10ms pre-step state only; 1ms internal path is not archived. "
                       "No state-mediation or generalized learning claim."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d22-result", required=True, type=Path)
    parser.add_argument("--d22-traces", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite D23 output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        source = json.loads(args.d22_result.read_text(encoding="utf-8"))
        traces = json.loads(args.d22_traces.read_text(encoding="utf-8"))
        output = analyze(source, traces)
    except (ValueError, TypeError, KeyError, IndexError, OSError,
            json.JSONDecodeError) as exc:
        output = {"schema": SCHEMA, "protocol": PROTOCOL,
                  "classification": "invalid_archived_motor_state_trace",
                  "error": str(exc), "new_training_episodes": 0,
                  "new_ppo_updates": 0, "new_motor_episodes": 0,
                  "new_physical_keydowns": 0}
        args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print("D23_FAILURE", json.dumps(output, ensure_ascii=False), flush=True)
        raise SystemExit(1)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    brief = {
        "classification": output["classification"],
        "first_differences": {name: pair["first_state_differences"]
                              for name, pair in output["pairs"].items()},
        "snapshots": {name: {k: val for k, val in items.items()
                             if k in ("362", "363", "368", "369", "371")}
                      for name, items in output["snapshots"].items()
                      if name in ("before", "after", "before_to_left",
                                  "after_to_none")},
        "first_keydown": {name: pair["first_keydown"]["delta_ms"]
                          for name, pair in output["pairs"].items()},
        "closer_to_threshold": output[
            "early_trace_closer_to_threshold_at_step368"],
        "farther_from_threshold": output[
            "reverse_trace_farther_from_threshold_at_step368"],
    }
    print("D23_RESULT", json.dumps(brief, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
