from __future__ import annotations

"""Read-only, checkpoint-frozen Prestige I Stage C v2.1 diagnostics.

Selected Validation anchors are replayed strictly as *diagnostics*, never
training, selection, or FINAL evaluation.  Nothing writes to checkpoint files.
"""

from dataclasses import asdict
from pathlib import Path
import json
import math

import torch

from dmdod.envs.n_key import (
    DiagnosticHudNKeyRealChartMotorEnv,
    encode_n_key_hud_real_chart_observation,
)
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG
from dmdod.n_key_motor import NKeyAction
from . import prestige1 as p
from . import prestige1_transfer as transfer

FORMAT = "prestige1-stage-c-v21-readonly-diagnostics"
REPORT = "stage_c_v21_diagnostics.json"
DEFAULT_ANCHORS = (7, 11, 18)
TRAILING_FRAMES = 5
TRAILING_EVENTS_S = .150


def _load_provenance(config, *, device):
    root = transfer.output_dir(config)
    report_file = root / transfer.MOTOR_REPORT
    if not report_file.is_file():
        raise FileNotFoundError("Stage C v2 report missing")
    stage_c = json.loads(report_file.read_text(encoding="utf-8"))
    stage_a, _pretrained, stage_b = transfer._verify_stage_b(config, device=device)
    if (stage_c.get("format") != transfer.EXPERIMENT_VERSION
            or stage_c.get("stage") != "C"
            or stage_c.get("stage_a_gate_overridden") is not False
            or stage_c.get("stage_a_original_gate") != stage_a["gate"]
            or stage_c.get("pretrained_sha256") != stage_b["pretrained_sha256"]
            or stage_c.get("core_sha256") != stage_b["core_sha256"]
            or stage_c.get("validation_used_for_selection") is not False
            or stage_c.get("no_final") is not True
            or stage_c.get("effective_updates_per_arm", 0) < 1):
        raise RuntimeError("Stage C v2 provenance mismatch")
    results = stage_c.get("results", {})
    if set(results) != {"control", "prestige"}:
        raise RuntimeError("Both Control and Prestige Stage C results required")
    for arm in ("control", "prestige"):
        item = results[arm]
        rows = item.get("validation_per_anchor", [])
        if len(rows) != 20 or any(row.get("anchor") != i for i, row in enumerate(rows, 1)):
            raise RuntimeError("Stage C v2 Validation20 anchor record mismatch")
    return stage_a, stage_b, stage_c


def _load_model(config, *, arm, device, stage_b, stage_c):
    item = stage_c["results"][arm]
    expected = item.get("checkpoint_sha256")
    path = transfer.checkpoint_dir(config) / f"{arm}_v2.pt"
    if not path.is_file() or not expected or p.sha256(path) != expected:
        raise RuntimeError(f"Stage C v2 {arm} checkpoint hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("format") != transfer.EXPERIMENT_VERSION
            or payload.get("stage") != "C"
            or payload.get("arm") != arm
            or payload.get("config_sha256") != config.signature()
            or payload.get("pretrained_sha256") != stage_b["pretrained_sha256"]
            or payload.get("core_sha256") != stage_b["core_sha256"]
            or payload.get("warm_start_old_best") is not False
            or payload.get("seed") != config.seed
            or payload.get("updates") != stage_c["effective_updates_per_arm"]
            or payload.get("train_anchor_manifest") !=
               stage_c["_validated_train_anchor_manifest"]):
        raise RuntimeError(f"Stage C v2 {arm} checkpoint metadata mismatch")
    # Instantiate a fresh model and load ONLY this frozen model state.
    torch.manual_seed(config.seed)
    model = p.PrestigeMotorStudent(config.core_path, config.temporal_hidden).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.prepare_recurrent_runtime()
    model.eval()
    if any(x.requires_grad and x.grad is not None for x in model.parameters()):
        raise RuntimeError("Unexpected gradient in diagnostic model")
    return model, str(path), expected


def _manifest(config, stage_a):
    # Avoid p.load_role_sequences(): it would re-capture all Train20/Validation20.
    from dmdod.data.multichart import discover_multichart_dataset
    from dmdod.training.real_chart import compile_role, build_validation_segments
    dataset = discover_multichart_dataset(config.dataset)
    validation = build_validation_segments(
        compile_role(dataset.validation), window_s=config.validation_window_s)
    expected = stage_a["metadata"]["dataset_split"]["validation"]
    actual = [dict(anchor=i, name=segment.chart_name, sha256=segment.chart_sha256)
              for i, segment in enumerate(validation, 1)]
    if len(validation) != 20 or actual != expected:
        raise RuntimeError("Validation chart identities differ from Stage A v2")
    return validation


def _metrics_on_expert_sequence(model, sequence):
    # Expert sequence stays isolated from the closed-loop policy trajectory.
    from dmdod.training.prestige1 import _motor_timing_metrics
    return _motor_timing_metrics(model, [sequence])["per_anchor"][0]


def _closed_loop(model, named, *, device, anchor):
    env = DiagnosticHudNKeyRealChartMotorEnv(
        named.segment, key_count=p.KEY_COUNT, capture_overload_trace=True,
        control_dt_s=p.CONTROL_DT, physics_dt_s=p.PHYSICS_DT,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors)
    observation = env.reset()
    state = model.initial_state(device)
    predictions = []
    time_values = []
    probabilities = []
    chords = []
    truths = []
    masks = []
    max_steps = int((named.segment.duration_s + 2.0) / p.CONTROL_DT) + 200
    model.eval()
    with torch.inference_mode():
        for index in range(max_steps):
            now = float(env.privileged_episode_time_s())
            visible = encode_n_key_hud_real_chart_observation(observation)
            if len(visible) != p.INPUT_DIM:
                raise RuntimeError("Expected only 263D visible HUD input")
            x = torch.as_tensor(visible, dtype=torch.float32, device=device)
            means, _std, _value, state, raw = model.forward_step_with_timing(x, state)
            executed = torch.tanh(means)
            predicted_s, event_logit, chord_logits = p.decode_temporal(raw)
            probability = float(torch.sigmoid(event_logit))
            expected_floor = env._floor_index_at_chart_time(
                named.segment.chart_time_from_episode(now))
            truth, observable = p.target_labels(
                named.segment.targets, now, expected_floor,
                behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
                ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors)
            # Privileged truth is used only below, in diagnostics.
            command = NKeyAction(tuple(float(v) for v in executed.tolist()))
            step = env.step(command)
            predictions.append({
                "frame": index,
                "command_time_s": now,
                "action_8k": list(command.values),
                "time_until_event_predicted_ms": float(predicted_s) * 1000.,
                "event_within_200ms_probability": probability,
                "predicted_chord_size": int(chord_logits.argmax()) + 1,
                "evaluator_only": {
                    "observable": bool(observable),
                    "near_event": bool(truth[1]),
                    "time_until_event_truth_ms": truth[0] * 1000. if observable and truth[1] else None,
                },
            })
            time_values.append(float(predicted_s))
            probabilities.append(probability)
            chords.append(int(chord_logits.argmax()))
            truths.append(truth)
            masks.append(bool(observable))
            observation = step.observation
            if step.done:
                break
        else:
            raise RuntimeError("Closed-loop diagnostic exceeded step budget")
    trace = env.overload_trace
    if trace is None:
        raise RuntimeError("Diagnostic trace missing")
    stats = env.stats
    recorded = len(predictions)
    if len(trace.action_frames) != recorded:
        raise RuntimeError("Action/Timing trace length differs")
    for prediction, frame in zip(predictions, trace.action_frames):
        if (abs(prediction["command_time_s"] - frame.time_s) > 1e-6
                or any(abs(a-b) > 1e-6 for a,b in zip(
                    prediction["action_8k"], frame.action_values))):
            raise RuntimeError("Physical and Timing traces are not synchronized")
    metric = p._evaluation(
        torch.tensor(time_values), torch.tensor(probabilities),
        torch.tensor(truths), torch.tensor(masks, dtype=torch.bool),
        anchor=anchor, chord_pred=torch.tensor(chords))
    termination = trace.termination
    if bool(stats.overloaded) != (termination.reason == "Overload"):
        raise RuntimeError("Overload termination reason inconsistent with physics stats")
    unresolved = int(stats.targets) - int(stats.hits) - int(stats.misses)
    if unresolved < 0:
        raise RuntimeError("Hits + misses exceed chart targets")
    # Fixed 5 control frames, with evaluator physics state at frame end.
    last = []
    for prediction, physical in list(zip(predictions, trace.action_frames))[-TRAILING_FRAMES:]:
        last.append({
            **prediction,
            "physical_state_at_frame_end": {
                "positions_m": list(physical.positions_m),
                "velocities_m_s": list(physical.velocities_m_s),
                "pressed_8k": list(physical.pressed_flags),
            },
            "evaluator_only_next_target_index": physical.next_target_index,
            "evaluator_only_next_target_time_s": physical.next_target_time_s,
        })
    event_from = float(termination.time_s) - TRAILING_EVENTS_S
    early = [asdict(event) for event in trace.too_early_events
             if event.time_s >= event_from - 1e-8]
    down = [asdict(event) for event in trace.physical_keydowns
            if event.time_s >= event_from - 1e-8]
    if int(stats.too_early_presses) != len(trace.too_early_events):
        raise RuntimeError("Too Early count differs from physics event trace")
    if int(env.physical_keydowns) != len(trace.physical_keydowns):
        raise RuntimeError("KeyDown count differs from physics event trace")
    return {
        "anchor": anchor, "chart": named.chart_name,
        "stats": {
            "hits": int(stats.hits), "misses": int(stats.misses),
            "targets": int(stats.targets), "unjudged_targets": unresolved,
            "too_early": int(stats.too_early_presses),
            "overload": bool(stats.overloaded), "keydowns": int(env.physical_keydowns),
            "accuracy": float(stats.x_accuracy_percent),
        },
        "termination": asdict(termination),
        "termination_note": "Overload event time is the exact physical KeyDown time; frame motor state is at control-step end.",
        "last_5_control_frames": last,
        "keydowns_last_150ms": down,
        "too_early_last_150ms": early,
        "all_too_early": [asdict(event) for event in trace.too_early_events],
        "closed_loop_timing": metric,
        "executed_frames": recorded,
    }


def _verify_replay(record, stored, *, anchor, arm):
    counts = record["stats"]
    for key in ("hits", "misses", "too_early", "overload", "keydowns"):
        if counts[key] != stored[key]:
            raise RuntimeError(
                f"Read-only replay divergence {arm} anchor {anchor}: "
                f"{key} recorded={stored[key]} observed={counts[key]}")
    if record["chart"] != stored["chart"]:
        raise RuntimeError("Read-only replay chart name mismatch")


def run_readonly_diagnostics(config: p.PrestigeConfig, *,
                             anchors=DEFAULT_ANCHORS):
    anchors = tuple(anchors)
    if not anchors or len(set(anchors)) != len(anchors) or any(
            type(anchor) is not int or not 1 <= anchor <= 20 for anchor in anchors):
        raise ValueError("Specify distinct Validation anchors from 1 through 20")
    output = transfer.output_dir(config) / REPORT
    if output.exists():
        raise FileExistsError("Diagnostics already exist: " + str(output))
    old_best_before = p._protect_old_best(config)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    stage_a, stage_b, stage_c = _load_provenance(config, device=device)
    validation = _manifest(config, stage_a)
    stage_c["_validated_train_anchor_manifest"] = stage_a["metadata"]["dataset_split"]["train"]
    originals = {"stage_a": transfer.output_dir(config) / p.STAGE_A_REPORT,
                 "stage_b": transfer.output_dir(config) / transfer.TRANSFER_REPORT,
                 "stage_c": transfer.output_dir(config) / transfer.MOTOR_REPORT,
                 "stage_a_gru": transfer.output_dir(config) / "gru_v2.pt",
                 "core": Path(config.core_path)}
    for arm in ("control", "prestige"):
        originals[arm] = transfer.checkpoint_dir(config) / f"{arm}_v2.pt"
    originals_before = {name: p.sha256(path) for name,path in originals.items()}
    models = {}
    checkpoints = {}
    for arm in ("control", "prestige"):
        model, path, digest = _load_model(
            config, arm=arm, device=device, stage_b=stage_b, stage_c=stage_c)
        models[arm] = model
        checkpoints[arm] = {"path": path, "sha256": digest}
    results = {}
    for anchor in anchors:
        named = validation[anchor-1]
        # Expert HUD may be inspected, but it is NEVER used as a teacher or
        # source for any model updates in this script.
        expert = p._collect_one(
            named, role="validation", anchor=anchor, lead_s=config.lead_s,
            device=device)
        paired = {}
        for arm, model in models.items():
            loop = _closed_loop(model, named, device=device, anchor=anchor)
            stored = stage_c["results"][arm]["validation_per_anchor"][anchor-1]
            _verify_replay(loop, stored, anchor=anchor, arm=arm)
            loop["expert_hud_timing"] = _metrics_on_expert_sequence(model, expert)
            paired[arm] = loop
            print(f"Stage C v2.1 {arm} #{anchor:02d} "
                  f"hits={loop['stats']['hits']} "
                  f"unjudged={loop['stats']['unjudged_targets']} "
                  f"overload={loop['stats']['overload']} "
                  f"closed_loop_mae={loop['closed_loop_timing']['mae_ms']}ms",
                  flush=True)
        results[str(anchor)] = paired
    # Read-only claim is mechanically checked before writing the NEW report.
    originals_after = {name: p.sha256(path) for name,path in originals.items()}
    if originals_after != originals_before or p._protect_old_best(config) != old_best_before:
        raise RuntimeError("Read-only diagnostic modified a protected artifact")
    report = {
        "format": FORMAT, "stage": "C-v2.1", "read_only": True,
        "no_training": True, "no_final": True, "exploratory": True,
        "validation_used_for_selection": False,
        "stage_a_original_gate": stage_a["gate"],
        "stage_a_gate_overridden": False,
        "config_sha256": config.signature(),
        "checkpoint_sha256": {k:v["sha256"] for k,v in checkpoints.items()},
        "protected_artifact_sha256": originals_before,
        "original_artifacts_unchanged": True,
        "anchors": list(anchors),
        "control_dt_s": p.CONTROL_DT, "physics_dt_s": p.PHYSICS_DT,
        "temporal_comparison": "Expert HUD and on-policy HUD are DIFFERENT trajectories; time metrics have separate support and cannot be treated as paired frames.",
        "replay_matches_stage_c_v2": True,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    p._no_replace_json(output, report)
    return report
