from __future__ import annotations

"""Prestige I Stage B/C v2: pretrained temporal transfer and matched motor BC.

This is a NEW exploratory comparison. It never changes the failed Stage A MAE
gate, and never uses Validation or FINAL to pick a candidate or hyperparameter.
The sole Student input is the causal 263D HUD; no privileged timing truth is
present in the inference path.
"""

from pathlib import Path
import json
import math
import random

import torch

from . import prestige1 as p


EXPERIMENT_VERSION = "prestige1-bc-pretrained-transfer-v2"
TRANSFER_REPORT = "stage_b_v2.json"
MOTOR_REPORT = "stage_c_v2.json"
PARITY_ATOL_CPU = 2e-5
PARITY_ATOL_CUDA = 2e-4
GRU_WEIGHTS = {
    "weight_ih_l0": "weight_ih",
    "weight_hh_l0": "weight_hh",
    "bias_ih_l0": "bias_ih",
    "bias_hh_l0": "bias_hh",
}


def output_dir(config: p.PrestigeConfig) -> Path:
    return Path(config.output_dir) / "diagnostics" / "prestige1"


def checkpoint_dir(config: p.PrestigeConfig) -> Path:
    return Path(config.output_dir) / "checkpoints" / "prestige1"


def _verify_stage_a(config: p.PrestigeConfig, *, device: torch.device):
    stage_file = output_dir(config) / p.STAGE_A_REPORT
    if not stage_file.is_file():
        raise RuntimeError("Stage A v2 report missing; no transfer possible")
    stage = json.loads(stage_file.read_text(encoding="utf-8"))
    if stage.get("format") != p.VERSION or stage.get("stage") != "A":
        raise RuntimeError("Stage A v2 format mismatch")
    metadata = stage.get("metadata", {})
    if metadata.get("config_sha256") != config.signature():
        raise RuntimeError("Stage A config changed; refuse experimental transfer")
    if metadata.get("expected_old_best_sha256") not in (None, p.BEST_SHA256):
        raise RuntimeError("Stage A Best provenance mismatch")
    if stage.get("train_partition", {}).get("validation_used_for_selection") is not False:
        raise RuntimeError("Stage A data provenance contains Validation selection")
    if stage.get("gate", {}).get("gate_data_role") != "train_heldout":
        raise RuntimeError("Stage A did not use the chart-disjoint Train gate")
    # A failed 20%-MAE Stage A gate is NOT silently changed to PASS.
    artifact = output_dir(config) / "gru_v2.pt"
    expected = stage.get("checkpoint_sha256", {}).get("gru")
    if not artifact.is_file() or not expected or p.sha256(artifact) != expected:
        raise RuntimeError("Stage A pretrained GRU SHA256 mismatch")
    saved = torch.load(artifact, map_location="cpu", weights_only=True)
    if (saved.get("format") != p.VERSION or saved.get("stage") != "A"
            or saved.get("name") != "gru" or saved.get("initial_seed") != config.seed
            or saved.get("metadata") != metadata
            or saved.get("train_partition") != stage.get("train_partition")):
        raise RuntimeError("Stage A GRU checkpoint provenance mismatch")
    model = p.TemporalReadout(config.temporal_hidden, recurrent=True).to(device)
    model.load_state_dict(saved["state"], strict=True)
    model.eval()
    return stage, model, str(expected)


def transfer_temporal_weights(readout: p.TemporalReadout,
                              student: p.PrestigeMotorStudent) -> None:
    if not readout.recurrent:
        raise ValueError("Only Stage A recurrent GRU transfers into GRUCell")
    actual = set(readout.state_dict())
    if actual != {"encoder." + x for x in GRU_WEIGHTS} | {"head.weight", "head.bias"}:
        raise RuntimeError("Unexpected Stage A GRU state layout")
    with torch.no_grad():
        for source, dest in GRU_WEIGHTS.items():
            source_tensor = getattr(readout.encoder, source)
            dest_tensor = getattr(student.temporal, dest)
            if source_tensor.shape != dest_tensor.shape:
                raise RuntimeError("Temporal GRU/GRUCell incompatible shapes")
            dest_tensor.copy_(source_tensor)
        if (readout.head.weight.shape != student.timing_head.weight.shape
                or readout.head.bias.shape != student.timing_head.bias.shape):
            raise RuntimeError("Stage A/Stage C timing-head dimension mismatch")
        student.timing_head.weight.copy_(readout.head.weight)
        student.timing_head.bias.copy_(readout.head.bias)


@torch.no_grad()
def transfer_parity(readout: p.TemporalReadout, student: p.PrestigeMotorStudent,
                    *, seed: int) -> dict:
    """Check multi-frame numerical equality of GRU and mapped GRUCell."""
    device = student.temporal.weight_ih.device
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    xs = torch.randn(24, p.INPUT_DIM, generator=generator).to(device) * .1
    xs[0].zero_()
    student.eval()
    readout.eval()
    reference = readout(xs)
    state = torch.zeros(student.temporal_hidden, dtype=xs.dtype, device=device)
    mapped = []
    for x in xs:
        state = student.temporal(x, state)
        mapped.append(student.timing_head(state))
    candidate = torch.stack(mapped)
    max_abs = float((reference - candidate).abs().max())
    threshold = PARITY_ATOL_CUDA if device.type == "cuda" else PARITY_ATOL_CPU
    if not math.isfinite(max_abs) or max_abs > threshold:
        raise RuntimeError(f"Stage B GRUCell transfer parity failed: {max_abs} > {threshold}")
    return {"frames": len(xs), "max_abs": max_abs, "atol": threshold, "passed": True}


def _student(config: p.PrestigeConfig, readout: p.TemporalReadout,
             *, device: torch.device) -> p.PrestigeMotorStudent:
    # Old Best weights are NEVER loaded; only fixed MaleCNS structure and
    # pretrained Stage A visible-only Timing weights are inherited.
    torch.manual_seed(config.seed)
    model = p.PrestigeMotorStudent(config.core_path, config.temporal_hidden).to(device)
    transfer_temporal_weights(readout, model)
    model.prepare_recurrent_runtime()
    return model


def run_stage_b(config: p.PrestigeConfig) -> dict:
    output = output_dir(config) / TRANSFER_REPORT
    if output.exists():
        raise FileExistsError("Stage B report already exists: " + str(output))
    best_hash = p._protect_old_best(config)
    core = Path(config.core_path)
    if not core.is_file():
        raise FileNotFoundError("Prestige MaleCNS core missing: " + str(core))
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    stage, readout, readout_hash = _verify_stage_a(config, device=device)
    model = _student(config, readout, device=device)
    if model.backbone.hidden_dim != 4096:
        raise RuntimeError("MaleCNS4096 topology changed")
    parity = transfer_parity(readout, model, seed=config.seed)
    result = {
        "format": EXPERIMENT_VERSION, "stage": "B",
        "stage_a_original_gate": stage["gate"],
        "stage_a_gate_overridden": False,
        "purpose": "exploratory physical utility of visible-only pretrained temporal representations",
        "parity": parity, "pretrained_sha256": readout_hash,
        "core_sha256": p.sha256(core), "config_sha256": config.signature(),
        "best_sha256": best_hash, "no_motor_updates": True,
        "validation_used_for_selection": False, "no_final": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    p._no_replace_json(output, result)
    return result


def _verify_stage_b(config: p.PrestigeConfig, *, device: torch.device):
    file = output_dir(config) / TRANSFER_REPORT
    if not file.is_file():
        raise RuntimeError("Stage B transfer/parity must complete before Stage C v2")
    stage, readout, readout_hash = _verify_stage_a(config, device=device)
    report = json.loads(file.read_text(encoding="utf-8"))
    if (report.get("format") != EXPERIMENT_VERSION or report.get("stage") != "B"
            or report.get("config_sha256") != config.signature()
            or report.get("pretrained_sha256") != readout_hash
            or report.get("core_sha256") != p.sha256(Path(config.core_path))
            or report.get("stage_a_original_gate") != stage["gate"]
            or report.get("stage_a_gate_overridden") is not False
            or report.get("validation_used_for_selection") is not False
            or report.get("no_final") is not True
            or report.get("no_motor_updates") is not True
            or report.get("parity", {}).get("passed") is not True):
        raise RuntimeError("Stage B v2 provenance/transfer audit altered")
    return stage, readout, report


def _weights_equal(left: p.PrestigeMotorStudent, right: p.PrestigeMotorStudent):
    p._identical_initial_states(left, right)


def run_stage_c_v2(config: p.PrestigeConfig) -> dict:
    report_path = output_dir(config) / MOTOR_REPORT
    paths = {arm: checkpoint_dir(config) / f"{arm}_v2.pt"
             for arm in ("control", "prestige")}
    if report_path.exists() or any(f.exists() for f in paths.values()):
        raise FileExistsError("Existing Stage C v2 output; never overwrite a checkpoint")
    before_best = p._protect_old_best(config)
    core = Path(config.core_path)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    stage_a, pretrained, stage_b = _verify_stage_b(config, device=device)
    # Stage A's original failed 20%-MAE criterion stays a failed criterion.
    # This is a separately designated *exploratory* physical comparison.
    sequences, signature, segments = p.load_role_sequences(config, device=device)
    if signature != stage_a["metadata"]["dataset_split"]:
        raise RuntimeError("Stage A and Stage C chart identity/split differ")
    options = p._fixed_chunks(sequences["train"], config.batch_frames, require_labels=False)
    effective_updates = max(config.stage_c_updates, len(options))
    rng = random.Random(config.seed)
    schedule = list(range(len(options)))
    rng.shuffle(schedule)
    repeats = [schedule[i % len(schedule)] for i in range(effective_updates)]
    from dmdod.training.n_key import n_key_actuation_loss
    from dmdod.training.real_chart import evaluate_role_continuous
    control = _student(config, pretrained, device=device)
    prestige = _student(config, pretrained, device=device)
    _weights_equal(control, prestige)
    parity = transfer_parity(pretrained, control, seed=config.seed)
    results = {}
    # Complete BOTH fixed-budget Train-only optimizations before EITHER
    # Validation physical evaluation. Identical initial weights, schedule,
    # optimizer type, learning rate and train frames.
    for arm, model in (("control", control), ("prestige", prestige)):
        optimizer = torch.optim.AdamW(
            [w for w in model.parameters() if w.requires_grad],
            lr=config.learning_rate)
        logs = []
        model.train()
        for i, selected in enumerate(repeats, 1):
            sequence, start, end = options[selected]
            optimizer.zero_grad(set_to_none=True)
            actions, timing, _ = model.forward_sequence(
                sequence.visible[start:end], model.initial_state(device))
            motor = n_key_actuation_loss(
                torch.tanh(actions), sequence.teacher_action[start:end])
            auxiliary = p.timing_loss(
                timing, sequence.truth[start:end], sequence.mask[start:end])
            total = motor + (config.timing_weight * auxiliary if arm == "prestige" else 0.)
            if not bool(torch.isfinite(total)):
                raise RuntimeError("Nonfinite Stage C v2 Train loss")
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                [w for w in model.parameters() if w.requires_grad],
                1., error_if_nonfinite=True)
            optimizer.step()
            logs.append({
                "step": i, "motor_loss": float(motor.detach()),
                "timing_loss": float(auxiliary.detach()),
                "total_loss": float(total.detach()), "grad_norm": float(norm)})
            if i == 1 or i % 16 == 0 or i == effective_updates:
                print(f"Prestige Stage C v2 {arm} {i}/{effective_updates}: "
                      f"motor={logs[-1]['motor_loss']:.5f}, "
                      f"timing={logs[-1]['timing_loss']:.5f}", flush=True)
        results[arm] = {
            "model": model, "optimizer": optimizer.state_dict(),
            "loss_history": logs,
        }
    # Strictly after Train has completed for both arms: freeze + physical eval.
    for arm in ("control", "prestige"):
        item = results[arm]
        model = item.pop("model")
        model.eval()
        with torch.no_grad():
            replay = evaluate_role_continuous(
                model, segments["validation"], label="Prestige C v2 " + arm,
                control_dt_s=p.CONTROL_DT, physics_dt_s=p.PHYSICS_DT,
                device=device, verbose=False)
            timing = p._motor_timing_metrics(model, sequences["validation"])
        rows = p._results_to_dict(replay, segments["validation"])
        item.update({
            "validation": p._aggregate_motor(rows),
            "validation_per_anchor": rows, "timing_validation": timing,
            "parameter_count": sum(w.numel() for w in model.parameters()),
        })
        payload = {
            "format": EXPERIMENT_VERSION, "stage": "C", "arm": arm,
            "config_sha256": config.signature(),
            "pretrained_sha256": stage_b["pretrained_sha256"],
            "core_sha256": stage_b["core_sha256"], "model": model.state_dict(),
            "optimizer": item.pop("optimizer"),
            "updates": effective_updates,
            "train_anchor_manifest": signature["train"],
            "loss_history": item["loss_history"], "seed": config.seed,
            "warm_start_old_best": False,
        }
        item["checkpoint"] = str(paths[arm])
        item["checkpoint_sha256"] = p._save_checkpoint(paths[arm], payload)
    if p.sha256(core) != stage_b["core_sha256"]:
        raise RuntimeError("MaleCNS topology core changed during Stage C v2")
    if p._protect_old_best(config) != before_best:
        raise RuntimeError("Old Best identity changed")
    report = {
        "format": EXPERIMENT_VERSION, "stage": "C",
        "exploratory": True, "validation_used_for_selection": False,
        "stage_a_original_gate": stage_a["gate"], "stage_a_gate_overridden": False,
        "stage_b_parity": parity, "pretrained_sha256": stage_b["pretrained_sha256"],
        "core_sha256": stage_b["core_sha256"],
        "effective_updates_per_arm": effective_updates,
        "training_chunks": len(options), "full_train_pass": effective_updates >= len(options),
        "identical_pretrained_initial_weights": True,
        "same_minibatch_schedule": True,
        "experiment": "Control and Prestige both load same visible-only Stage A GRU; "
                      "only Prestige continues Timing auxiliary supervision. "
                      "No old Student weights. No hyperparameter selection from Validation.",
        "results": results, "no_final": True,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    p._no_replace_json(report_path, report)
    return report
