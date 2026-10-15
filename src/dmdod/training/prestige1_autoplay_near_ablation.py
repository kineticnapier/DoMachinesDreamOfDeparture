from __future__ import annotations

"""Exactly one controlled Stage T GRU loss ablation: remove far-time regression.

This experiment is NOT adopted even if it passes the research-only gate.
The only training change from the frozen Stage T GRU is the timing loss:
physical exact-time labels remain the same, but regression is supervised
only for events within 200ms. BCE near-event classification still sees
all visible frames. Same model, seed, chunks, optimizer, LR, and updates.
"""

from pathlib import Path
import json
import math
import random

import torch
from torch.nn import functional as F

from . import prestige1 as p
from . import prestige1_autoplay_readout as stage_t
from . import prestige1_autoplay_diagnostics as audit
from . import prestige1_transfer as transfer

FORMAT = "prestige1-stage-t-near-only-regression-ablation-v1"
REPORT = "stage_timing_autoplay_near_only_gru_train20.json"
CHECKPOINT = "stage_timing_autoplay_near_only_gru.pt"


def near_only_timing_loss(
    raw: torch.Tensor, sequence: stage_t.TimingSequence, start: int, end: int,
) -> torch.Tensor:
    valid = sequence.mask[start:end]
    if not bool(valid.any()):
        return raw.sum() * 0.
    predicted_s, probability = stage_t.decode(raw)
    near = sequence.near[start:end][valid]
    # No oracle label is ever an inference input. Far events are still
    # classified as negatives, but their potentially huge true time values
    # cannot dominate the timestamp regression objective.
    event_loss = F.binary_cross_entropy(
        probability[valid], near.to(probability.dtype))
    if not bool(near.any()):
        return event_loss
    timing = F.smooth_l1_loss(
        predicted_s[valid][near] / stage_t.TIME_UNIT_S,
        sequence.remaining_s[start:end][valid][near] / stage_t.TIME_UNIT_S,
    )
    return timing + event_loss


def train_near_only(model: stage_t.TimingReadout,
                    fit: list[stage_t.TimingSequence],
                    *, updates: int, frames: int, lr: float, seed: int):
    # This follows stage_t.train_one exactly other than which loss is called.
    choices = stage_t._chunks(fit, frames)
    order = list(range(len(choices)))
    rng = random.Random(seed)
    rng.shuffle(order)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    history = []
    for step in range(max(updates, len(choices))):
        if step and step % len(order) == 0:
            rng.shuffle(order)
        sequence, start, end = choices[order[step % len(order)]]
        optimizer.zero_grad(set_to_none=True)
        raw = model(sequence.visible[start:end])
        loss = near_only_timing_loss(raw, sequence, start, end)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite near-only Timing loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        history.append(float(loss.detach()))
    return history, len(choices)


def run_near_only_ablation(config: p.PrestigeConfig) -> dict:
    root = transfer.output_dir(config)
    old_distance = root / audit.REPORT
    report_path = root / REPORT
    checkpoint_path = root / CHECKPOINT
    if any(path.exists() for path in (report_path, checkpoint_path)):
        raise FileExistsError("Near-only ablation outputs already exist")
    if not old_distance.is_file():
        raise FileNotFoundError(
            "Run the read-only Stage T distance audit and check its result first")
    old_distance_sha = p.sha256(old_distance)
    old_distance_report = json.loads(old_distance.read_text(encoding="utf-8"))
    source = audit.prepare(config, output_paths=(report_path, checkpoint_path))
    if (
        old_distance_report.get("format") != audit.FORMAT
        or old_distance_report.get("no_training") is not True
        or old_distance_report.get("validation_evaluated") is not False
        or old_distance_report.get("final_evaluated") is not False
        or old_distance_report.get("source_sha256") != source.source_sha256
        or old_distance_report.get("partition") != source.partition
    ):
        raise RuntimeError("Diagnostic must match frozen Stage T checkpoint provenance")
    # Guard against a changed re-capture or a corrupted pre-ablation
    # diagnostic, even though the original Stage T MAE has been rechecked.
    current = {
        split: {
            name: audit.distance_metrics(model, group, source.reference)
            for name, model in (("constant", None), *source.models.items())
        }
        for split, group in (("train_fit", source.fit), ("train_heldout", source.heldout))
    }
    audit._verify_distance_rollup(current, source)
    for split in ("train_fit", "train_heldout"):
        for name in ("constant", "mlp", "gru"):
            for b in audit.BINS:
                prev = old_distance_report["scores"][split][name]["aggregate"][b]
                present = current[split][name]["aggregate"][b]
                if (
                    prev["frames"] != present["frames"]
                    or (
                        prev["mae_ms"] is not None and
                        abs(prev["mae_ms"] - present["mae_ms"]) > .05
                    )
                ):
                    raise RuntimeError("Stage T distance baseline changed")

    torch.manual_seed(config.seed)
    model = stage_t.TimingReadout(
        config.temporal_hidden, recurrent=True).to(
            torch.device(config.device))
    if not all(bool(torch.equal(
        a.cpu(), b.cpu()
    )) for a, b in zip(
        model.state_dict().values(),
        # Match original GRU's freshly seeded initialization, not its
        # *trained* weights. The check below explicitly reconstructs it.
        _fresh_seeded_gru(config).state_dict().values(),
    )):
        raise RuntimeError("GRU initial weights are not reproducible")

    history, chunk_count = train_near_only(
        model, source.fit, updates=config.stage_a_updates,
        frames=config.batch_frames, lr=config.learning_rate, seed=config.seed)
    original_coverage = source.stage_report["training_coverage"]["gru"]
    if (
        original_coverage["nonempty_fit_chunks"] != chunk_count
        or original_coverage["updates"] != len(history)
    ):
        raise RuntimeError("Controlled ablation train schedule changed")

    reference = source.reference
    current_scores = {
        split: stage_t.evaluate(model, group, reference)
        for split, group in (("train_fit", source.fit), ("train_heldout", source.heldout))
    }
    new_bins = {
        split: audit.distance_metrics(model, group, reference)
        for split, group in (("train_fit", source.fit), ("train_heldout", source.heldout))
    }
    gate = stage_t.exploratory_gate({
        "constant": source.stage_report["train_heldout"]["constant"],
        "mlp": source.stage_report["train_heldout"]["mlp"],
        "gru": current_scores["train_heldout"],
    })
    if (
        old_distance_sha != p.sha256(old_distance)
        or source.source_sha256["stage_t_readout"] != p.sha256(root / stage_t.REPORT)
        or p._protect_old_best(config) != source.original_best_sha
    ):
        raise RuntimeError("Frozen artifact changed before near-only save")

    root.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": FORMAT, "model_name": "gru_near_only",
        "model_state_dict": model.state_dict(),
        "config_sha256": config.signature(),
        "baseline_sha256": source.source_sha256,
        "distance_audit_sha256": old_distance_sha,
        "train_partition": source.partition,
        "research_only": True, "motor_adoption_permitted": False,
    }
    with checkpoint_path.open("xb") as f:
        torch.save(checkpoint, f)
    checkpoint_sha = p.sha256(checkpoint_path)
    result = {
        "format": FORMAT,
        "role": "Train20 chart-disjoint fit/heldout only",
        "experiment": "One loss-only change: exact-time regression near <=200ms; BCE near classifier on all visible frames",
        "only_changed_variable": "remove far >200ms timestamp regression term",
        "training_fingerprint": {
            "model": "GRU 263D -> 128 hidden -> time, event",
            "seed": config.seed,
            "batch_frames": config.batch_frames,
            "learning_rate": config.learning_rate,
            "updates": len(history),
            "fit_chunks": chunk_count,
            "loss_changed_only": True,
        },
        "source_sha256": source.source_sha256,
        "distance_audit_sha256": old_distance_sha,
        "checkpoint_sha256": checkpoint_sha,
        "train_partition": source.partition,
        "old_research_gate": source.stage_report["research_gate"],
        "new_research_gate": gate,
        "old_stage_a_gate": source.stage_report["original_stage_a_gate"],
        "old_stage_a_gate_overridden": False,
        "automatically_adopted": False,
        "motor_student_training_permitted": False,
        "validation_evaluated": False,
        "final_evaluated": False,
        "fingering_trained": False,
        "motor_student_trained": False,
        "training_losses": history,
        "baseline": {
            split: {
                name: source.stage_report[split][name]["aggregate"]
                for name in ("constant", "mlp", "gru")
            }
            for split in ("train_fit", "train_heldout")
        },
        "near_only_gru": {
            split: current_scores[split]
            for split in ("train_fit", "train_heldout")
        },
        "near_only_distance_bins": new_bins,
        "limitations": [
            "Exactly one loss-only ablation; no learning-rate, epoch or model sweep.",
            "This removes far-distance time regression and may worsen all-visible MAE.",
            "Exact source timestamps are preserved; pretrained Stage T checkpoints untouched.",
            "Expert motor trajectories can act as visible anticipatory cues.",
            "No closed-loop Student, original Stage A pass, Validation or FINAL.",
        ],
    }
    p._no_replace_json(report_path, result)
    return result


def _fresh_seeded_gru(config: p.PrestigeConfig) -> stage_t.TimingReadout:
    torch.manual_seed(config.seed)
    return stage_t.TimingReadout(config.temporal_hidden, recurrent=True)
