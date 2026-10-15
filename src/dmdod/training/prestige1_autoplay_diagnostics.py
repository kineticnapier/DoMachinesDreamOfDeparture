from __future__ import annotations

"""Read-only timing-distance audit of the frozen Stage T MLP/GRU checkpoints.

Evaluation truth (exact compiled Autoplay schedule) is separate from causal
263D HUD network input. This module never changes Stage A or a checkpoint.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import math

import torch

from . import prestige1 as p
from . import prestige1_alignment_audit as alignment
from . import prestige1_autoplay_readout as stage_t
from . import prestige1_autoplay_timing as autoplay
from . import prestige1_transfer as transfer


FORMAT = "prestige1-stage-t-distance-bins-train20-v1"
REPORT = "stage_timing_autoplay_distance_bins_train20.json"
BINS = ("0-50ms", "50-100ms", "100-200ms", "over-200ms")


@dataclass(frozen=True)
class FrozenTimingInputs:
    root: Path
    stage_report: dict
    source_sha256: dict[str, str]
    fit: list[stage_t.TimingSequence]
    heldout: list[stage_t.TimingSequence]
    partition: dict
    reference: dict
    models: dict[str, stage_t.TimingReadout]
    original_best_sha: str | None


def _verify_source_document(config: p.PrestigeConfig, root: Path):
    source = root / stage_t.REPORT
    if not source.is_file():
        raise FileNotFoundError("Stage T result missing: " + str(source))
    report = json.loads(source.read_text(encoding="utf-8"))
    if (
        report.get("format") != stage_t.FORMAT
        or report.get("config_sha256") != config.signature()
        or report.get("role") != "Train20 fit / chart-disjoint Train heldout ONLY"
        or report.get("original_stage_a_gate_overridden") is not False
        or report.get("validation_evaluated") is not False
        or report.get("final_evaluated") is not False
        or report.get("motor_student_training_permitted") is not False
        or report.get("automatically_adopted") is not False
        or report.get("original_stage_a_gate", {}).get("passed") is not False
        or report.get("research_gate", {}).get("observability_evidence_passed") is not False
        or set(report.get("experimental_checkpoint_sha256", {})) != {"mlp", "gru"}
    ):
        raise RuntimeError("Frozen Stage T FAIL provenance disagrees")
    return report, p.sha256(source)


def _checkpoint(config: p.PrestigeConfig, path: Path, expected_sha: str,
                name: str, partition: dict, oracle_sha: str, device: torch.device):
    if not path.is_file() or p.sha256(path) != expected_sha:
        raise RuntimeError("Checkpoint integrity failure: " + str(path))
    payload = torch.load(path, weights_only=True, map_location="cpu")
    if (
        payload.get("format") != stage_t.FORMAT
        or payload.get("model_name") != name
        or payload.get("config_sha256") != config.signature()
        or payload.get("train_partition") != partition
        or payload.get("oracle_sha256") != oracle_sha
        or payload.get("research_only") is not True
        or payload.get("motor_adoption_permitted") is not False
    ):
        raise RuntimeError("Stage T checkpoint metadata mismatch: " + str(path))
    model = stage_t.TimingReadout(config.temporal_hidden, recurrent=name=="gru")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if not all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()):
        raise RuntimeError("Nonfinite Stage T model parameters")
    return model.to(device).eval()


def prepare(config: p.PrestigeConfig, *, output_paths: Iterable[Path] = ()) -> FrozenTimingInputs:
    root = transfer.output_dir(config)
    output_paths = tuple(output_paths)
    if any(path.exists() for path in output_paths):
        raise FileExistsError("Refuse to overwrite Timing audit/ablation output")
    old_sha = p._protect_old_best(config)
    old_a, oracle, frozen_sources = stage_t._read_frozen_sources(config, root)
    report, report_sha = _verify_source_document(config, root)
    if (
        report["original_stage_a_gate"] != old_a["gate"]
        or report["oracle_source_sha256"] != frozen_sources
        or report["train_manifest"] != oracle["train_manifest"]
        or report["train_manifest"] != old_a["metadata"]["dataset_split"]["train"]
    ):
        raise RuntimeError("Frozen Stage T/Stage A/Autoplay Oracle mismatch")
    selected, manifest = alignment._train_manifest(config)
    if manifest != report["train_manifest"]:
        raise RuntimeError("Train20 manifest changed")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA unavailable")
    sequences = []
    for i, (named, expected) in enumerate(zip(selected, oracle["per_anchor"]), 1):
        truth = autoplay.AutoplayTimingTeacher.from_segment(named.segment)
        timeline = [
            {
                "event_index": event.event_index,
                "target_ordinals": list(event.target_ordinals),
                "floor_indices": list(event.floor_indices),
                "episode_time_s": event.episode_time_s,
                "chart_time_s": event.chart_time_s,
                "chord_size": event.chord_size,
            }
            for event in truth.events
        ]
        if (
            expected["anchor"] != i
            or expected["chart"] != named.chart_name
            or expected["chart_sha256"] != named.chart_sha256
            or expected["timeline"] != timeline
        ):
            raise RuntimeError("Exact timing schedule changed at Train anchor " + str(i))
        seq = stage_t.capture_one(named, i, config, device=device)
        sequences.append(seq)
        print(f"Distance audit capture Train {i:02d}/20 ({len(seq.remaining_s)} frames)", flush=True)
    fit, heldout, partition = p.split_stage_a_train(
        sequences, {"train": report["train_manifest"]}, seed=config.seed)
    if partition != report["train_partition"]:
        raise RuntimeError("Chart-disjoint Train partition changed")
    reference = stage_t.constant_reference(fit)
    if (
        reference["fit_valid_frames"] != report["reference"]["fit_valid_frames"]
        or reference["fit_near_positive_count"] != report["reference"]["fit_near_positive_count"]
        or abs(reference["time_s"] - report["reference"]["time_s"]) > 1e-6
        or abs(reference["near_probability"] - report["reference"]["near_probability"]) > 1e-6
    ):
        raise RuntimeError("Previously trained constant reference changed")
    models = {
        name: _checkpoint(
            config,
            root / (stage_t.GRU_CHECKPOINT if name == "gru" else stage_t.MLP_CHECKPOINT),
            report["experimental_checkpoint_sha256"][name],
            name, partition, frozen_sources["exact_autoplay_oracle"], device,
        )
        for name in ("mlp", "gru")
    }
    # Recompute metrics from re-captured visible HUD, refusing to silently
    # compare checkpoints against drifted simulator trajectories.
    for split_name, group in (("train_fit", fit), ("train_heldout", heldout)):
        for name, model in (("constant", None), *models.items()):
            current = stage_t.evaluate(model, group, reference)["aggregate"]
            saved = report[split_name][name]["aggregate"]
            for field in ("frames_observable", "near_positive_frames", "near_negative_frames",
                          "tp", "tn", "fp", "fn"):
                if current[field] != saved[field]:
                    raise RuntimeError(f"Stage T replay changed {split_name}/{name}/{field}")
            for field in ("near_mae_ms", "all_visible_mae_ms", "near_p95_abs_ms"):
                if abs(current[field] - saved[field]) > .05:
                    raise RuntimeError(
                        f"Stage T replay changed {split_name}/{name}/{field}: "
                        f"{current[field]} vs {saved[field]}")
    if (
        p.sha256(root / stage_t.REPORT) != report_sha
        or frozen_sources != stage_t._read_frozen_sources(config, root)[2]
        or p._protect_old_best(config) != old_sha
    ):
        raise RuntimeError("Frozen training artifacts changed during replay")
    return FrozenTimingInputs(
        root=root, stage_report=report,
        source_sha256={
            "stage_t_readout": report_sha,
            "stage_t_mlp_checkpoint": report["experimental_checkpoint_sha256"]["mlp"],
            "stage_t_gru_checkpoint": report["experimental_checkpoint_sha256"]["gru"],
            **frozen_sources,
        },
        fit=fit, heldout=heldout, partition=partition,
        reference=reference, models=models, original_best_sha=old_sha,
    )


def _bin_masks(seq: stage_t.TimingSequence) -> dict[str, torch.Tensor]:
    valid = seq.mask
    # The near flag is defined by the exact double-precision oracle, rather
    # than by float32 rounding at 200ms. Every valid frame gets exactly one bin.
    masks = {
        BINS[0]: valid & seq.near & (seq.remaining_s < .050),
        BINS[1]: valid & seq.near & (seq.remaining_s >= .050) & (seq.remaining_s < .100),
        BINS[2]: valid & seq.near & (seq.remaining_s >= .100),
        BINS[3]: valid & ~seq.near,
    }
    count = sum(int(mask.sum()) for mask in masks.values())
    if count != int(valid.sum()):
        raise RuntimeError("Distance bins do not partition observable frames")
    return masks


@torch.no_grad()
def distance_metrics(model: stage_t.TimingReadout | None,
                     group: list[stage_t.TimingSequence], reference: dict) -> dict:
    if model is not None:
        model.eval()
    errors: dict[str, list[torch.Tensor]] = {name: [] for name in BINS}
    signed: dict[str, list[torch.Tensor]] = {name: [] for name in BINS}
    rows = []
    for seq in group:
        if model is None:
            prediction = seq.remaining_s.new_full(seq.remaining_s.shape, reference["time_s"])
        else:
            prediction, _ = stage_t.decode(model(seq.visible))
        by_bin = _bin_masks(seq)
        metrics = {}
        for name, mask in by_bin.items():
            error = ((prediction[mask] - seq.remaining_s[mask]) * 1000.).cpu()
            errors[name].append(error.abs())
            signed[name].append(error)
            metrics[name] = {
                "frames": int(mask.sum()),
                "mae_ms": float(error.abs().mean()) if error.numel() else None,
            }
        rows.append({
            "anchor": seq.anchor,
            "chart_sha256": seq.chart_sha256,
            "bins": metrics,
        })
    aggregate = {}
    for name in BINS:
        absolute = torch.cat(errors[name])
        residual = torch.cat(signed[name])
        aggregate[name] = {
            "frames": absolute.numel(),
            "mae_ms": float(absolute.mean()) if absolute.numel() else None,
            "median_abs_ms": float(torch.quantile(absolute, .5)) if absolute.numel() else None,
            "p95_abs_ms": float(torch.quantile(absolute, .95)) if absolute.numel() else None,
            "mean_signed_ms": float(residual.mean()) if residual.numel() else None,
        }
    return {"aggregate": aggregate, "per_anchor": rows}


def _verify_distance_rollup(scores: dict, source: FrozenTimingInputs):
    for split, group in (("train_fit", source.fit), ("train_heldout", source.heldout)):
        for name in ("constant", "mlp", "gru"):
            buckets = scores[split][name]["aggregate"]
            original = source.stage_report[split][name]["aggregate"]
            near_frames = sum(buckets[b]["frames"] for b in BINS[:3])
            far_frames = buckets[BINS[3]]["frames"]
            if (
                near_frames != original["near_positive_frames"]
                or near_frames + far_frames != original["frames_observable"]
            ):
                raise RuntimeError(f"Distance bin frame mismatch {split}/{name}")
            # Near MAE must match the earlier aggregate, not merely look plausible.
            near_mae = sum(
                buckets[b]["mae_ms"] * buckets[b]["frames"]
                for b in BINS[:3] if buckets[b]["frames"]
            ) / near_frames
            if abs(near_mae - original["near_mae_ms"]) > .05:
                raise RuntimeError(f"Distance bin Near MAE mismatch {split}/{name}")


def run_distance_audit(config: p.PrestigeConfig) -> dict:
    root = transfer.output_dir(config)
    output = root / REPORT
    source = prepare(config, output_paths=(output,))
    result = {
        "format": FORMAT, "role": "Train20 chart-disjoint only",
        "description": "Read-only distance stratification of saved Stage T weights",
        "distance_bins": list(BINS), "no_training": True,
        "stage_a_overridden": False, "validation_evaluated": False,
        "final_evaluated": False, "motor_student_training_permitted": False,
        "source_sha256": source.source_sha256,
        "partition": source.partition,
        "scores": {
            split: {
                name: distance_metrics(model, group, source.reference)
                for name, model in (("constant", None), *source.models.items())
            }
            for split, group in (("train_fit", source.fit), ("train_heldout", source.heldout))
        },
    }
    _verify_distance_rollup(result["scores"], source)
    if p._protect_old_best(config) != source.original_best_sha:
        raise RuntimeError("Old Best modified during Timing audit")
    root.mkdir(parents=True, exist_ok=True)
    p._no_replace_json(output, result)
    return result
