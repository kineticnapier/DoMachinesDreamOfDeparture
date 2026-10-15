from __future__ import annotations

"""Read-only replay: did the saved near-only GRU learn a real countdown?

The original Stage T and near-only trained checkpoints are restored as-is.
Both models predict from the same causal 263D HUD stream, with heldout
Train charts separate from Train-fit. Exact Autoplay time and event IDs
are used ONLY as evaluator-side labels and diagnostic segmentation.
"""

from html import escape
import json
import math
from pathlib import Path

import torch

from . import prestige1 as p
from . import prestige1_autoplay_countdown as countdown
from . import prestige1_autoplay_diagnostics as audit
from . import prestige1_autoplay_near_ablation as near
from . import prestige1_autoplay_readout as stage_t
from . import prestige1_transfer as transfer

FORMAT = "prestige1-stage-t-near-only-vs-frozen-countdown-train20-v1"
REPORT = "stage_timing_autoplay_near_only_countdown_train20.json"
PLOT_PREFIX = "stage_timing_autoplay_near_only_countdown_holdout"


def _check_scores(current: dict, saved: dict, *, label: str) -> None:
    """Fail closed if restored inference disagrees with the frozen report."""
    now, previous = current["aggregate"], saved["aggregate"]
    for key in (
        "frames_observable", "frames_masked", "near_positive_frames",
        "near_negative_frames", "tp", "tn", "fp", "fn",
    ):
        if now[key] != previous[key]:
            raise RuntimeError(f"{label}: frozen score count mismatch: {key}")
    for key in (
        "near_mae_ms", "near_p50_abs_ms", "near_p95_abs_ms",
        "all_visible_mae_ms",
    ):
        if abs(now[key] - previous[key]) > .05:
            raise RuntimeError(f"{label}: frozen error mismatch: {key}")
    for key in ("false_positive_rate", "event_recall", "balanced_accuracy"):
        if abs(now[key] - previous[key]) > 1e-5:
            raise RuntimeError(f"{label}: frozen classification mismatch: {key}")
    if len(current["per_anchor"]) != len(saved["per_anchor"]):
        raise RuntimeError(f"{label}: frozen anchor count mismatch")
    for a, b in zip(current["per_anchor"], saved["per_anchor"]):
        for key in (
            "anchor", "chart_sha256", "frames", "observable_frames",
            "masked_frames", "near_positive_frames", "tp", "tn", "fp", "fn",
        ):
            if a[key] != b[key]:
                raise RuntimeError(f"{label}: frozen anchor mismatch: {key}")
        for key in ("near_mae_ms", "all_visible_mae_ms"):
            av, bv = a[key], b[key]
            if (av is None) != (bv is None) or (
                av is not None and abs(av - bv) > .05
            ):
                raise RuntimeError(f"{label}: frozen anchor MAE mismatch: {key}")


def _restore_near_only(
    config: p.PrestigeConfig, source: audit.FrozenTimingInputs,
) -> tuple[stage_t.TimingReadout, dict, dict[str, str]]:
    root = source.root
    report_path = root / near.REPORT
    ckpt_path = root / near.CHECKPOINT
    audit_path = root / audit.REPORT
    old_countdown_path = root / countdown.REPORT
    for path in (report_path, ckpt_path, audit_path, old_countdown_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen Timing audit/ablation absent: {path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    old_distance = json.loads(audit_path.read_text(encoding="utf-8"))
    old_countdown = json.loads(old_countdown_path.read_text(encoding="utf-8"))
    if (
        report.get("format") != near.FORMAT
        or report.get("role") != "Train20 chart-disjoint fit/heldout only"
        or report.get("only_changed_variable") != "remove far >200ms timestamp regression term"
        or report.get("old_stage_a_gate_overridden") is not False
        or report.get("old_stage_a_gate") != source.stage_report["original_stage_a_gate"]
        or report.get("automatically_adopted") is not False
        or report.get("motor_student_training_permitted") is not False
        or report.get("validation_evaluated") is not False
        or report.get("final_evaluated") is not False
        or report.get("fingering_trained") is not False
        or report.get("motor_student_trained") is not False
        or report.get("source_sha256") != source.source_sha256
        or report.get("train_partition") != source.partition
        or report.get("old_research_gate") != source.stage_report["research_gate"]
        or report.get("training_fingerprint", {}).get("seed") != config.seed
        or report.get("training_fingerprint", {}).get("batch_frames") != config.batch_frames
        or report.get("training_fingerprint", {}).get("learning_rate") != config.learning_rate
        or report.get("training_fingerprint", {}).get("loss_changed_only") is not True
        or report.get("training_fingerprint", {}).get("updates")
            != source.stage_report["training_coverage"]["gru"]["updates"]
        or report.get("training_fingerprint", {}).get("fit_chunks")
            != source.stage_report["training_coverage"]["gru"]["nonempty_fit_chunks"]
        or report.get("new_research_gate", {}).get("original_stage_a_gate_overridden") is not False
        or report.get("new_research_gate", {}).get("motor_student_training_permitted") is not False
        or old_distance.get("format") != audit.FORMAT
        or old_distance.get("source_sha256") != source.source_sha256
        or old_distance.get("partition") != source.partition
        or old_distance.get("no_training") is not True
        or old_distance.get("validation_evaluated") is not False
        or old_distance.get("final_evaluated") is not False
        or report.get("distance_audit_sha256") != p.sha256(audit_path)
        or old_countdown.get("format") != countdown.FORMAT
        or old_countdown.get("source_sha256") != source.source_sha256
        or old_countdown.get("train_partition") != source.partition
        or old_countdown.get("no_training") is not True
        or old_countdown.get("read_only_models") is not True
        or old_countdown.get("stage_a_overridden") is not False
        or old_countdown.get("validation_evaluated") is not False
        or old_countdown.get("final_evaluated") is not False
    ):
        raise RuntimeError("Near-only/frozen countdown provenance is inconsistent")
    if p.sha256(ckpt_path) != report.get("checkpoint_sha256"):
        raise RuntimeError("Near-only checkpoint SHA256 changed")
    payload = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    if (
        payload.get("format") != near.FORMAT
        or payload.get("model_name") != "gru_near_only"
        or payload.get("config_sha256") != config.signature()
        or payload.get("baseline_sha256") != source.source_sha256
        or payload.get("distance_audit_sha256") != p.sha256(audit_path)
        or payload.get("train_partition") != source.partition
        or payload.get("research_only") is not True
        or payload.get("motor_adoption_permitted") is not False
    ):
        raise RuntimeError("Near-only model checkpoint metadata changed")
    model = stage_t.TimingReadout(config.temporal_hidden, recurrent=True)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if not all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()):
        raise RuntimeError("Near-only model has nonfinite parameters")
    model = model.to(torch.device(config.device)).eval()

    for split, group in (("train_fit", source.fit), ("train_heldout", source.heldout)):
        _check_scores(
            stage_t.evaluate(model, group, source.reference),
            report["near_only_gru"][split], label=f"near_only/{split}",
        )
        for name, baseline in (("constant", None), *source.models.items()):
            _check_scores(
                stage_t.evaluate(baseline, group, source.reference),
                source.stage_report[split][name], label=f"frozen_{name}/{split}",
            )
            original = countdown.dynamics(baseline, group, source.reference)["aggregate"]
            expected = old_countdown["timing_dynamics"][split][name]["aggregate"]
            for section in (
                "same_event_countdown", "same_event_near_countdown",
                "event_switch_residual",
            ):
                if original[section]["count"] != expected[section]["count"]:
                    raise RuntimeError("Frozen Countdown frame count changed")
                if original[section]["mae_ms"] is not None and abs(
                    original[section]["mae_ms"] - expected[section]["mae_ms"]
                ) > .05:
                    raise RuntimeError("Frozen Countdown MAE changed")
            for field in (
                "true_positive_delta_switches",
                "correctly_upward_switches", "nonpositive_delta_switches",
            ):
                if original[field] != expected[field]:
                    raise RuntimeError("Frozen Countdown switch counts changed")
    fingerprints = {
        **source.source_sha256,
        "near_only_report": p.sha256(report_path),
        "near_only_checkpoint": p.sha256(ckpt_path),
        "distance_audit": p.sha256(audit_path),
        "old_countdown": p.sha256(old_countdown_path),
    }
    return model, report, fingerprints


def _near_slope_metrics(dynamics: dict) -> dict:
    aggregate = dynamics["aggregate"]
    near_stats = aggregate["same_event_near_countdown"]
    signed = near_stats["bias_ms"]
    if signed is None:
        raise RuntimeError("No consecutive near-event frames")
    predicted_near_step = signed - 1000 * p.CONTROL_DT
    return {
        "near_countdown_mae_ms": near_stats["mae_ms"],
        "near_countdown_p95_ms": near_stats["p95_abs_ms"],
        "near_countdown_bias_ms": signed,
        "mean_predicted_near_step_ms": predicted_near_step,
        "ideal_step_ms": -1000 * p.CONTROL_DT,
        "mean_predicted_step_ms": aggregate["mean_predicted_step_ms"],
        "same_event_pair_count": aggregate["same_event_countdown"]["count"],
        "near_pair_count": near_stats["count"],
        "switch_mae_ms": aggregate["event_switch_residual"]["mae_ms"],
        "switch_count": aggregate["event_switch_residual"]["count"],
        "switch_rise_recall": aggregate["upward_switch_recall"],
        "switch_rise_count": aggregate["true_positive_delta_switches"],
        "switch_rise_detected": aggregate["correctly_upward_switches"],
    }


def _geometry_near_subset(
    group: list[stage_t.TimingSequence], original: stage_t.TimingReadout,
    new: stage_t.TimingReadout, reference: dict,
) -> dict:
    """Score existing HUD-only geometry and two GRUs on matched eligible frames."""
    signed = {"geometry": [], "old_gru": [], "near_only_gru": []}
    eligible = 0
    for seq in group:
        frames = [tuple(row) for row in seq.visible.detach().cpu().tolist()]
        old_pred = countdown._predicted_seconds(seq, original, reference)
        new_pred = countdown._predicted_seconds(seq, new, reference)
        for i in range(1, len(frames)):
            if not (bool(seq.mask[i]) and bool(seq.near[i])):
                continue
            estimated, _ = countdown.geometry_one(frames[i-1], frames[i])
            if estimated is None:
                continue
            eligible += 1
            actual = float(seq.remaining_s[i])
            for name, estimate in (
                ("geometry", estimated),
                ("old_gru", old_pred[i]),
                ("near_only_gru", new_pred[i]),
            ):
                signed[name].append((estimate-actual)*1000)
    return {
        "eligible_near_frames": eligible,
        "matched_only": True,
        "error": {name: countdown._summary(values) for name, values in signed.items()},
    }


def _comparison_svg(seq: stage_t.TimingSequence, reference: dict,
                    original: stage_t.TimingReadout,
                    new: stage_t.TimingReadout) -> str:
    start, end = countdown._representative_window(seq)
    width, height = 920, 360
    left, right, top, bottom = 75, 24, 50, 55
    ceiling = .350  # display only; loss and reported metrics unmodified
    def x(index):
        return left+(index-start)/max(1, end-start-1)*(width-left-right)
    def y(seconds):
        return top+(1-max(0., min(ceiling, seconds))/ceiling)*(height-top-bottom)
    models = (
        ("Auto truth", [float(v) for v in seq.remaining_s.cpu().tolist()], "#111111"),
        ("Constant", countdown._predicted_seconds(seq, None, reference), "#999999"),
        ("Old GRU", countdown._predicted_seconds(seq, original, reference), "#3066be"),
        ("Near-only GRU", countdown._predicted_seconds(seq, new, reference), "#13844c"),
    )
    result = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="24" font-size="17" fill="#111">Train heldout #{seq.anchor:02d} — old vs near-only GRU (view capped at 350ms)</text>',
    ]
    for ms in (0, 100, 200, 300):
        yy=y(ms/1000)
        result.append(f'<line x1="{left}" x2="{width-right}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#ddd"/>')
        result.append(f'<text x="12" y="{yy+4:.1f}" font-size="13">{ms}ms</text>')
    for k in range(start+1, end):
        if seq.event_indices[k] != seq.event_indices[k-1]:
            xx=x(k)
            result.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{top}" y2="{height-bottom}" stroke="#aaa" stroke-dasharray="4,4"/>')
    for name, values, color in models:
        points=" ".join(f"{x(i):.1f},{y(values[i]):.1f}" for i in range(start,end))
        result.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    for i, (name, _values, color) in enumerate(models):
        xx=left+i*192
        yy=height-19
        result.append(f'<line x1="{xx}" x2="{xx+24}" y1="{yy}" y2="{yy}" stroke="{color}" stroke-width="3"/>')
        result.append(f'<text x="{xx+30}" y="{yy+4}" font-size="13" fill="#333">{escape(name)}</text>')
    result.append('</svg>')
    return "\n".join(result)+"\n"


def run_near_countdown_audit(config: p.PrestigeConfig) -> dict:
    root = transfer.output_dir(config)
    report_path = root / REPORT
    plot_paths = {
        i: root / f"{PLOT_PREFIX}_{i:02d}.svg"
        for i in countdown.PLOT_ANCHORS
    }
    source = audit.prepare(config, output_paths=(report_path, *plot_paths.values()))
    model, near_report, fingerprints = _restore_near_only(config, source)
    frozen_sha = dict(fingerprints)
    metrics = {}
    geometry = {}
    for split, group in (("train_fit", source.fit), ("train_heldout", source.heldout)):
        models = {
            "constant": None, "mlp": source.models["mlp"],
            "old_gru": source.models["gru"], "near_only_gru": model,
        }
        timed = {
            name: countdown.dynamics(current, group, source.reference)
            for name, current in models.items()
        }
        scored = {
            name: stage_t.evaluate(current, group, source.reference)["aggregate"]
            for name, current in models.items()
        }
        metrics[split] = {
            name: {
                "prediction": scored[name],
                "countdown": _near_slope_metrics(timed[name]),
                "full_dynamics": timed[name],
            }
            for name in models
        }
        geometry[split] = _geometry_near_subset(
            group, source.models["gru"], model, source.reference)
    old_geometry = json.loads(
        (root / countdown.REPORT).read_text(encoding="utf-8")
    )["geometry_baseline"]
    for split in ("train_fit", "train_heldout"):
        previous = old_geometry[split]
        current = geometry[split]
        if current["eligible_near_frames"] != previous["near_eligible_frames"]:
            raise RuntimeError(f"Geometry eligibility drifted: {split}")
        previous_mae = previous["same_eligible_frame_comparison"]["gru"]["near_200ms"]["mae_ms"]
        if previous_mae is not None and abs(
            previous_mae - current["error"]["old_gru"]["mae_ms"]
        ) > .05:
            raise RuntimeError("Old GRU geometry-subset replay mismatch")
    heldout = {seq.anchor: seq for seq in source.heldout}
    if set(heldout) != set(plot_paths):
        raise RuntimeError("Unexpected Train-heldout partition / plot anchors")
    plots = {
        plot_paths[i]: _comparison_svg(seq, source.reference, source.models["gru"], model)
        for i, seq in heldout.items()
    }
    protected = {
        **source.source_sha256,
        "near_only_report": p.sha256(root / near.REPORT),
        "near_only_checkpoint": p.sha256(root / near.CHECKPOINT),
        "distance_audit": p.sha256(root / audit.REPORT),
        "old_countdown": p.sha256(root / countdown.REPORT),
    }
    if frozen_sha != protected or p._protect_old_best(config) != source.original_best_sha:
        raise RuntimeError("Protected Stage T artifact changed during near-only replay")

    report = {
        "format": FORMAT,
        "role": "Train20 fit / chart-disjoint heldout ONLY",
        "purpose": "Read-only saved checkpoint comparison: countdown vs near-MAE",
        "inputs": "causal 263D HUD only; event indices/times evaluator-only",
        "old_stage_a_gate": near_report["old_stage_a_gate"],
        "old_stage_a_gate_overridden": False,
        "motor_student_training_permitted": False,
        "no_training": True,
        "validation_evaluated": False, "final_evaluated": False,
        "automatically_adopted": False,
        "source_sha256": protected, "train_partition": source.partition,
        "metrics": metrics, "geometry_near_subset": geometry,
        "plots": [name.name for name in plot_paths.values()],
        "interpretation_guardrails": [
            "A decrease in near-MAE alone does not demonstrate real 10ms countdown.",
            "Countdown uses only successive frames of the SAME oracle event; switches scored separately.",
            "The eligibility of the simple-floor geometry subset is computed from HUD, never oracle.",
            "SVG is vertically clipped to 350ms ONLY FOR DISPLAY.",
            "No Motor/Fingering learning, original Stage A override, Validation/FINAL or new model training.",
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    for path, svg in plots.items():
        with path.open("x", encoding="utf-8") as out:
            out.write(svg)
    p._no_replace_json(report_path, report)
    return report
