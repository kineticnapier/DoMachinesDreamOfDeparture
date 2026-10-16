from __future__ import annotations

"""Exploratory Stage T: learn exact compiled-Autoplay timing from causal 263D HUD.

Train20 only, chart-disjoint FIT vs HELDOUT. The event timestamps and
visibility masks come solely from privileged evaluator labels. Models ingest
only 263D visible HUD and have no future-frame access. This module does NOT
alter frozen Stage A v2, enable Motor Student learning, use Validation/FINAL,
or select hyperparameters from heldout data.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import math
import random

import torch
from torch import nn
from torch.nn import functional as F

from dmdod.envs.n_key import encode_n_key_hud_real_chart_observation
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG
from dmdod.motor.capacity import CenterFirstNKeyTeacher
from dmdod.training.n_key import _make_env, _teacher_action

from . import prestige1 as p
from . import prestige1_alignment_audit as alignment
from . import prestige1_autoplay_export as exact_export
from . import prestige1_autoplay_timing as auto
from . import prestige1_transfer as transfer

FORMAT = "prestige1-stage-t-auto-exact-train20-chart-disjoint-v1"
REPORT = "stage_timing_autoplay_readout_train20.json"
MLP_CHECKPOINT = "stage_timing_autoplay_mlp.pt"
GRU_CHECKPOINT = "stage_timing_autoplay_gru.pt"
TIME_UNIT_S = 0.200
FAR_TIME_WEIGHT = 0.25  # fixed design choice, not a searched parameter


@dataclass(frozen=True)
class TimingSequence:
    anchor: int
    chart_sha256: str
    visible: torch.Tensor  # [T,263] predictor input ONLY
    remaining_s: torch.Tensor  # [T] exact oracle supervision ONLY
    near: torch.Tensor  # [T] evaluator binary "event within 200ms"
    mask: torch.Tensor  # [T] evaluator floor-visibility and future-availability
    event_indices: tuple[int, ...]  # audit metadata ONLY (not model input)

    def __post_init__(self):
        n = int(self.visible.shape[0])
        if (
            not 1 <= self.anchor <= 20
            or len(self.chart_sha256) != 64
            or self.visible.shape != (n, p.INPUT_DIM)
            or self.remaining_s.shape != (n,)
            or self.near.shape != (n,)
            or self.mask.shape != (n,)
            or len(self.event_indices) != n
            or n == 0
            or self.near.dtype != torch.bool
            or self.mask.dtype != torch.bool
            or not bool(torch.isfinite(self.visible).all())
            or not bool(torch.isfinite(self.remaining_s).all())
            or bool((self.remaining_s < 0).any())
        ):
            raise ValueError("Invalid causal Timing sequence / supervision")


class TimingReadout(nn.Module):
    """Independent one-frame MLP or strictly forward-only GRU, no motor head."""

    def __init__(self, hidden: int, *, recurrent: bool):
        super().__init__()
        self.recurrent = recurrent
        if recurrent:
            self.encoder = nn.GRU(p.INPUT_DIM, hidden, batch_first=True)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(p.INPUT_DIM, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
            )
        self.head = nn.Linear(hidden, 2)  # raw time; near-event logit

    def forward(self, visible: torch.Tensor) -> torch.Tensor:
        if visible.ndim != 2 or visible.shape[1] != p.INPUT_DIM:
            raise ValueError("Timing Student input must be causal [T,263]")
        if self.recurrent:
            encoded, _ = self.encoder(visible.unsqueeze(0))
            encoded = encoded.squeeze(0)
        else:
            encoded = self.encoder(visible)
        return self.head(encoded)


def decode(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("Expected [T,2] timing readout")
    # Unbounded nonnegative time prediction: no 200ms output ceiling.
    return F.softplus(raw[:, 0]) * TIME_UNIT_S, torch.sigmoid(raw[:, 1])


def timing_loss(raw: torch.Tensor, seq: TimingSequence, start: int, end: int) -> torch.Tensor:
    valid = seq.mask[start:end]
    if not bool(valid.any()):
        return raw.sum() * 0.
    predicted_s, event_p = decode(raw)
    expected = seq.remaining_s[start:end][valid]
    near = seq.near[start:end][valid]
    event_p = event_p[valid]
    # Exact, unrounded distance for BOTH near and far observable targets.
    # Far samples are downweighted but NEVER clipped or replaced by 200ms.
    per_time = F.smooth_l1_loss(
        predicted_s[valid] / TIME_UNIT_S,
        expected / TIME_UNIT_S, reduction="none",
    )
    weights = torch.where(near, 1., FAR_TIME_WEIGHT).to(per_time.dtype)
    time_loss = (per_time * weights).sum() / weights.sum()
    event_loss = F.binary_cross_entropy(event_p, near.to(event_p.dtype))
    return time_loss + event_loss


def capture_one(named, anchor: int, config: p.PrestigeConfig, *, device: torch.device):
    """Capture HUD from the original expert environment; oracle labels separate.

    Expert motor driving is only an observation-data collection regime.
    It is NOT the timing target: all timing labels come from the frozen
    Autoplay timeline. No hidden state or future target enters model inputs.
    """
    teacher = auto.AutoplayTimingTeacher.from_segment(named.segment)
    env = _make_env(
        named.segment, key_count=p.KEY_COUNT, control_dt_s=p.CONTROL_DT,
        physics_dt_s=p.PHYSICS_DT,
    )
    actuator = CenterFirstNKeyTeacher(p.KEY_COUNT)
    observation = env.reset()
    hud, remaining, near, masks, indices = [], [], [], [], []
    max_steps = math.ceil((named.segment.duration_s + 2.0) / p.CONTROL_DT) + 200
    last_time = -math.inf
    for _ in range(max_steps):
        features = tuple(encode_n_key_hud_real_chart_observation(observation, legacy_bpm_encoding=True))
        if len(features) != p.INPUT_DIM or not all(math.isfinite(x) for x in features):
            raise RuntimeError("Causal 263D HUD observation failed")
        now = float(env.privileged_episode_time_s())
        if not math.isfinite(now) or now <= last_time:
            raise RuntimeError("Expert capture clock failed monotonicity")
        last_time = now
        floor = env._floor_index_at_chart_time(named.segment.chart_time_from_episode(now))
        label = teacher.label_at(
            now, current_floor=floor,
            behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
            ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
        )
        hud.append(features)
        remaining.append(label.relative_time_s if label else 0.)
        near.append(bool(label and label.near_200ms))
        masks.append(bool(label and label.visible_floor_window))
        indices.append(label.event_index if label else -1)
        # Motor trajectories only create visible HUD states. No motor action
        # is included in training labels or passed to the Timing network.
        action = _teacher_action(env, actuator, observation, lead_s=config.lead_s)
        state = env.step(action)
        observation = state.observation
        if state.done:
            break
    else:
        raise RuntimeError("Train-only expert capture exceeded step budget")
    return TimingSequence(
        anchor, named.chart_sha256,
        torch.tensor(hud, dtype=torch.float32, device=device),
        torch.tensor(remaining, dtype=torch.float32, device=device),
        torch.tensor(near, dtype=torch.bool, device=device),
        torch.tensor(masks, dtype=torch.bool, device=device),
        tuple(indices),
    )


def _median(values: torch.Tensor) -> float:
    if values.numel() == 0:
        raise RuntimeError("No valid constant Timing baseline labels")
    return float(torch.quantile(values.float(), .5))


def constant_reference(fit: list[TimingSequence]) -> dict:
    positives = torch.cat([s.remaining_s[s.mask & s.near] for s in fit])
    flags = torch.cat([s.near[s.mask] for s in fit])
    if not positives.numel() or not flags.numel() or not bool((~flags).any()):
        raise RuntimeError("Fit partition lacks observable positive/negative frames")
    return {
        "time_s": _median(positives),  # strong near-event constant baseline
        "near_probability": float(flags.float().mean()),
        "fit_near_positive_count": int(positives.numel()),
        "fit_valid_frames": int(flags.numel()),
    }


def _chunks(sequences: list[TimingSequence], max_frames: int):
    chunks = [
        (seq, start, min(start + max_frames, seq.visible.shape[0]))
        for seq in sequences
        for start in range(0, seq.visible.shape[0], max_frames)
        if bool(seq.mask[start:min(start + max_frames, seq.visible.shape[0])].any())
    ]
    if not chunks:
        raise RuntimeError("No causally observable training chunks")
    return chunks


def train_one(model: TimingReadout, sequences: list[TimingSequence],
              *, updates: int, frames: int, lr: float, seed: int) -> tuple[list[float], int]:
    chunks = _chunks(sequences, frames)
    indices = list(range(len(chunks)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    history: list[float] = []
    # Fixed budget: at least one visit to every non-empty chunk. Hidden state
    # resets at chunk boundaries; heldout inference processes each anchor forward.
    for step in range(max(updates, len(chunks))):
        if step and step % len(indices) == 0:
            rng.shuffle(indices)
        seq, a, b = chunks[indices[step % len(indices)]]
        optimizer.zero_grad(set_to_none=True)
        raw = model(seq.visible[a:b])
        loss = timing_loss(raw, seq, a, b)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite Auto Timing loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        history.append(float(loss.detach()))
    return history, len(chunks)


@torch.no_grad()
def evaluate(model: TimingReadout | None, sequences: list[TimingSequence],
             reference: dict) -> dict:
    all_near_errors, all_visible_errors = [], []
    tp = tn = fp = fn = 0
    valid_count = masked_count = near_count = 0
    anchor_rows = []
    if model is not None:
        model.eval()
    for seq in sequences:
        if model is None:
            times = seq.remaining_s.new_full(seq.remaining_s.shape, reference["time_s"])
            chance = times.new_full(times.shape, reference["near_probability"])
        else:
            times, chance = decode(model(seq.visible))
        is_valid = seq.mask
        is_near = is_valid & seq.near
        is_far = is_valid & ~seq.near
        guess_near = chance >= .5
        tp_a = int((is_near & guess_near).sum())
        fn_a = int((is_near & ~guess_near).sum())
        fp_a = int((is_far & guess_near).sum())
        tn_a = int((is_far & ~guess_near).sum())
        tp += tp_a; fn += fn_a; fp += fp_a; tn += tn_a
        near_error = (times[is_near] - seq.remaining_s[is_near]).abs() * 1000.
        visible_error = (times[is_valid] - seq.remaining_s[is_valid]).abs() * 1000.
        all_near_errors.append(near_error.cpu())
        all_visible_errors.append(visible_error.cpu())
        near_n = int(is_near.sum())
        valid_n = int(is_valid.sum())
        valid_count += valid_n
        near_count += near_n
        masked_count += int((~is_valid).sum())
        anchor_rows.append({
            "anchor": seq.anchor,
            "chart_sha256": seq.chart_sha256,
            "frames": len(seq.remaining_s),
            "observable_frames": valid_n,
            "masked_frames": int((~is_valid).sum()),
            "near_positive_frames": near_n,
            "near_negative_frames": valid_n - near_n,
            "near_mae_ms": float(near_error.mean()) if near_n else None,
            "all_visible_mae_ms": float(visible_error.mean()) if valid_n else None,
            "tp": tp_a, "fn": fn_a, "fp": fp_a, "tn": tn_a,
        })
    if not valid_count or not near_count or not (valid_count - near_count):
        raise RuntimeError("Fail Closed: heldout Timing labels not sufficiently observable")
    near_errors = torch.cat(all_near_errors)
    visible_errors = torch.cat(all_visible_errors)
    recall = tp / (tp+fn)
    fpr = fp / (fp+tn)
    aggregate = {
        "frames_observable": valid_count, "frames_masked": masked_count,
        "near_positive_frames": near_count,
        "near_negative_frames": valid_count-near_count,
        "near_mae_ms": float(near_errors.mean()),
        "near_p50_abs_ms": float(torch.quantile(near_errors, .5)),
        "near_p95_abs_ms": float(torch.quantile(near_errors, .95)),
        "all_visible_mae_ms": float(visible_errors.mean()),
        "event_recall": recall, "false_positive_rate": fpr,
        "balanced_accuracy": (recall + 1 - fpr) / 2,
        "precision": tp / (tp+fp) if tp+fp else None,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "observable_anchors": sum(row["observable_frames"] > 0 for row in anchor_rows),
        "positive_anchors": sum(row["near_positive_frames"] > 0 for row in anchor_rows),
    }
    return {"aggregate": aggregate, "per_anchor": anchor_rows}


def exploratory_gate(heldout: dict, *, improvement: float = .20) -> dict:
    """A descriptive research threshold. NEVER unlock the original Stage A."""
    const = heldout["constant"]["aggregate"]
    mlp = heldout["mlp"]["aggregate"]
    gru = heldout["gru"]["aggregate"]
    best_simple = min(const["near_mae_ms"], mlp["near_mae_ms"])
    reasons = []
    if gru["positive_anchors"] < 3:
        reasons.append("fewer_than_three_positive_heldout_train_anchors")
    if gru["near_mae_ms"] > best_simple*(1-improvement):
        reasons.append("gru_not_20pct_better_than_constant_and_mlp")
    if gru["false_positive_rate"] > mlp["false_positive_rate"]+.05:
        reasons.append("event_fpr_regression")
    if gru["balanced_accuracy"] < .55 or gru["false_positive_rate"] >= .95:
        reasons.append("event_classifier_degenerate")
    return {
        "observability_evidence_passed": not reasons,
        "reasons": reasons,
        "criterion": "GRU near MAE >=20% better than both constant and MLP; positive >=3 Train heldout anchors; FPR <= MLP+0.05; balanced accuracy >=0.55; FPR <0.95",
        "original_stage_a_gate_overridden": False,
        "motor_student_training_permitted": False,
        "automatically_adopted": False,
    }


def _read_frozen_sources(config: p.PrestigeConfig, root: Path):
    original_a = root / p.STAGE_A_REPORT
    oracle_file = root / exact_export.OUTPUT
    for source in (original_a, oracle_file):
        if not source.is_file():
            raise FileNotFoundError("Frozen Auto Timing source missing: " + str(source))
    stage_a = json.loads(original_a.read_text(encoding="utf-8"))
    oracle = json.loads(oracle_file.read_text(encoding="utf-8"))
    if (
        stage_a.get("format") != p.VERSION or stage_a.get("stage") != "A"
        or stage_a.get("gate", {}).get("passed") is not False
        or stage_a.get("gate", {}).get("may_run_stage_c") is not False
        or stage_a.get("metadata", {}).get("config_sha256") != config.signature()
        or oracle.get("format") != exact_export.FORMAT
        or oracle.get("teacher_mode") != "autoplay_exact_compiled_event_timestamps"
        or oracle.get("role") != "Train20 only"
        or oracle.get("stage_a_original_gate") != stage_a["gate"]
        or oracle.get("config_sha256") != config.signature()
        or oracle.get("total_targets") != 850
        or oracle.get("total_distinct_onsets") != 850
        or oracle.get("no_training") is not True
        or oracle.get("student_inference_receives_privileged_timestamps") is not False
        or oracle.get("validation_evaluated") is not False
        or oracle.get("final_evaluated") is not False
        or oracle.get("protected_inputs_unchanged") is not True
        or len(oracle.get("per_anchor", [])) != 20
    ):
        raise RuntimeError("Invalid frozen Stage A FAIL / exact Auto oracle provenance")
    protected = oracle.get("protected_input_sha256", {})
    protected_paths = {
        "stage_a_v2": original_a,
        "stage_c_v30_finger_phase_annealing": root / "stage_c_v30_finger_phase_annealing.json",
        "stage_c_v30_safe_teacher_plans": root / "stage_c_v30_safe_teacher_plans.json",
    }
    if protected != {key: p.sha256(file) for key, file in protected_paths.items()}:
        raise RuntimeError("Auto oracle dependencies changed since freezing")
    return stage_a, oracle, {
        "stage_a_v2": p.sha256(original_a),
        "exact_autoplay_oracle": p.sha256(oracle_file),
        **{key: p.sha256(path) for key, path in protected_paths.items() if key != "stage_a_v2"},
    }


def run_autoplay_timing_readout(config: p.PrestigeConfig) -> dict:
    root = transfer.output_dir(config)
    targets = [root / REPORT, root / MLP_CHECKPOINT, root / GRU_CHECKPOINT]
    if any(path.exists() for path in targets):
        raise FileExistsError("Refuse to overwrite experimental Timing outputs")
    original_best_sha = p._protect_old_best(config)
    stage_a, oracle, frozen_sha = _read_frozen_sources(config, root)
    selected, manifest = alignment._train_manifest(config)
    if (
        manifest != oracle["train_manifest"]
        or manifest != stage_a["metadata"]["dataset_split"]["train"]
    ):
        raise RuntimeError("Train20 dataset manifest does not match frozen oracle")
    sequences: list[TimingSequence] = []
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    for i, (named, expected) in enumerate(zip(selected, oracle["per_anchor"]), 1):
        reconstructed = auto.AutoplayTimingTeacher.from_segment(named.segment)
        expected_timeline = [
            {
                "event_index": ev.event_index,
                "target_ordinals": list(ev.target_ordinals),
                "floor_indices": list(ev.floor_indices),
                "episode_time_s": ev.episode_time_s,
                "chart_time_s": ev.chart_time_s,
                "chord_size": ev.chord_size,
            }
            for ev in reconstructed.events
        ]
        if (
            expected["anchor"] != i
            or expected["chart_sha256"] != named.chart_sha256
            or expected["chart"] != named.chart_name
            or expected["timeline"] != expected_timeline
        ):
            raise RuntimeError("Autoplay exact event timeline mismatch on Train anchor " + str(i))
        seq = capture_one(named, i, config, device=device)
        sequences.append(seq)
        print(
            f"Auto Timing capture Train {i:02d}/20: "
            f"frames={seq.visible.shape[0]} observable={int(seq.mask.sum())} "
            f"near={int((seq.mask & seq.near).sum())}",
            flush=True,
        )
    # reuse the frozen chart-disjoint split algorithm, WITHOUT Validation
    # sequences and WITHOUT changing the original Stage A v2 training code.
    fit, heldout, partition = p.split_stage_a_train(
        sequences, {"train": oracle["train_manifest"]}, seed=config.seed)
    reference = constant_reference(fit)
    torch.manual_seed(config.seed)
    models = {
        "mlp": TimingReadout(config.temporal_hidden, recurrent=False).to(device),
    }
    torch.manual_seed(config.seed)
    models["gru"] = TimingReadout(config.temporal_hidden, recurrent=True).to(device)
    history, coverage = {}, {}
    for model_name, model in models.items():
        losses, chunks = train_one(
            model, fit, updates=config.stage_a_updates,
            frames=config.batch_frames, lr=config.learning_rate, seed=config.seed)
        history[model_name] = losses
        coverage[model_name] = {
            "nonempty_fit_chunks": chunks,
            "updates": len(losses),
            "at_least_one_pass": len(losses) >= chunks,
            "hidden_state_reset_per_fit_chunk": True,
        }
    def score(group):
        return {
            "constant": evaluate(None, group, reference),
            "mlp": evaluate(models["mlp"], group, reference),
            "gru": evaluate(models["gru"], group, reference),
        }
    fit_scores = score(fit)
    heldout_scores = score(heldout)
    gate = exploratory_gate(heldout_scores)
    if frozen_sha != _read_frozen_sources(config, root)[2]:
        raise RuntimeError("Protected Timing sources changed during experiment")
    if original_best_sha != p._protect_old_best(config):
        raise RuntimeError("Frozen original Best changed during experiment")
    report = {
        "format": FORMAT,
        "role": "Train20 fit / chart-disjoint Train heldout ONLY",
        "teacher": "compiled Autoplay exact event times, not physical KeyDown",
        "student_input": "causal 263D visible HUD only",
        "predicts_fingers": False,
        "output_time_clipped": False,
        "near_event_window_s": .2,
        "far_timing_loss_weight": FAR_TIME_WEIGHT,
        "student_motor_training_performed": False,
        "timing_readouts_trained_exploratorily": True,
        "original_stage_a_gate": stage_a["gate"],
        "original_stage_a_gate_overridden": False,
        "validation_evaluated": False,
        "final_evaluated": False,
        "validation_used_for_selection": False,
        "motor_student_training_permitted": False,
        "automatically_adopted": False,
        "config_sha256": config.signature(),
        "train_manifest": manifest,
        "train_partition": partition,
        "oracle_source_sha256": frozen_sha,
        "reference": reference,
        "training_coverage": coverage,
        "training_loss": history,
        "model_parameter_counts": {
            name: sum(parameter.numel() for parameter in model.parameters())
            for name, model in models.items()
        },
        "train_fit": fit_scores,
        "train_heldout": heldout_scores,
        "research_gate": gate,
        "limitations": [
            "Training observation trajectories are driven by a privileged center-first expert; policy-induced covariate shift remains untested.",
            "Even exact target timestamps are not proof that HUD makes the next note observable.",
            "The model predicts one next event; sub-10ms multiple events in one control step require additional future design.",
            "No physical Motor rollout or Fingering learning was performed.",
            "This experiment cannot unlock the frozen Stage A gate.",
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    saved = {}
    for name, model in models.items():
        path = root / (MLP_CHECKPOINT if name == "mlp" else GRU_CHECKPOINT)
        with path.open("xb") as f:
            torch.save({
                "format": FORMAT, "model_name": name,
                "model_state_dict": model.state_dict(),
                "config_sha256": config.signature(),
                "train_partition": partition,
                "oracle_sha256": frozen_sha["exact_autoplay_oracle"],
                "research_only": True, "motor_adoption_permitted": False,
            }, f)
        saved[name] = p.sha256(path)
    report["experimental_checkpoint_sha256"] = saved
    p._no_replace_json(root / REPORT, report)
    return report
