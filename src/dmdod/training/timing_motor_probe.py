from __future__ import annotations

"""Read-only timing-representation / physical-motor separation probe.

Student actions consume exactly the normal 263D visible observation.
Privileged target times and motor event timestamps are labels/diagnostics only.
Readouts are separate linear regressors trained on Train anchors, never saved
into the Student or used to affect its closed-loop actions.
"""

import csv
from dataclasses import asdict, dataclass
import gzip
import json
import math
from pathlib import Path

import torch
from torch import nn

from dmdod.envs.n_key import (
    DiagnosticHudNKeyRealChartMotorEnv,
    NKeyOverloadTrace,
    encode_n_key_hud_real_chart_observation,
)
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG
from dmdod.motor.keyboard import KeyEvent
from dmdod.motor.n_key import NKeyAction
from dmdod.training.recovery_probe import (
    format_recovery_probe,
    run_teacher_takeover_probe,
)


@dataclass
class ProbeRollout:
    role: str
    anchor_index: int
    chart_name: str
    observations: torch.Tensor
    hidden_states: torch.Tensor
    actions: torch.Tensor
    remaining_s: torch.Tensor
    target_indices: tuple[int | None, ...]
    frame_times_s: tuple[float, ...]
    trace: NKeyOverloadTrace
    motor_events: list[dict]
    stats: object


def _environment(prepared, named):
    return DiagnosticHudNKeyRealChartMotorEnv(
        named.segment,
        key_count=prepared.key_count,
        capture_overload_trace=True,
        control_dt_s=prepared.control_dt_s,
        physics_dt_s=prepared.physics_dt_s,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )


def pair_command_events(
    commands: list[tuple[float, tuple[float, ...]]],
    physical_events: list[tuple[float, str, str]],
    keys: tuple[str, ...],
    *,
    press_threshold: float = 0.25,
    release_threshold: float = -0.45,
) -> list[dict]:
    """Associate threshold crossings with same-key physical edges.

    This is an operational latency proxy, not an inference of intended presses.
    Negative latencies are never silently clamped; unmatched events remain null.
    """
    pending_press: dict[str, float | None] = {key: None for key in keys}
    pending_release: dict[str, float | None] = {key: None for key in keys}
    events = []
    prior = [0.0] * len(keys)
    merged = []
    for time_s, values in commands:
        if len(values) != len(keys):
            raise ValueError("command/key width mismatch")
        for index, (key, value) in enumerate(zip(keys, values)):
            if value >= press_threshold and prior[index] < press_threshold:
                merged.append((time_s, 0, key, "press_command"))
            if value <= release_threshold and prior[index] > release_threshold:
                merged.append((time_s, 0, key, "release_command"))
        prior = list(values)
    for time_s, key, event in physical_events:
        if key not in pending_press or event not in ("down", "up"):
            raise ValueError("invalid physical event")
        merged.append((time_s, 1, key, "keydown" if event == "down" else "keyup"))
    # Commands are issued at control-frame starts; physical edges occur later.
    merged.sort(key=lambda x: (x[0], x[1]))
    for time_s, _, key, kind in merged:
        latency_ms = None
        origin_s = None
        if kind == "press_command":
            pending_press[key] = time_s
        elif kind == "release_command":
            pending_release[key] = time_s
        elif kind == "keydown":
            origin_s = pending_press[key]
            if origin_s is not None:
                latency_ms = 1000.0 * (time_s - origin_s)
                pending_press[key] = None
        elif kind == "keyup":
            origin_s = pending_release[key]
            if origin_s is not None:
                latency_ms = 1000.0 * (time_s - origin_s)
                pending_release[key] = None
        events.append({
            "time_s": float(time_s), "key": key, "kind": kind,
            "command_origin_s": origin_s, "latency_ms": latency_ms,
        })
    return events


def collect_student_rollout(model, prepared, named, *, role: str, index: int) -> ProbeRollout:
    """Identical action/termination path to normal _evaluate_continuous."""
    env = _environment(prepared, named)
    obs = env.reset()
    state = model.initial_state(prepared.device)
    observations, hidden_states, actions, remaining, indices, times = [], [], [], [], [], []
    commands: list[tuple[float, tuple[float, ...]]] = []
    edges: list[tuple[float, str, str]] = []
    max_steps = int((named.segment.duration_s + 2.0) / prepared.control_dt_s) + 200
    model.eval()
    with torch.no_grad():
        for _ in range(max_steps):
            t = float(env.privileged_episode_time_s())
            next_idx = env._next_target_index()  # evaluator-only; NEVER model input
            target = env.privileged_next_target()
            truth = None if target is None else float(target.episode_time_s) - t
            visible = encode_n_key_hud_real_chart_observation(obs)
            if len(visible) != prepared.input_dim:
                raise RuntimeError("Student visible input dimension changed")
            x = torch.tensor(visible, dtype=torch.float32, device=prepared.device)
            mean, _, _, state = model.forward_step(x, state)
            action_tuple = tuple(float(v.item()) for v in torch.tanh(mean))
            if len(action_tuple) != prepared.key_count:
                raise RuntimeError("Student action width mismatch")
            observations.append(x.detach().cpu())
            hidden_states.append(state.detach().cpu().clone())
            actions.append(torch.tensor(action_tuple, dtype=torch.float32))
            remaining.append(float("nan") if truth is None else truth)
            indices.append(next_idx)
            times.append(t)
            commands.append((t, action_tuple))
            result = env.step(NKeyAction(action_tuple))
            obs = result.observation
            transition = env.motor.last_transition
            if transition is None:
                raise RuntimeError("missing motor transition")
            edges.extend((
                float(edge.time_s), str(edge.key), edge.event.value
            ) for edge in transition.evaluator_events)
            if result.done:
                break
        else:
            raise RuntimeError("Student probe exceeded normal Validation step budget")
    trace = env.overload_trace
    if trace is None:
        raise RuntimeError("Student episode has no terminal trace")
    if trace.termination.hits != env.stats.hits:
        raise RuntimeError("Student trace does not match scored hits")
    motor_events = pair_command_events(commands, edges, tuple(env.key_names))
    # Join scoring target/time to physically recorded key-downs, without
    # claiming that every TooEarly was unintended.
    physical_downs = {(e.time_s, e.key): e for e in trace.physical_keydowns}
    early_events = {(e.time_s, e.key) for e in trace.too_early_events}
    for event in motor_events:
        if event["kind"] != "keydown":
            continue
        physical = physical_downs.get((event["time_s"], event["key"]))
        event["target_index"] = physical.target_index if physical else None
        event["target_error_ms"] = (
            (event["time_s"] - physical.target_time_s) * 1000.0
            if physical is not None and physical.target_time_s is not None else None
        )
        event["too_early"] = (event["time_s"], event["key"]) in early_events
        event["unmatched_target"] = physical is None or physical.target_index is None
        event["scored_by_evaluator"] = physical is not None
    return ProbeRollout(
        role=role, anchor_index=index, chart_name=named.chart_name,
        observations=torch.stack(observations),
        hidden_states=torch.stack(hidden_states),
        actions=torch.stack(actions),
        remaining_s=torch.tensor(remaining),
        target_indices=tuple(indices), frame_times_s=tuple(times),
        trace=trace, motor_events=motor_events, stats=env.stats,
    )


class LinearTimeReadout(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.linear = nn.Linear(width, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


@dataclass
class TrainedReadout:
    model: LinearTimeReadout
    input_mean: torch.Tensor
    input_std: torch.Tensor
    target_mean: torch.Tensor
    target_std: torch.Tensor


def train_readout(
    features: torch.Tensor, labels: torch.Tensor,
    *, device: torch.device, steps: int = 250,
    batch_size: int = 256, seed: int = 2401,
) -> TrainedReadout:
    """Only caller-supplied Train rows may be used for fitting/normalization."""
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("readout feature/label shape mismatch")
    if steps < 1 or batch_size < 1:
        raise ValueError("readout steps/batch_size must be positive")
    usable = torch.isfinite(labels) & torch.isfinite(features).all(dim=1)
    x, y = features[usable].float(), labels[usable].float()
    if x.shape[0] < 2:
        raise ValueError("readout requires at least two Train frames")
    mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-4)
    y_mean, y_std = y.mean(), y.std(unbiased=False).clamp_min(1e-4)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    # fork_rng prevents readout initialization changing global Student RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        head = LinearTimeReadout(x.shape[1]).to(device)
    # The diagnostic readout needs a higher learning rate to fit linear labels.
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(steps):
        ids = torch.randint(x.shape[0], (min(batch_size, x.shape[0]),), generator=generator)
        xb = ((x[ids] - mean) / std).to(device)
        yb = ((y[ids] - y_mean) / y_std).to(device)
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.smooth_l1_loss(head(xb), yb)
        loss.backward()
        opt.step()
    head.eval()
    return TrainedReadout(head, mean, std, y_mean, y_std)


@torch.no_grad()
def predict_readout(trained: TrainedReadout, features: torch.Tensor, device: torch.device) -> torch.Tensor:
    # Do not transfer the entire 20-anchor state matrix to small GPUs.
    out = []
    for batch in features.split(512):
        x = ((batch.float() - trained.input_mean) / trained.input_std).to(device)
        out.append((trained.model(x).cpu() * trained.target_std + trained.target_mean))
    return torch.cat(out)


def error_distribution(errors_ms: list[float]) -> dict:
    if not errors_ms:
        return {"n": 0, "mae_ms": None, "p50_ms": None, "p90_ms": None, "p99_ms": None}
    a = torch.tensor([abs(float(x)) for x in errors_ms], dtype=torch.float64)
    return {
        "n": int(a.numel()), "mae_ms": float(a.mean()),
        "p50_ms": float(torch.quantile(a, .5)),
        "p90_ms": float(torch.quantile(a, .9)),
        "p99_ms": float(torch.quantile(a, .99)),
    }


def evaluate_time_readouts(
    rollout: ProbeRollout, state_readout: TrainedReadout,
    observation_readout: TrainedReadout, device: torch.device,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    hidden_predictions = predict_readout(state_readout, rollout.hidden_states, device)
    visible_predictions = predict_readout(observation_readout, rollout.observations, device)
    finite = torch.isfinite(rollout.remaining_s)
    near = finite & (rollout.remaining_s >= 0.0) & (rollout.remaining_s <= 0.150)
    def report(pred):
        error = ((pred - rollout.remaining_s) * 1000.0)
        return {
            "all": error_distribution(error[finite].tolist()),
            "prehit_0_150ms": error_distribution(error[near].tolist()),
        }
    return {
        "hidden": report(hidden_predictions),
        "visible_only": report(visible_predictions),
        "valid_frames": int(finite.sum()),
        "prehit_frames": int(near.sum()),
    }, hidden_predictions, visible_predictions


def save_frame_data(rollout: ProbeRollout, path: Path, hidden_pred=None, visible_pred=None):
    """Compressed JSON Lines retains full 263D, full RNN state and action."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=4) as output:
        for i in range(len(rollout.frame_times_s)):
            record = {
                "t_s": rollout.frame_times_s[i],
                "next_target_index": rollout.target_indices[i],
                "target_remaining_s": (
                    float(rollout.remaining_s[i]) if math.isfinite(float(rollout.remaining_s[i]))
                    else None
                ),
                "visible_263d": rollout.observations[i].tolist(),
                "student_rnn_state_after_step": rollout.hidden_states[i].tolist(),
                "continuous_action": rollout.actions[i].tolist(),
                "hidden_readout_s": None if hidden_pred is None else float(hidden_pred[i]),
                "visible_readout_s": None if visible_pred is None else float(visible_pred[i]),
            }
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def motor_statistics(rollout: ProbeRollout) -> dict:
    events = rollout.motor_events
    down = [e for e in events if e["kind"] == "keydown" and e.get("scored_by_evaluator")]
    errors = [e["target_error_ms"] for e in down if e["target_error_ms"] is not None]
    press_delays = [e["latency_ms"] for e in events if e["kind"] == "keydown" and e["latency_ms"] is not None]
    release_delays = [e["latency_ms"] for e in events if e["kind"] == "keyup" and e["latency_ms"] is not None]
    trace = rollout.trace
    return {
        "hits": int(rollout.stats.hits),
        "targets": int(rollout.stats.targets),
        "misses": int(rollout.stats.misses),
        "too_early": int(rollout.stats.too_early_presses),
        "overloaded": bool(rollout.stats.overloaded),
        "keydowns": int(trace.termination.keydowns),
        "terminal_s": trace.termination.time_s,
        "terminal_reason": trace.termination.reason,
        "keydown_target_error": error_distribution(errors),
        "press_command_to_down": error_distribution(press_delays),
        "release_command_to_up": error_distribution(release_delays),
        "too_early_keydowns": sum(bool(e.get("too_early")) for e in down),
        "unmatched_or_unscored_keydowns": sum(bool(e.get("unmatched_target")) for e in events if e["kind"] == "keydown"),
        "matched_press_count": len(press_delays),
        "matched_release_count": len(release_delays),
    }


def _write_event_csv(path: Path, rollout: ProbeRollout):
    fields = ["role", "anchor", "chart", "time_s", "key", "kind",
              "command_origin_s", "latency_ms", "target_index",
              "target_error_ms", "too_early", "unmatched_target", "scored_by_evaluator"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rollout.motor_events:
            writer.writerow({"role": rollout.role, "anchor": rollout.anchor_index,
                             "chart": rollout.chart_name, **row})


def _joint_train_rows(rollouts: list[ProbeRollout], attr: str) -> tuple[torch.Tensor, torch.Tensor]:
    data, target = [], []
    for rollout in rollouts:
        valid = torch.isfinite(rollout.remaining_s)
        if valid.any():
            data.append(getattr(rollout, attr)[valid])
            target.append(rollout.remaining_s[valid])
    if not data:
        raise ValueError("no labelled Train frames")
    return torch.cat(data), torch.cat(target)


def _aggregate_readout_stats(reports: list[dict], label: str) -> dict:
    # Frame-weighted aggregate MAE from per-anchor MAEs (same fixed Validation).
    def merged(section):
        parts = [r["timing"][label][section] for r in reports]
        n = sum(p["n"] for p in parts)
        return {"n": n, "mae_ms": (
            sum(p["n"] * p["mae_ms"] for p in parts if p["n"]) / n if n else None
        )}
    return {"all": merged("all"), "prehit_0_150ms": merged("prehit_0_150ms")}


def _check_student_unchanged(model, original: dict[str, torch.Tensor]):
    now = model.state_dict()
    if set(now) != set(original) or any(
        not torch.equal(now[key].detach().cpu(), old) for key, old in original.items()
    ):
        raise RuntimeError("Student checkpoint state changed during diagnostic probe")


def run_timing_motor_probe(
    prepared,
    *,
    output_dir: Path,
    readout_steps: int = 250,
    with_teacher_takeover: bool = True,
):
    """Train only two small readouts, evaluate Student on fixed Train/Validation."""
    model = prepared.model
    if prepared.key_count != 8 or prepared.input_dim != 263:
        raise ValueError("probe requires 8-key 263D Student")
    if len(prepared.anchors) != 20 or len(prepared.validation) != 20:
        raise ValueError("probe requires fixed 20 Train and 20 Validation anchors")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"probe output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    try:
        print("=== Timing / Motor separation: Student fixed ===", flush=True)
        train_rollouts = []
        for index, named in enumerate(prepared.anchors, 1):
            rollout = collect_student_rollout(
                model, prepared, named, role="train", index=index
            )
            train_rollouts.append(rollout)
            save_frame_data(
                rollout, output_dir / "frames" / f"train_{index:02d}.jsonl.gz"
            )
            _write_event_csv(
                output_dir / f"motor_train_{index:02d}.csv", rollout
            )
            print(f"train #{index:02d} {named.chart_name} H={rollout.stats.hits} "
                  f"Overload={rollout.stats.overloaded}", flush=True)
        obs_x, obs_y = _joint_train_rows(train_rollouts, "observations")
        state_x, state_y = _joint_train_rows(train_rollouts, "hidden_states")
        if not torch.equal(obs_y, state_y):
            raise RuntimeError("Train timing labels differ across representations")
        print(f"readout Train frames={len(obs_y)} hidden_dim={state_x.shape[1]}", flush=True)
        readout_hidden = train_readout(
            state_x, state_y, device=prepared.device, steps=readout_steps,
            seed=2401
        )
        readout_visible = train_readout(
            obs_x, obs_y, device=prepared.device, steps=readout_steps,
            seed=2402
        )
        del obs_x, state_x, obs_y, state_y, train_rollouts

        reports = []
        motor_errors = []
        for index, named in enumerate(prepared.validation, 1):
            rollout = collect_student_rollout(
                model, prepared, named, role="validation", index=index
            )
            timing, pred_state, pred_visible = evaluate_time_readouts(
                rollout, readout_hidden, readout_visible, prepared.device
            )
            motor = motor_statistics(rollout)
            report = {"anchor": index, "chart": named.chart_name,
                      "chart_sha256": named.chart_sha256,
                      "timing": timing, "motor": motor}
            reports.append(report)
            motor_errors.extend(
                e["target_error_ms"] for e in rollout.motor_events
                if e["kind"] == "keydown" and e.get("scored_by_evaluator")
                and e.get("target_error_ms") is not None
            )
            save_frame_data(
                rollout, output_dir / "frames" / f"validation_{index:02d}.jsonl.gz",
                pred_state, pred_visible
            )
            _write_event_csv(
                output_dir / f"motor_validation_{index:02d}.csv", rollout
            )
            print(f"validation #{index:02d} {named.chart_name} "
                  f"H={motor['hits']}/{motor['targets']} "
                  f"Early={motor['too_early']} Overload={motor['overloaded']} "
                  f"timing MAE hidden={timing['hidden']['all']['mae_ms']} "
                  f"visible={timing['visible_only']['all']['mae_ms']}", flush=True)
            if index == 9 and with_teacher_takeover:
                result = run_teacher_takeover_probe(
                    model, named, alpha=0.0, lead_s=prepared.lead_s,
                    control_dt_s=prepared.control_dt_s,
                    physics_dt_s=prepared.physics_dt_s, device=prepared.device,
                    expected_trace=rollout.trace,
                )
                report["teacher_takeover"] = asdict(result)
                for line in format_recovery_probe(result):
                    print(line, flush=True)
        overall = {
            "source_checkpoint": str(prepared.source_checkpoint),
            "readout": {
                "hidden": _aggregate_readout_stats(reports, "hidden"),
                "visible_only": _aggregate_readout_stats(reports, "visible_only"),
                "fit_only": "Train anchors",
                "readout_steps": readout_steps,
            },
            "motor": {
                "keydown_target_error": error_distribution(motor_errors),
                "safe_anchors": sum(not row["motor"]["overloaded"] for row in reports),
                "hits": sum(row["motor"]["hits"] for row in reports),
                "misses": sum(row["motor"]["misses"] for row in reports),
                "too_early": sum(row["motor"]["too_early"] for row in reports),
                "keydowns": sum(row["motor"]["keydowns"] for row in reports),
            },
            "validation": reports,
            "notes": [
                "Student consumed only the original visible 263D input.",
                "Target labels and event timing are evaluator-only diagnostics.",
                "Physical press/release latencies are threshold-crossing proxies.",
                "TooEarly or unmatched KeyDown does not prove intent.",
                "The linear readout measures decodability, not Student use of timing.",
                "Teacher takeover uses privileged targets; it is not a fair input-matched policy.",
                "No Student training, checkpoint writes, or FINAL holdout evaluations.",
            ],
        }
        (output_dir / "summary.json").write_text(
            json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (output_dir / "per_anchor.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "anchor", "chart", "hits", "misses", "early",
                "keydowns", "overload", "termination_s", "reason",
                "hidden_mae_ms", "visible_mae_ms",
                "hidden_prehit_mae_ms", "visible_prehit_mae_ms",
                "keydown_error_mae_ms", "press_latency_mae_ms", "release_latency_mae_ms",
            ])
            for r in reports:
                t, m = r["timing"], r["motor"]
                writer.writerow([
                    r["anchor"], r["chart"], m["hits"], m["misses"],
                    m["too_early"], m["keydowns"], int(m["overloaded"]),
                    m["terminal_s"], m["terminal_reason"],
                    t["hidden"]["all"]["mae_ms"], t["visible_only"]["all"]["mae_ms"],
                    t["hidden"]["prehit_0_150ms"]["mae_ms"],
                    t["visible_only"]["prehit_0_150ms"]["mae_ms"],
                    m["keydown_target_error"]["mae_ms"],
                    m["press_command_to_down"]["mae_ms"],
                    m["release_command_to_up"]["mae_ms"],
                ])
        print(f"=== probe complete: {output_dir}/summary.json ===", flush=True)
        print(json.dumps({"readout": overall["readout"], "motor": overall["motor"]},
                         ensure_ascii=False, indent=2), flush=True)
        return overall
    finally:
        _check_student_unchanged(model, original)
