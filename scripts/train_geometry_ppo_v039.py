from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from . import train_geometry_ppo_v038 as v038
except ImportError:  # direct script execution
    import train_geometry_ppo_v038 as v038

from dmdod.motion_geometry_env import MotionGeometryEnv
from dmdod.predictive_recurrent_policy import PredictiveRecurrentActorCritic
from dmdod.privileged_teacher import PrivilegedLeadTeacher, calibrate_single_press_lead
from dmdod.rhythm_env import RewardConfig, make_regular_targets
from dmdod.toy_policy import MOTION_GEOMETRY_INPUT_DIM, observation_tensor


v036 = v038.v036
v035 = v038.v035
base = v038.base

# v0.3.9: the P3 failure is treated as a policy-initialization/exploration
# problem, not a chart-memory problem. A privileged one-note teacher generates
# speed-dependent motor demonstrations. The student is behavior-cloned using
# only the same 14-D visible observation used by PPO, then PPO resumes normally.
IMITATION_EPISODES = 192
IMITATION_EPOCHS = 12
IMITATION_LR = 3e-4
IMITATION_ACTIVE_WEIGHT = 2.5
IMITATION_BPM_MIN = 145.0
IMITATION_BPM_MAX = 240.0
IMITATION_JITTER_MS = 65.0
IMITATION_SEED = 3901

_ORIGINAL_V038_SAVE_CHECKPOINT = v038.save_checkpoint
_IMITATION_STATE: dict[str, object] = {}


@dataclass(frozen=True)
class Demonstration:
    observations: object
    actions: object


def _make_teacher_env(*, bpm: float, start_s: float, control_dt: float, seed: int):
    target = make_regular_targets(bpm=bpm, count=1, start_s=start_s, pattern="left")[0]
    env = MotionGeometryEnv(
        [target],
        bpm=bpm,
        same_hand=True,
        control_dt_s=control_dt,
        vision_config=base.core.clean_vision_config(),
        perception_seed=seed,
        reward_config=RewardConfig(too_early_penalty=v035.TOO_EARLY_PENALTY),
    )
    return env, target


def _demonstration_bpm(index: int, rng: random.Random) -> float:
    # Force both edges to appear frequently, with the other half sampled
    # continuously across the range so a fixed-angle lookup cannot fit the set.
    phase = index % 4
    if phase == 0:
        return IMITATION_BPM_MIN
    if phase == 1:
        return IMITATION_BPM_MAX
    return rng.uniform(IMITATION_BPM_MIN, IMITATION_BPM_MAX)


def collect_demonstrations(
    teacher: PrivilegedLeadTeacher,
    *,
    episodes: int,
    control_dt: float,
    device,
    seed: int = IMITATION_SEED,
) -> list[Demonstration]:
    torch = base.core.torch
    rng = random.Random(seed)
    result: list[Demonstration] = []

    for episode in range(episodes):
        bpm = _demonstration_bpm(episode, rng)
        start_s = max(
            0.050,
            base.core.curriculum_start_s(1)
            + rng.uniform(-IMITATION_JITTER_MS, IMITATION_JITTER_MS) / 1000.0,
        )
        env, target = _make_teacher_env(
            bpm=bpm,
            start_s=start_s,
            control_dt=control_dt,
            seed=seed + episode * 1009,
        )
        observation = env.reset()
        xs = []
        ys = []

        while True:
            now_s = env.motor.diagnostics().time_s
            action = teacher.action(
                now_s=now_s,
                target_time_s=target.time_s,
                motor=observation.motor,
            )
            xs.append(observation_tensor(observation, device).detach())
            ys.append(torch.tensor([action.left, action.right], dtype=torch.float32, device=device))
            transition = env.step(action)
            observation = transition.observation
            if transition.done:
                break

        result.append(Demonstration(torch.stack(xs), torch.stack(ys)))

    return result


def imitation_pretrain(
    model: PredictiveRecurrentActorCritic,
    demonstrations: list[Demonstration],
    *,
    epochs: int,
    lr: float,
    active_weight: float = IMITATION_ACTIVE_WEIGHT,
) -> float:
    """Behavior-clone teacher actions through the recurrent student policy."""

    torch = base.core.torch
    nn = base.core.nn
    parameters = [
        *model.input_layer.parameters(),
        *model.gru.parameters(),
        *model.post.parameters(),
        *model.actor_mean.parameters(),
    ]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    rng = random.Random(IMITATION_SEED + 1)
    final_loss = 0.0
    model.train()

    for _ in range(max(0, epochs)):
        order = list(range(len(demonstrations)))
        rng.shuffle(order)
        total_loss = 0.0
        total_steps = 0

        for index in order:
            demo = demonstrations[index]
            observations = demo.observations
            targets = demo.actions
            state = model.initial_state(observations.device)
            predicted = []
            for x in observations:
                mean, _, _, state = model.forward_step(x, state)
                predicted.append(torch.tanh(mean))
            prediction = torch.stack(predicted)

            per_step = ((prediction - targets) ** 2).mean(dim=1)
            # Command-onset samples matter more than long neutral pre-rolls.
            weights = 1.0 + active_weight * targets[:, 0].abs()
            loss = (per_step * weights).sum() / weights.sum().clamp_min(1.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()

            total_loss += float(loss.item()) * int(observations.shape[0])
            total_steps += int(observations.shape[0])

        final_loss = total_loss / max(1, total_steps)

    return final_loss


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def imitation_probe(
    model: PredictiveRecurrentActorCritic,
    *,
    control_dt: float,
    device,
) -> dict[str, float | None]:
    """Quick deterministic P3 probe after BC, before PPO changes the policy."""

    edge_errors: dict[float, list[float]] = {
        IMITATION_BPM_MIN: [],
        IMITATION_BPM_MAX: [],
    }
    all_errors: list[float] = []
    was_training = model.training
    model.eval()

    with base.core.torch.no_grad():
        for bpm in (IMITATION_BPM_MIN, IMITATION_BPM_MAX):
            for i in range(9):
                offset = -IMITATION_JITTER_MS + 2.0 * IMITATION_JITTER_MS * i / 8.0
                start_s = max(0.050, base.core.curriculum_start_s(1) + offset / 1000.0)
                env, _ = _make_teacher_env(
                    bpm=bpm,
                    start_s=start_s,
                    control_dt=control_dt,
                    seed=IMITATION_SEED + int(bpm * 10) + i,
                )
                observation = env.reset()
                state = model.initial_state(device)
                while True:
                    action, state = model.deterministic_action(
                        observation_tensor(observation, device), state
                    )
                    transition = env.step(action)
                    observation = transition.observation
                    if transition.done:
                        break
                for error in env.timing_errors_ms:
                    edge_errors[bpm].append(float(error))
                    all_errors.append(float(error))

    if was_training:
        model.train()

    return {
        "mean_error_ms": _mean(all_errors),
        "edge_145_error_ms": _mean(edge_errors[IMITATION_BPM_MIN]),
        "edge_240_error_ms": _mean(edge_errors[IMITATION_BPM_MAX]),
        "hit_fraction": len(all_errors) / 18.0,
    }


def _fmt_optional(value: object) -> str:
    if value is None:
        return "--"
    return f"{float(value):+.1f}"


def _optimizer_lr(saved: dict[str, object]) -> float:
    optimizer = saved.get("optimizer")
    if isinstance(optimizer, dict):
        groups = optimizer.get("param_groups")
        if isinstance(groups, list) and groups:
            lr = groups[0].get("lr")
            if lr is not None:
                return float(lr)
    return 1e-4


def _upgrade_resume_if_requested() -> None:
    global _IMITATION_STATE

    source_arg = v038._pop_arg_value("--upgrade-resume")
    if source_arg is None:
        return

    source = Path(source_arg)
    target = Path(
        v038._arg_value("--checkpoint", "checkpoints/planet_geometry_v11_imitation.pt")
        or "checkpoints/planet_geometry_v11_imitation.pt"
    )
    if source.resolve() == target.resolve():
        raise SystemExit("--upgrade-resume requires a different --checkpoint destination")
    if not source.exists():
        raise SystemExit(f"upgrade source checkpoint not found: {source}")

    saved = base.core.torch.load(source, map_location="cpu")
    if saved.get("experiment") != "planet-geometry-ppo-v0.3":
        raise SystemExit("upgrade source is not geometry PPO v0.3")
    source_state = saved.get("model")
    if not isinstance(source_state, dict):
        raise SystemExit("upgrade source has no model state")

    control_dt = float(saved.get("control_dt", 0.010))
    model = PredictiveRecurrentActorCritic(
        input_dim=MOTION_GEOMETRY_INPUT_DIM,
        hidden_dim=int(saved.get("hidden_dim", 64)),
        initial_log_std=float(saved.get("initial_log_std", -0.70)),
    )
    copied = v038._copy_recurrent_weights(model, source_state)

    calibration = calibrate_single_press_lead(control_dt_s=control_dt, same_hand=True)
    teacher = PrivilegedLeadTeacher(calibration)
    demonstrations = collect_demonstrations(
        teacher,
        episodes=IMITATION_EPISODES,
        control_dt=control_dt,
        device=base.core.torch.device("cpu"),
    )
    bc_loss = imitation_pretrain(
        model,
        demonstrations,
        epochs=IMITATION_EPOCHS,
        lr=IMITATION_LR,
    )
    probe = imitation_probe(
        model,
        control_dt=control_dt,
        device=base.core.torch.device("cpu"),
    )

    ppo_optimizer = base.core.torch.optim.Adam(model.parameters(), lr=_optimizer_lr(saved))
    _IMITATION_STATE = {
        "source": str(source),
        "copied": copied,
        "optimizer_reset": True,
        "curriculum_position_preserved": True,
        "episodes": IMITATION_EPISODES,
        "epochs": IMITATION_EPOCHS,
        "lr": IMITATION_LR,
        "active_weight": IMITATION_ACTIVE_WEIGHT,
        "bpm_min": IMITATION_BPM_MIN,
        "bpm_max": IMITATION_BPM_MAX,
        "jitter_ms": IMITATION_JITTER_MS,
        "teacher_press_latency_s": calibration.press_latency_s,
        "teacher_lead_s": calibration.lead_s,
        "bc_final_loss": bc_loss,
        "student_probe": probe,
        "student_observation": "14D motor+geometry+visible-delta only",
        "teacher_privileged_target_time": True,
        "teacher_privilege_transferred_to_student_input": False,
    }

    migrated = dict(saved)
    migrated.pop("pattern_memory_v037", None)
    migrated.pop("pattern_memory_config", None)
    migrated.pop("v037_migration", None)
    migrated["format_version"] = 18
    migrated["model"] = model.state_dict()
    migrated["optimizer"] = ppo_optimizer.state_dict()
    migrated["input_dim"] = MOTION_GEOMETRY_INPUT_DIM
    migrated["trainer_ui_version"] = "0.3.9-imitation-bootstrap"
    migrated["imitation_v039"] = dict(_IMITATION_STATE)
    target.parent.mkdir(parents=True, exist_ok=True)
    base.core.torch.save(migrated, target)

    if "--resume" not in sys.argv:
        sys.argv.append("--resume")
    base._write(
        f"v0.3.9 imitation upgrade: {source.name} -> {target.name}; "
        f"lead={calibration.lead_s*1000:.1f}ms BC={bc_loss:.5f} "
        f"probe145={_fmt_optional(probe['edge_145_error_ms'])}ms "
        f"probe240={_fmt_optional(probe['edge_240_error_ms'])}ms; "
        f"resume P{int(saved.get('curriculum_phase_index', 1))}"
    )


def _load_imitation_state() -> None:
    global _IMITATION_STATE
    checkpoint_arg = v038._arg_value("--checkpoint")
    if not checkpoint_arg:
        return
    checkpoint = Path(checkpoint_arg)
    if not checkpoint.exists():
        return
    saved = base.core.torch.load(checkpoint, map_location="cpu")
    state = saved.get("imitation_v039")
    if isinstance(state, dict):
        _IMITATION_STATE = dict(state)


def _print_header(args, phases: tuple[base.core.CurriculumPhase, ...], mode: str) -> None:
    base._write("=== DMDOD / Planet Geometry PPO v0.3.9 ===")
    base._write(
        f"mode={mode} device={args.device} seed={args.seed} checkpoint={args.checkpoint}"
    )
    base._write(
        f"task={args.notes} BPM={args.bpm_min:g}..{args.bpm_max:g} "
        f"vision={args.vision_hz:g}Hz/{args.vision_latency_ms:g}ms | "
        f"gate X={base.PRECISION_OVERALL_XACC:g}%/min{base.PRECISION_MIN_BPM_XACC:g}% "
        f"PP={base.PRECISION_OVERALL_PP:.0%}/min{base.PRECISION_MIN_BPM_PP:.0%}"
    )
    lead = _IMITATION_STATE.get("teacher_lead_s")
    bc_loss = _IMITATION_STATE.get("bc_final_loss")
    probe = _IMITATION_STATE.get("student_probe")
    probe145 = probe.get("edge_145_error_ms") if isinstance(probe, dict) else None
    probe240 = probe.get("edge_240_error_ms") if isinstance(probe, dict) else None
    base._write(
        f"input=14D visible motion | imitation={int(_IMITATION_STATE.get('episodes', 0))}ep/"
        f"{int(_IMITATION_STATE.get('epochs', 0))}epochs "
        f"lead={(float(lead)*1000 if lead is not None else float('nan')):.1f}ms "
        f"BC={(float(bc_loss) if bc_loss is not None else float('nan')):.5f} "
        f"probe145={_fmt_optional(probe145)}ms probe240={_fmt_optional(probe240)}ms"
    )
    base._write(
        f"PPO+self-supervised next-delta coef={v038.PREDICTION_COEF:g} | "
        "teacher privilege absent from student input"
    )
    base._write(
        f"focus EMA={v036.FOCUS_EMA_ALPHA:.2f} hold={v036.FOCUS_CLEAR_SCREENS} screens"
    )
    if v035.VERBOSE_OUTPUT:
        base._write(
            "teacher uses target timestamp only to create BC actions; PPO policy sees motor+visible geometry/delta"
        )
    else:
        base._write("compact output; use --verbose for per-BPM details")
    base._write()


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    *,
    args,
    phase_index: int,
    phase: base.core.CurriculumPhase,
    global_update: int,
    probe: base.core.Probe,
) -> None:
    _ORIGINAL_V038_SAVE_CHECKPOINT(
        path,
        model,
        optimizer,
        args=args,
        phase_index=phase_index,
        phase=phase,
        global_update=global_update,
        probe=probe,
    )
    saved = base.core.torch.load(path, map_location="cpu")
    saved["format_version"] = 18
    saved["trainer_ui_version"] = "0.3.9-imitation-bootstrap"
    saved["imitation_v039"] = dict(_IMITATION_STATE)
    base.core.torch.save(saved, path)


def _pop_positive_int(name: str, default: int) -> int:
    value = v038._pop_arg_value(name)
    if value is None:
        return default
    result = int(value)
    if result <= 0:
        raise SystemExit(f"{name} must be positive")
    return result


def _pop_positive_float(name: str, default: float) -> float:
    value = v038._pop_arg_value(name)
    if value is None:
        return default
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise SystemExit(f"{name} must be positive")
    return result


def main() -> None:
    global IMITATION_EPISODES, IMITATION_EPOCHS, IMITATION_LR

    IMITATION_EPISODES = _pop_positive_int("--imitation-episodes", IMITATION_EPISODES)
    IMITATION_EPOCHS = _pop_positive_int("--imitation-epochs", IMITATION_EPOCHS)
    IMITATION_LR = _pop_positive_float("--imitation-lr", IMITATION_LR)

    _upgrade_resume_if_requested()
    _load_imitation_state()

    # v0.3.8 remains the PPO/predictive-motion implementation. Replace only
    # the header/checkpoint metadata after bootstrapping the student with BC.
    v038._print_header = _print_header
    v038.save_checkpoint = save_checkpoint
    v038.main()


if __name__ == "__main__":
    main()
