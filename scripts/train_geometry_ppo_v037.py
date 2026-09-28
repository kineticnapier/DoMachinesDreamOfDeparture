from __future__ import annotations

import sys
from pathlib import Path

try:
    from . import train_geometry_ppo_v036 as v036
except ImportError:  # direct script execution
    import train_geometry_ppo_v036 as v036

from dmdod.pattern_geometry_env import PatternMemoryGeometryEnv
from dmdod.pattern_memory import PatternMemory
from dmdod.rhythm_env import RewardConfig, make_regular_targets
from dmdod.toy_policy import PATTERN_GEOMETRY_INPUT_DIM


v035 = v036.v035
v034 = v036.v034
base = v036.base

PRACTICE_MEMORY_ENABLED = True
PATTERN_MEMORY = PatternMemory(history_frames=3)

_ORIGINAL_V036_SAVE_CHECKPOINT = v036.save_checkpoint
_ORIGINAL_WARM_START = base.core.warm_start


def _chart_id(*, bpm: float, notes: int, config) -> str:
    """Synthetic chart identity without target timestamps or phase start time."""

    return (
        f"straight|n={notes}|bpm={bpm:.6f}|"
        f"lat={config.latency_s:.6f}|jit={config.latency_jitter_s:.6f}|"
        f"sample={config.sample_period_s:.6f}|noise={config.position_noise_std:.6f}|"
        f"drop={config.dropout_probability:.6f}"
    )


def make_env(
    *,
    bpm: float,
    notes: int,
    start_s: float,
    control_dt: float,
    config,
    seed: int,
) -> PatternMemoryGeometryEnv:
    return PatternMemoryGeometryEnv(
        make_regular_targets(bpm=bpm, count=notes, start_s=start_s, pattern="left"),
        bpm=bpm,
        pattern_memory=PATTERN_MEMORY,
        chart_id=_chart_id(bpm=bpm, notes=notes, config=config),
        practice_memory=PRACTICE_MEMORY_ENABLED,
        same_hand=True,
        control_dt_s=control_dt,
        vision_config=config,
        perception_seed=seed,
        reward_config=RewardConfig(too_early_penalty=v035.TOO_EARLY_PENALTY),
    )


def _copy_recurrent_weights(model, source_state: dict[str, object]) -> str:
    """Expand 10-D geometry checkpoints to the 22-D pattern input safely."""

    target = model.state_dict()
    copied: list[str] = []
    for name, target_value in target.items():
        source_value = source_state.get(name)
        if source_value is None:
            continue
        if source_value.shape == target_value.shape:
            target_value.copy_(source_value)
            copied.append(name)
            continue
        if (
            name == "input_layer.weight"
            and source_value.ndim == 2
            and target_value.ndim == 2
            and source_value.shape[0] == target_value.shape[0]
        ):
            # Preserve the old policy exactly at migration time: new memory
            # columns initially have zero influence and are learned from there.
            target_value.zero_()
            columns = min(source_value.shape[1], target_value.shape[1])
            target_value[:, :columns].copy_(source_value[:, :columns])
            copied.append(f"{name}[0:{columns}]")

    model.load_state_dict(target)
    return ", ".join(copied) or "no compatible recurrent weights"


def warm_start(model, checkpoint: Path, device) -> str:
    saved = base.core.torch.load(checkpoint, map_location=device)
    source = saved.get("model")
    if saved.get("experiment") == "planet-geometry-ppo-v0.3" and isinstance(source, dict):
        return _copy_recurrent_weights(model, source)
    return _ORIGINAL_WARM_START(model, checkpoint, device)


def _arg_value(name: str, default: str | None = None) -> str | None:
    prefix = name + "="
    for i, value in enumerate(sys.argv):
        if value.startswith(prefix):
            return value[len(prefix) :]
        if value == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def _pop_arg_value(name: str) -> str | None:
    prefix = name + "="
    for i, value in enumerate(tuple(sys.argv)):
        if value.startswith(prefix):
            sys.argv.remove(value)
            return value[len(prefix) :]
        if value == name:
            if i + 1 >= len(sys.argv):
                raise SystemExit(f"{name} requires a value")
            result = sys.argv[i + 1]
            del sys.argv[i : i + 2]
            return result
    return None


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
    """Create a 22-D checkpoint while preserving the old curriculum position."""

    source_arg = _pop_arg_value("--upgrade-resume")
    if source_arg is None:
        return

    source = Path(source_arg)
    target = Path(
        _arg_value("--checkpoint", "checkpoints/planet_geometry_v09_pattern_memory.pt")
        or "checkpoints/planet_geometry_v09_pattern_memory.pt"
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

    model = base.core.RecurrentActorCritic(
        input_dim=PATTERN_GEOMETRY_INPUT_DIM,
        hidden_dim=int(saved.get("hidden_dim", 64)),
        initial_log_std=float(saved.get("initial_log_std", -0.70)),
    )
    copied = _copy_recurrent_weights(model, source_state)
    optimizer = base.core.torch.optim.Adam(model.parameters(), lr=_optimizer_lr(saved))

    PATTERN_MEMORY.load_state_dict(saved.get("pattern_memory_v037"))
    migrated = dict(saved)
    migrated["format_version"] = 16
    migrated["model"] = model.state_dict()
    migrated["optimizer"] = optimizer.state_dict()
    migrated["input_dim"] = PATTERN_GEOMETRY_INPUT_DIM
    migrated["trainer_ui_version"] = "0.3.7-pattern-memory"
    migrated["pattern_memory_v037"] = PATTERN_MEMORY.state_dict()
    migrated["v037_migration"] = {
        "source": str(source),
        "copied": copied,
        "optimizer_reset": True,
        "curriculum_position_preserved": True,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    base.core.torch.save(migrated, target)

    if "--resume" not in sys.argv:
        sys.argv.append("--resume")
    base._write(
        f"v0.3.7 checkpoint upgrade: {source.name} -> {target.name}; "
        f"resume P{int(saved.get('curriculum_phase_index', 1))}"
    )


def _load_memory_for_existing_run() -> None:
    source: Path | None = None
    if "--resume" in sys.argv:
        checkpoint = _arg_value("--checkpoint")
        if checkpoint:
            source = Path(checkpoint)
    else:
        warm = _arg_value("--warm-start")
        if warm:
            source = Path(warm)
    if source is None or not source.exists():
        return
    saved = base.core.torch.load(source, map_location="cpu")
    PATTERN_MEMORY.load_state_dict(saved.get("pattern_memory_v037"))


def _print_header(args, phases: tuple[base.core.CurriculumPhase, ...], mode: str) -> None:
    base._write("=== DMDOD / Planet Geometry PPO v0.3.7 ===")
    base._write(
        f"mode={mode} device={args.device} seed={args.seed} checkpoint={args.checkpoint}"
    )
    base._write(
        f"task={args.notes} BPM={args.bpm_min:g}..{args.bpm_max:g} "
        f"vision={args.vision_hz:g}Hz/{args.vision_latency_ms:g}ms | "
        f"gate X={base.PRECISION_OVERALL_XACC:g}%/min{base.PRECISION_MIN_BPM_XACC:g}% "
        f"PP={base.PRECISION_OVERALL_PP:.0%}/min{base.PRECISION_MIN_BPM_PP:.0%}"
    )
    base._write(
        f"input={PATTERN_GEOMETRY_INPUT_DIM}D: motor+geometry+frame-delta+"
        f"shared/chart pattern memory | practice={'on' if PRACTICE_MEMORY_ENABLED else 'off'}"
    )
    base._write(
        f"memory shared={PATTERN_MEMORY.shared_entry_count} chart={PATTERN_MEMORY.chart_entry_count}; "
        f"focus EMA={v036.FOCUS_EMA_ALPHA:.2f} hold={v036.FOCUS_CLEAR_SCREENS}"
    )
    if v035.VERBOSE_OUTPUT:
        base._write(
            "pattern keys use only recent visible geometry; feedback stores hit error/action, not target time"
        )
        base._write("visible excludes time/BPM/target-angle/error/direction")
    else:
        base._write("compact output; use --verbose for per-BPM details")
    base._write()


def save_checkpoint(
    path: Path,
    model: base.core.RecurrentActorCritic,
    optimizer: base.core.torch.optim.Optimizer,
    *,
    args,
    phase_index: int,
    phase: base.core.CurriculumPhase,
    global_update: int,
    probe: base.core.Probe,
) -> None:
    _ORIGINAL_V036_SAVE_CHECKPOINT(
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
    saved["format_version"] = 16
    saved["trainer_ui_version"] = "0.3.7-pattern-memory"
    saved["input_dim"] = model.input_dim
    saved["pattern_memory_v037"] = PATTERN_MEMORY.state_dict()
    saved["pattern_memory_config"] = {
        "history_frames": PATTERN_MEMORY.history_frames,
        "shared_typical_patterns": True,
        "per_chart_practice": PRACTICE_MEMORY_ENABLED,
        "explicit_frame_delta": True,
        "stores_target_timestamp": False,
    }
    base.core.torch.save(saved, path)


def main() -> None:
    global PRACTICE_MEMORY_ENABLED

    if "--sight-read" in sys.argv:
        sys.argv.remove("--sight-read")
        PRACTICE_MEMORY_ENABLED = False

    # New observation columns require a model migration once. --upgrade-resume
    # preserves the v0.3.6 curriculum phase while intentionally resetting the
    # optimizer, because its input-layer moments have the old shape.
    _upgrade_resume_if_requested()
    _load_memory_for_existing_run()

    base.core.GEOMETRY_INPUT_DIM = PATTERN_GEOMETRY_INPUT_DIM
    base.core.warm_start = warm_start
    v035.make_env = make_env
    v036._print_header = _print_header
    v036.save_checkpoint = save_checkpoint
    v036.main()


if __name__ == "__main__":
    main()
