from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True, slots=True)
class RunConfig:
    dataset: str
    checkpoint: str
    output: str
    device: str = "auto"


@dataclass(frozen=True, slots=True)
class DataConfig:
    anchor_limit: int | None = None
    validation_limit: int | None = None
    chunk_steps: int | None = None


@dataclass(frozen=True, slots=True)
class ActionTrustConfig:
    actor_steps: int = 8
    lr: float = 3e-4
    stay_coef: float = 20.0
    initial_action_rms: float = 0.01
    min_action_rms: float = 1e-6
    lr_backoffs: int = 10
    min_lr: float = 1e-8


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    hours: float = 8.0
    reserve_minutes: float = 8.0
    max_trials: int = 10_000
    reject_shrink: float = 0.5
    safe_grow: float = 1.25


@dataclass(frozen=True, slots=True)
class TrajectoryProbeConfig:
    candidate_action_rms: float = 2.5e-5


@dataclass(frozen=True, slots=True)
class BoundaryTrustConfig:
    preserve_safe_only: bool = True
    mismatch_grace_s: float = 0.030


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    mode: str
    run: RunConfig
    data: DataConfig
    action_trust: ActionTrustConfig
    budget: BudgetConfig
    trajectory_probe: TrajectoryProbeConfig
    boundary_trust: BoundaryTrustConfig

    def as_dict(self) -> dict:
        return asdict(self)


def _section(raw: dict, name: str) -> dict:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    return value


def _optional_positive_int(value, *, name: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def load_training_config(path: str | Path) -> TrainingConfig:
    path = Path(path)
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    mode = str(raw.get("mode", "")).strip()
    if not mode:
        raise ValueError("top-level 'mode' is required")

    run_raw = _section(raw, "run")
    missing = [name for name in ("dataset", "checkpoint", "output") if name not in run_raw]
    if missing:
        raise ValueError("[run] is missing: " + ", ".join(missing))
    run = RunConfig(
        dataset=str(run_raw["dataset"]),
        checkpoint=str(run_raw["checkpoint"]),
        output=str(run_raw["output"]),
        device=str(run_raw.get("device", "auto")),
    )

    data_raw = _section(raw, "data")
    data = DataConfig(
        anchor_limit=_optional_positive_int(
            data_raw.get("anchor_limit"),
            name="data.anchor_limit",
        ),
        validation_limit=_optional_positive_int(
            data_raw.get("validation_limit"),
            name="data.validation_limit",
        ),
        chunk_steps=_optional_positive_int(
            data_raw.get("chunk_steps"),
            name="data.chunk_steps",
        ),
    )

    trust_raw = _section(raw, "action_trust")
    action_trust = ActionTrustConfig(
        actor_steps=int(trust_raw.get("actor_steps", 8)),
        lr=float(trust_raw.get("lr", 3e-4)),
        stay_coef=float(trust_raw.get("stay_coef", 20.0)),
        initial_action_rms=float(trust_raw.get("initial_action_rms", 0.01)),
        min_action_rms=float(trust_raw.get("min_action_rms", 1e-6)),
        lr_backoffs=int(trust_raw.get("lr_backoffs", 10)),
        min_lr=float(trust_raw.get("min_lr", 1e-8)),
    )

    budget_raw = _section(raw, "budget")
    budget = BudgetConfig(
        hours=float(budget_raw.get("hours", 8.0)),
        reserve_minutes=float(budget_raw.get("reserve_minutes", 8.0)),
        max_trials=int(budget_raw.get("max_trials", 10_000)),
        reject_shrink=float(budget_raw.get("reject_shrink", 0.5)),
        safe_grow=float(budget_raw.get("safe_grow", 1.25)),
    )

    probe_raw = _section(raw, "trajectory_probe")
    trajectory_probe = TrajectoryProbeConfig(
        candidate_action_rms=float(
            probe_raw.get("candidate_action_rms", 2.5e-5)
        ),
    )

    boundary_raw = _section(raw, "boundary_trust")
    boundary_trust = BoundaryTrustConfig(
        preserve_safe_only=bool(boundary_raw.get("preserve_safe_only", True)),
        mismatch_grace_s=float(boundary_raw.get("mismatch_grace_s", 0.030)),
    )

    _validate(action_trust, budget, trajectory_probe, boundary_trust)
    return TrainingConfig(
        mode=mode,
        run=run,
        data=data,
        action_trust=action_trust,
        budget=budget,
        trajectory_probe=trajectory_probe,
        boundary_trust=boundary_trust,
    )


def _validate(
    action: ActionTrustConfig,
    budget: BudgetConfig,
    trajectory_probe: TrajectoryProbeConfig,
    boundary_trust: BoundaryTrustConfig,
) -> None:
    if action.actor_steps <= 0:
        raise ValueError("action_trust.actor_steps must be positive")
    if action.lr <= 0.0 or action.min_lr <= 0.0:
        raise ValueError("action_trust learning rates must be positive")
    if action.stay_coef < 0.0:
        raise ValueError("action_trust.stay_coef must be non-negative")
    if action.initial_action_rms <= 0.0 or action.min_action_rms <= 0.0:
        raise ValueError("action trust RMS bounds must be positive")
    if action.min_action_rms > action.initial_action_rms:
        raise ValueError(
            "action_trust.min_action_rms cannot exceed initial_action_rms"
        )
    if action.lr_backoffs < 0:
        raise ValueError("action_trust.lr_backoffs must be non-negative")

    if budget.hours <= 0.0:
        raise ValueError("budget.hours must be positive")
    if budget.reserve_minutes < 0.0:
        raise ValueError("budget.reserve_minutes must be non-negative")
    if budget.reserve_minutes * 60.0 >= budget.hours * 3600.0:
        raise ValueError("budget reserve must be smaller than total budget")
    if budget.max_trials <= 0:
        raise ValueError("budget.max_trials must be positive")
    if not (0.0 < budget.reject_shrink < 1.0):
        raise ValueError("budget.reject_shrink must be in (0, 1)")
    if budget.safe_grow < 1.0:
        raise ValueError("budget.safe_grow must be >= 1")
    if trajectory_probe.candidate_action_rms <= 0.0:
        raise ValueError("trajectory_probe.candidate_action_rms must be positive")
    if boundary_trust.mismatch_grace_s < 0.0:
        raise ValueError("boundary_trust.mismatch_grace_s must be non-negative")
