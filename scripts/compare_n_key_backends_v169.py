from __future__ import annotations

"""v1.6.9: compare GRU, sparse reservoir, and MaleCNS on one Validation set."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    import train_real_chart_v080 as v080
    import train_real_chart_v160_n_key_bootstrap as v160
except ModuleNotFoundError:
    from scripts import train_real_chart_v080 as v080
    from scripts import train_real_chart_v160_n_key_bootstrap as v160

from dmdod.fly_connectome_policy import (
    N_KEY_POLICY_BACKEND_FLY_CONNECTOME,
    NKeyFlyConnectomeActorCritic,
)
from dmdod.multichart_dataset import discover_multichart_dataset
from dmdod.n_key_policy import (
    N_KEY_POLICY_BACKEND_GRU,
    N_KEY_POLICY_BACKEND_SPARSE_RESERVOIR,
    NKeyPolicyBase,
    build_n_key_policy,
    n_key_policy_backend_from_checkpoint,
)
from dmdod.n_key_real_chart import n_key_hud_real_chart_input_dim


@dataclass(frozen=True, slots=True)
class Summary:
    backend: str
    hits: int
    targets: int
    x_accuracy_percent: float
    too_early: int
    overload_charts: int
    chart_count: int
    keydowns: int

    @property
    def hit_rate_percent(self) -> float:
        return 100.0 * self.hits / self.targets if self.targets else 0.0

    @property
    def too_early_per_target(self) -> float:
        return self.too_early / self.targets if self.targets else 0.0

    @property
    def keydowns_per_target(self) -> float:
        return self.keydowns / self.targets if self.targets else 0.0


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    return device


def _checkpoint_backend(checkpoint: dict) -> str:
    backend = str(checkpoint.get("n_key_policy_backend", N_KEY_POLICY_BACKEND_GRU))
    if backend == N_KEY_POLICY_BACKEND_FLY_CONNECTOME:
        return backend
    return n_key_policy_backend_from_checkpoint(checkpoint)


def _load_policy(path: Path, device: torch.device) -> tuple[NKeyPolicyBase, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"{path}: checkpoint must be a dict")

    backend = _checkpoint_backend(checkpoint)
    input_dim = int(checkpoint["input_dim"])
    key_count = int(checkpoint["key_count"])
    expected_input = n_key_hud_real_chart_input_dim(key_count)
    if input_dim != expected_input:
        raise SystemExit(
            f"{path}: input_dim={input_dim} does not match {key_count}K expected {expected_input}"
        )

    if backend == N_KEY_POLICY_BACKEND_FLY_CONNECTOME:
        core_path = checkpoint.get("fly_connectome_core_path")
        if not core_path:
            raise SystemExit(f"{path}: fly checkpoint is missing fly_connectome_core_path")
        model = NKeyFlyConnectomeActorCritic(
            input_dim=input_dim,
            key_count=key_count,
            core_path=core_path,
            sensory_dim=int(checkpoint.get("fly_connectome_sensory_dim", 128)),
            recurrent_gain=float(checkpoint.get("fly_connectome_recurrent_gain", 0.9)),
            projection_seed=int(checkpoint.get("fly_connectome_projection_seed", 1701)),
            initial_log_std=-1.20,
        )
    else:
        model = build_n_key_policy(
            backend=backend,
            input_dim=input_dim,
            key_count=key_count,
            hidden_dim=int(checkpoint["hidden_dim"]),
            initial_log_std=-1.20,
            reservoir_density=float(checkpoint.get("reservoir_density", 0.10)),
            reservoir_gain=float(checkpoint.get("reservoir_gain", 0.90)),
            reservoir_seed=int(checkpoint.get("reservoir_seed", 1701)),
        )

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.prepare_recurrent_runtime()
    model.eval()
    return model, checkpoint


def summarize(backend: str, results: list[tuple[object, int]]) -> Summary:
    targets = sum(int(stats.targets) for stats, _ in results)
    hits = sum(int(stats.hits) for stats, _ in results)
    early = sum(int(stats.too_early_presses) for stats, _ in results)
    keydowns = sum(int(keydowns) for _, keydowns in results)
    overload_charts = sum(1 for stats, _ in results if bool(stats.overloaded))
    x_points = sum(float(stats.x_accuracy_points) for stats, _ in results)
    x_den = sum(float(stats.x_accuracy_denominator) for stats, _ in results)
    xacc = 100.0 * x_points / x_den if x_den > 0.0 else 0.0
    return Summary(
        backend=backend,
        hits=hits,
        targets=targets,
        x_accuracy_percent=xacc,
        too_early=early,
        overload_charts=overload_charts,
        chart_count=len(results),
        keydowns=keydowns,
    )


def _format_summary(summary: Summary) -> str:
    return (
        f"{summary.backend:18s} "
        f"H={summary.hits}/{summary.targets} "
        f"hit-rate={summary.hit_rate_percent:6.2f}% "
        f"X={summary.x_accuracy_percent:6.2f}% "
        f"early/target={summary.too_early_per_target:.4f} "
        f"overload={summary.overload_charts}/{summary.chart_count} "
        f"keydown/target={summary.keydowns_per_target:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare N-key policy backends on exactly the same Validation segments."
    )
    parser.add_argument("dataset")
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--validation-limit", type=int, default=20)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.validation_limit <= 0:
        raise SystemExit("--validation-limit must be positive")
    device = _device_from_arg(args.device)

    dataset = discover_multichart_dataset(args.dataset)
    validation_charts = v080._compile_role(dataset.validation)

    loaded: list[tuple[Path, NKeyPolicyBase, dict]] = []
    reference = None
    for raw_path in args.checkpoints:
        path = Path(raw_path)
        model, checkpoint = _load_policy(path, device)
        signature = (
            int(checkpoint["key_count"]),
            float(checkpoint.get("validation_window", v080.DEFAULT_VALIDATION_WINDOW_S)),
            float(checkpoint["control_dt"]),
            float(checkpoint.get("physics_dt", 0.001)),
        )
        if reference is None:
            reference = signature
        elif signature != reference:
            raise SystemExit(
                f"{path}: evaluation settings {signature} do not match first checkpoint {reference}"
            )
        loaded.append((path, model, checkpoint))

    assert reference is not None
    key_count, validation_window, control_dt, physics_dt = reference
    validation = v080._build_validation_segments(
        validation_charts,
        window_s=validation_window,
    )[: args.validation_limit]

    print("=== DMDOD v1.6.9 N-Key Backend Fair Validation Comparison ===")
    print(
        f"device={device} keys={key_count} validation={len(validation)} "
        f"window={validation_window:g}s control={control_dt * 1000:.1f}ms "
        f"physics={physics_dt * 1000:.1f}ms"
    )

    summaries: list[Summary] = []
    for path, model, checkpoint in loaded:
        backend = _checkpoint_backend(checkpoint)
        print(f"\n=== {backend} | {path} ===")
        results = v160._evaluate_role(
            model,
            validation,
            role_label=backend,
            control_dt_s=control_dt,
            physics_dt_s=physics_dt,
            device=device,
        )
        summaries.append(summarize(backend, results))

    print("\n=== FAIR VALIDATION SUMMARY ===")
    for summary in summaries:
        print(_format_summary(summary))


if __name__ == "__main__":
    main()
