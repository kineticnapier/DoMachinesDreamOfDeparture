from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from dmdod.connectome.fly_policy import NKeyFlyConnectomeActorCritic
from dmdod.n_key_motor import NKeyAction


def _legacy_actor_only_action(
    model: NKeyFlyConnectomeActorCritic,
    x: torch.Tensor,
    state: torch.Tensor,
) -> tuple[NKeyAction, torch.Tensor]:
    """Pre-batched-transfer actor-only path used for an apples-to-apples benchmark."""
    encoded = torch.tanh(model.sensory(x))
    next_state = model._advance(encoded, state)
    mean = model.actor_mean(next_state)
    squashed = torch.tanh(mean)
    return NKeyAction(tuple(float(value.item()) for value in squashed)), next_state


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run(
    *,
    model: NKeyFlyConnectomeActorCritic,
    observation: torch.Tensor,
    initial_state: torch.Tensor,
    steps: int,
    bulk_transfer: bool,
) -> tuple[float, NKeyAction, torch.Tensor]:
    state = initial_state.clone()
    last_action: NKeyAction | None = None
    _sync(observation.device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            if bulk_transfer:
                last_action, state = model.deterministic_action(observation, state)
            else:
                last_action, state = _legacy_actor_only_action(model, observation, state)
    _sync(observation.device)
    elapsed = time.perf_counter() - start
    assert last_action is not None
    return elapsed, last_action, state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark FlyConnectome closed-loop deterministic action host synchronization."
    )
    parser.add_argument("core", type=Path)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--input-dim", type=int, default=263)
    parser.add_argument("--keys", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.steps <= 0 or args.warmup < 0:
        raise SystemExit("--steps must be positive and --warmup must be non-negative")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = NKeyFlyConnectomeActorCritic(
        input_dim=args.input_dim,
        key_count=args.keys,
        core_path=args.core,
    ).to(device)
    model.prepare_recurrent_runtime()
    model.eval()
    observation = torch.randn(args.input_dim, dtype=torch.float32, device=device)
    initial_state = model.initial_state(device)

    if args.warmup:
        _run(
            model=model,
            observation=observation,
            initial_state=initial_state,
            steps=args.warmup,
            bulk_transfer=False,
        )
        _run(
            model=model,
            observation=observation,
            initial_state=initial_state,
            steps=args.warmup,
            bulk_transfer=True,
        )

    legacy_s, legacy_action, legacy_state = _run(
        model=model,
        observation=observation,
        initial_state=initial_state,
        steps=args.steps,
        bulk_transfer=False,
    )
    bulk_s, bulk_action, bulk_state = _run(
        model=model,
        observation=observation,
        initial_state=initial_state,
        steps=args.steps,
        bulk_transfer=True,
    )

    action_diff = max(
        abs(a - b) for a, b in zip(legacy_action.as_tuple(), bulk_action.as_tuple(), strict=True)
    )
    state_diff = float((legacy_state - bulk_state).abs().max().item())
    speedup = legacy_s / bulk_s if bulk_s > 0.0 else float("inf")
    print(
        f"device={device} nodes={model.hidden_dim} edges={model.recurrent_weight._nnz()} "
        f"keys={model.action_dim} steps={args.steps} warmup={args.warmup}"
    )
    print(f"legacy-item-x{model.action_dim}: {legacy_s:.6f}s ({args.steps / legacy_s:.1f} step/s)")
    print(f"bulk-cpu-tolist: {bulk_s:.6f}s ({args.steps / bulk_s:.1f} step/s)")
    print(f"speedup={speedup:.3f}x action-max-diff={action_diff:.9g} state-max-diff={state_diff:.9g}")


if __name__ == "__main__":
    main()
