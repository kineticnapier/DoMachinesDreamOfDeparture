from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from dmdod.fly_connectome_policy import NKeyFlyConnectomeActorCritic


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FlyConnectome forward_sequence runtime.")
    parser.add_argument("core")
    parser.add_argument("--input-dim", type=int, default=263)
    parser.add_argument("--keys", type=int, default=8)
    parser.add_argument("--steps", type=int, default=192)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    model = NKeyFlyConnectomeActorCritic(
        input_dim=args.input_dim,
        key_count=args.keys,
        core_path=Path(args.core),
    ).to(device)
    model.eval()

    torch.manual_seed(1701)
    observations = torch.randn(args.steps, args.input_dim, device=device)
    state = model.initial_state(device)

    with torch.no_grad():
        for _ in range(args.warmup):
            model.forward_sequence(observations, state)
        _sync(device)

        samples_ms: list[float] = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            model.forward_sequence(observations, state)
            _sync(device)
            samples_ms.append((time.perf_counter() - start) * 1000.0)

    samples_ms.sort()
    mean_ms = sum(samples_ms) / len(samples_ms)
    median_ms = samples_ms[len(samples_ms) // 2]
    print("=== FlyConnectome forward_sequence benchmark ===")
    print(
        f"device={device} nodes={model.hidden_dim} edges={model.recurrent_weight._nnz()} "
        f"steps={args.steps} repeats={args.repeat}"
    )
    print(
        f"mean={mean_ms:.3f}ms median={median_ms:.3f}ms "
        f"min={samples_ms[0]:.3f}ms max={samples_ms[-1]:.3f}ms"
    )


if __name__ == "__main__":
    main()
