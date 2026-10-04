from __future__ import annotations

import argparse
import statistics
import time

import torch

from dmdod.fly_connectome_policy import NKeyFlyConnectomeActorCritic


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(fn, *, warmup: int, repeat: int, device: torch.device) -> list[float]:
    for _ in range(warmup):
        fn()
    _sync(device)

    samples: list[float] = []
    for _ in range(repeat):
        started = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def _report(label: str, samples: list[float]) -> None:
    print(
        f"{label:8s} mean={statistics.mean(samples):.3f}ms "
        f"median={statistics.median(samples):.3f}ms "
        f"min={min(samples):.3f}ms max={max(samples):.3f}ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark eager vs torch.compile for FlyConnectome forward_sequence."
    )
    parser.add_argument("core")
    parser.add_argument("--steps", type=int, default=192)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.steps <= 0 or args.repeat <= 0 or args.warmup < 0:
        raise SystemExit("steps/repeat must be positive and warmup must be non-negative")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(1701)
    model = NKeyFlyConnectomeActorCritic(
        input_dim=263,
        key_count=8,
        core_path=args.core,
    ).to(device).eval()
    observations = torch.randn(args.steps, 263, device=device)
    initial_state = model.initial_state(device)

    @torch.no_grad()
    def eager_call():
        return model.forward_sequence(observations, initial_state)

    try:
        compiled = torch.compile(model.forward_sequence, fullgraph=False)
    except Exception as exc:
        raise SystemExit(f"torch.compile unavailable: {type(exc).__name__}: {exc}") from exc

    @torch.no_grad()
    def compiled_call():
        return compiled(observations, initial_state)

    eager_out = eager_call()
    try:
        compiled_out = compiled_call()
    except Exception as exc:
        raise SystemExit(f"compiled execution failed: {type(exc).__name__}: {exc}") from exc
    _sync(device)

    diffs = [
        float((first - second).abs().max().item())
        for first, second in zip(eager_out, compiled_out, strict=True)
    ]

    print("=== FlyConnectome torch.compile benchmark ===")
    print(
        f"device={device} nodes={model.hidden_dim} edges={model.recurrent_weight._nnz()} "
        f"steps={args.steps} repeats={args.repeat}"
    )
    print(
        "maxdiff means={:.3e} values={:.3e} state={:.3e}".format(*diffs)
    )

    eager_samples = _measure(
        eager_call, warmup=args.warmup, repeat=args.repeat, device=device
    )
    compiled_samples = _measure(
        compiled_call, warmup=args.warmup, repeat=args.repeat, device=device
    )
    _report("eager", eager_samples)
    _report("compile", compiled_samples)
    print(
        f"speedup={statistics.mean(eager_samples) / statistics.mean(compiled_samples):.2f}x"
    )


if __name__ == "__main__":
    main()
