from __future__ import annotations

"""Profile one representative MaleCNS/FlyConnectome BC chunk on CUDA.

This intentionally avoids dataset loading and evaluation.  It measures the hot
training path directly: forward_sequence -> actuation loss -> backward -> Adam.
The model, recurrent core, sequence length, and optimizer shape match the v1.6.8
bootstrap path closely enough to decide whether a fused/custom recurrent kernel
is worth implementing.
"""

import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from dmdod.connectome.fly_policy import NKeyFlyConnectomeActorCritic
from dmdod.n_key_training import n_key_actuation_loss


def _step(
    model: NKeyFlyConnectomeActorCritic,
    optimizer: torch.optim.Optimizer,
    observations: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    state = model.initial_state(observations.device)
    optimizer.zero_grad(set_to_none=True)
    with record_function("dmdod.forward_sequence"):
        means, _, _ = model.forward_sequence(observations, state)
        predicted = torch.tanh(means)
        loss = n_key_actuation_loss(predicted, targets)
    with record_function("dmdod.backward"):
        loss.backward()
    with record_function("dmdod.clip_grad"):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    with record_function("dmdod.adam_step"):
        optimizer.step()
    return float(loss.detach())


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile FlyConnectome BC hot path")
    parser.add_argument(
        "core",
        nargs="?",
        default="data/malecns/male-cns-v1.0-core4096-w5.pt",
    )
    parser.add_argument("--input-dim", type=int, default=263)
    parser.add_argument("--keys", type=int, default=8)
    parser.add_argument("--sensory-dim", type=int, default=128)
    parser.add_argument("--chunk-steps", type=int, default=192)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--active", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trace", default=None, help="optional Chrome trace JSON path")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this profiler")
    if args.chunk_steps <= 0 or args.warmup < 0 or args.active <= 0:
        raise SystemExit("chunk-steps/active must be positive and warmup non-negative")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model = NKeyFlyConnectomeActorCritic(
        input_dim=args.input_dim,
        key_count=args.keys,
        core_path=Path(args.core),
        sensory_dim=args.sensory_dim,
        initial_log_std=-1.20,
    ).to(device)
    model.prepare_recurrent_runtime()
    model.train()

    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
    )
    observations = torch.randn(args.chunk_steps, args.input_dim, device=device)
    targets = torch.empty(args.chunk_steps, args.keys, device=device).uniform_(-1.0, 1.0)

    print(
        f"device={torch.cuda.get_device_name()} nodes={model.hidden_dim} "
        f"edges={model.recurrent_weight._nnz()} steps={args.chunk_steps} "
        f"warmup={args.warmup} active={args.active}"
    )

    for _ in range(args.warmup):
        _step(model, optimizer, observations, targets)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        losses = []
        for _ in range(args.active):
            losses.append(_step(model, optimizer, observations, targets))
        torch.cuda.synchronize()

    print(f"loss_last={losses[-1]:.6f}")
    print("\n=== CUDA time ===")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
    print("\n=== CPU time ===")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))

    if args.trace:
        trace_path = Path(args.trace)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))
        print(f"trace={trace_path}")


if __name__ == "__main__":
    main()
