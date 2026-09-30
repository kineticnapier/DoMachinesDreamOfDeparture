from __future__ import annotations

"""Execution-only CUDA backend for expensive round BC proposals.

The simulator, guards, checkpoints, and trusted policy remain on CPU. BC is
trained on a CUDA copy of the trusted model and only committed back to the CPU
model after the complete epoch succeeds. A CUDA failure therefore leaves the
caller's model untouched and permits an exact CPU fallback.

Training semantics stay the same at the algorithm level: sequence order, chunk
boundaries, recurrent-state detach points, weighted loss, gradient clipping,
and one fresh Adam step per chunk are unchanged. CPU and CUDA floating-point
kernels are not bit-identical, so this backend intentionally promises numerical
closeness rather than byte identity.
"""

import copy
import os
import time
from dataclasses import dataclass

import torch

import train_real_chart_v056 as v056
import train_real_chart_v057 as v057
import train_real_chart_v070 as v070
from dmdod.training_progress import emit_progress


CUDA_BC_VERSION = "v090-bidir-bc-cuda-v3-packed-gru"
CUDA_PROGRESS_LOSS_EVERY = 32
CUDA_ENV = "DMDOD_BC_CUDA"


@dataclass(frozen=True, slots=True)
class _DeviceSequence:
    observations: torch.Tensor
    teacher_actions: torch.Tensor
    source: str

    @property
    def frames(self) -> int:
        return int(self.observations.shape[0])


@dataclass(frozen=True, slots=True)
class _DeviceStableSequence:
    sequence: _DeviceSequence
    loss_weights: torch.Tensor


_SEQUENCE_CACHE: dict[tuple, _DeviceStableSequence] = {}
_SEQUENCE_CACHE_HITS = 0
_SEQUENCE_CACHE_MISSES = 0


def cuda_requested() -> bool:
    """Return whether the CUDA BC backend should be attempted."""

    raw = os.environ.get(CUDA_ENV, "auto").strip().lower()
    if raw in {"0", "false", "off", "no", "cpu"}:
        return False
    if raw not in {"", "auto", "1", "true", "on", "yes", "cuda"}:
        raise SystemExit(f"{CUDA_ENV} must be auto/1/0/cuda/cpu")
    return bool(torch.cuda.is_available())


def cuda_device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _device_sequence(stable, device: torch.device) -> _DeviceStableSequence:
    """Cache immutable trajectories in VRAM by their exact sequence signature."""

    global _SEQUENCE_CACHE_HITS, _SEQUENCE_CACHE_MISSES
    signature = v070._stable_sequence_signature(stable)
    cached = _SEQUENCE_CACHE.get(signature)
    if cached is not None:
        _SEQUENCE_CACHE_HITS += 1
        return cached

    sequence = stable.sequence
    copied = _DeviceStableSequence(
        sequence=_DeviceSequence(
            observations=sequence.observations.to(device=device, dtype=torch.float32),
            teacher_actions=sequence.teacher_actions.to(device=device, dtype=torch.float32),
            source=str(sequence.source),
        ),
        loss_weights=stable.loss_weights.to(device=device, dtype=torch.float32),
    )
    _SEQUENCE_CACHE[signature] = copied
    _SEQUENCE_CACHE_MISSES += 1
    return copied


def _packed_sequence_gru(model, device: torch.device) -> torch.nn.GRU:
    """Create a local packed cuDNN GRU initialized from the model's GRUCell."""

    gru = torch.nn.GRU(
        model.hidden_dim,
        model.hidden_dim,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0.0,
        bidirectional=False,
        device=device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        gru.weight_ih_l0.copy_(model.gru.weight_ih)
        gru.weight_hh_l0.copy_(model.gru.weight_hh)
        gru.bias_ih_l0.copy_(model.gru.bias_ih)
        gru.bias_hh_l0.copy_(model.gru.bias_hh)
    gru.train(model.training)
    gru.flatten_parameters()
    return gru


def _cuda_policy_parameters(model, sequence_gru: torch.nn.GRU) -> list[torch.nn.Parameter]:
    """Match v0.5.7 policy parameter order, substituting the packed GRU."""

    return [
        *model.input_layer.parameters(),
        *sequence_gru.parameters(),
        *model.post.parameters(),
        *model.actor_mean.parameters(),
    ]


def _forward_sequence_cuda(
    model,
    sequence_gru: torch.nn.GRU,
    observations: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sequence forward using the packed GRU and the model's surrounding heads."""

    encoded = torch.tanh(model.input_layer(observations)).unsqueeze(1)
    hx = initial_state.reshape(1, 1, model.hidden_dim)
    recurrent, final_state = sequence_gru(encoded, hx)
    recurrent = recurrent.squeeze(1)
    features = torch.tanh(model.post(recurrent))
    means = model.actor_mean(features)
    values = model.critic(features).squeeze(-1)
    return means, values, final_state.reshape(model.hidden_dim)


def _copy_packed_gru_back(model, sequence_gru: torch.nn.GRU) -> None:
    """Commit the temporary GRU parameters back into the model's GRUCell."""

    with torch.no_grad():
        model.gru.weight_ih.copy_(sequence_gru.weight_ih_l0)
        model.gru.weight_hh.copy_(sequence_gru.weight_hh_l0)
        model.gru.bias_ih.copy_(sequence_gru.bias_ih_l0)
        model.gru.bias_hh.copy_(sequence_gru.bias_hh_l0)


def _fresh_cuda_optimizer(
    source_optimizer,
    parameters: list[torch.nn.Parameter],
) -> torch.optim.Optimizer:
    """Recreate the trainer's fresh Adam on CUDA without moving CPU Parameters."""

    if not isinstance(source_optimizer, torch.optim.Adam):
        raise TypeError("CUDA BC currently requires the trainer's Adam optimizer")
    if source_optimizer.state:
        raise ValueError("CUDA BC requires a fresh optimizer with no state")
    if len(source_optimizer.param_groups) != 1:
        raise ValueError("CUDA BC currently requires one Adam parameter group")

    group = source_optimizer.param_groups[0]
    optimizer = torch.optim.Adam(
        parameters,
        lr=float(group["lr"]),
        betas=tuple(float(value) for value in group["betas"]),
        eps=float(group["eps"]),
        weight_decay=float(group["weight_decay"]),
        amsgrad=bool(group["amsgrad"]),
        maximize=bool(group.get("maximize", False)),
    )
    return optimizer


def train_on_cuda(
    model,
    sequences,
    *,
    optimizer,
    chunk_steps: int,
    reverse_order: bool,
) -> float:
    """Train one BC epoch on CUDA in the requested sequence order."""

    if not sequences:
        raise ValueError("at least one stable DAgger sequence is required")
    if chunk_steps <= 0:
        raise ValueError("chunk_steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = cuda_device()
    total_started = time.perf_counter()

    # Preserve the CPU trusted model and source optimizer. The caller can safely
    # fall back to the CPU trainer if anything below raises.
    cuda_model = copy.deepcopy(model).to(device)
    sequence_gru = _packed_sequence_gru(cuda_model, device)
    parameters = _cuda_policy_parameters(cuda_model, sequence_gru)
    cuda_optimizer = _fresh_cuda_optimizer(optimizer, parameters)

    cache_hits_before = _SEQUENCE_CACHE_HITS
    cache_misses_before = _SEQUENCE_CACHE_MISSES
    device_sequences = [_device_sequence(stable, device) for stable in sequences]
    cpu_order = list(reversed(sequences)) if reverse_order else list(sequences)
    device_order = list(reversed(device_sequences)) if reverse_order else device_sequences
    ordered_pairs = list(zip(cpu_order, device_order))

    torch.cuda.synchronize(device)
    setup_seconds = time.perf_counter() - total_started

    total_chunks = sum(
        (int(cpu_stable.sequence.frames) + int(chunk_steps) - 1) // int(chunk_steps)
        for cpu_stable, _ in ordered_pairs
    )
    emit_progress(
        "bc_start",
        sequences=len(ordered_pairs),
        chunks=total_chunks,
        reverse=bool(reverse_order),
    )

    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    weighted_loss = torch.zeros((), dtype=torch.float32, device=device)
    weighted_elements = 0.0
    global_chunk = 0
    displayed_loss = 0.0
    train_started = time.perf_counter()

    try:
        for sequence_index, (cpu_stable, stable) in enumerate(ordered_pairs, 1):
            sequence = stable.sequence
            cpu_sequence = cpu_stable.sequence
            sequence_chunks = (
                int(cpu_sequence.frames) + int(chunk_steps) - 1
            ) // int(chunk_steps)
            source = str(cpu_sequence.source)
            emit_progress(
                "bc_sequence_start",
                index=sequence_index,
                total=len(ordered_pairs),
                chunks=sequence_chunks,
                frames=int(cpu_sequence.frames),
                source=source,
                reverse=bool(reverse_order),
            )

            state = cuda_model.initial_state(device)
            chunk_index = 0
            for start in range(0, int(cpu_sequence.frames), int(chunk_steps)):
                end = min(int(cpu_sequence.frames), start + int(chunk_steps))
                state = state.detach()

                means, _values, state = _forward_sequence_cuda(
                    cuda_model,
                    sequence_gru,
                    sequence.observations[start:end],
                    state,
                )
                predicted = torch.tanh(means)
                target = sequence.teacher_actions[start:end]
                weights = stable.loss_weights[start:end]
                loss = v056._weighted_actuation_loss(predicted, target, weights)

                cuda_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                cuda_optimizer.step()

                cpu_weights = cpu_stable.loss_weights[start:end]
                weight = max(float(cpu_weights.sum().item()), 1.0)
                weighted_loss.add_(loss.detach() * weight)
                weighted_elements += weight

                chunk_index += 1
                global_chunk += 1
                if (
                    global_chunk == 1
                    or global_chunk % CUDA_PROGRESS_LOSS_EVERY == 0
                    or global_chunk == total_chunks
                ):
                    displayed_loss = float(loss.detach().item())

                emit_progress(
                    "bc_chunk",
                    sequence=sequence_index,
                    sequence_total=len(ordered_pairs),
                    chunk=chunk_index,
                    chunk_total=sequence_chunks,
                    global_chunk=global_chunk,
                    global_total=total_chunks,
                    loss=displayed_loss,
                    source=source,
                    reverse=bool(reverse_order),
                )

            emit_progress(
                "bc_sequence_done",
                index=sequence_index,
                total=len(ordered_pairs),
                source=source,
            )

        torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - train_started
        final_loss = float((weighted_loss / max(1.0, weighted_elements)).item())

        _copy_packed_gru_back(cuda_model, sequence_gru)
        copy_started = time.perf_counter()
        cpu_state = {
            key: value.detach().cpu().clone()
            for key, value in cuda_model.state_dict().items()
        }
        model.load_state_dict(cpu_state)
        copy_seconds = time.perf_counter() - copy_started
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32

    total_seconds = time.perf_counter() - total_started
    cache_hits = _SEQUENCE_CACHE_HITS - cache_hits_before
    cache_misses = _SEQUENCE_CACHE_MISSES - cache_misses_before
    parity = "rev" if reverse_order else "fwd"
    print(
        f"bc-cuda: version={CUDA_BC_VERSION} parity={parity} "
        f"device={torch.cuda.get_device_name(device)} total={total_seconds:.2f}s "
        f"setup={setup_seconds:.2f}s train={train_seconds:.2f}s "
        f"copyback={copy_seconds:.2f}s chunks={global_chunk} "
        f"seq-cache={cache_hits}hit/{cache_misses}miss"
    )
    emit_progress("bc_done", loss=final_loss, reverse=bool(reverse_order))
    return final_loss


def train_reverse_on_cuda(
    model,
    sequences,
    *,
    optimizer,
    chunk_steps: int,
) -> float:
    """Backward-compatible wrapper for the original reverse-only API."""

    return train_on_cuda(
        model,
        sequences,
        optimizer=optimizer,
        chunk_steps=chunk_steps,
        reverse_order=True,
    )


def stats() -> dict[str, int]:
    return {
        "sequence_cache_entries": len(_SEQUENCE_CACHE),
        "sequence_cache_hits": _SEQUENCE_CACHE_HITS,
        "sequence_cache_misses": _SEQUENCE_CACHE_MISSES,
    }
