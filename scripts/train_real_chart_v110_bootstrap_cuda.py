from __future__ import annotations

"""Persistent CUDA backend for multi-chart bootstrap BC.

The v0.8+ bootstrap does not call ``v057._train_one_epoch``.  It owns a separate
``v062._bootstrap_multi_epoch`` loop with one Adam optimizer that persists across
bootstrap epochs.  The normal round CUDA/chunk acceleration therefore cannot
speed this stage.

This module mirrors that bootstrap loop on CUDA while preserving the algorithmic
semantics that matter to selection:

* the caller's Adam optimizer persists across epochs;
* forward/reverse trajectory order is unchanged;
* every trajectory starts from a fresh recurrent state;
* chunk boundaries and detach points are unchanged;
* the same actuation loss and gradient clipping are used;
* each finished epoch is copied back to the ordinary CPU model before gameplay
  evaluation.

The CUDA model, packed GRU, optimizer moments, and immutable expert trajectories
stay resident on the GPU between epochs.  CPU and CUDA kernels are not bitwise
identical, so this is an execution backend with numerical rather than byte-level
parity.
"""

import copy
import time
from dataclasses import dataclass

import torch

import train_real_chart_v054 as v054
import train_real_chart_v062 as v062
import train_real_chart_v090_cuda_bc as cuda_bc
from dmdod.training_progress import emit_progress


BOOTSTRAP_CUDA_VERSION = "v110-bootstrap-cuda-packed-gru-v1"
_BASE_BOOTSTRAP_MULTI_EPOCH = v062._bootstrap_multi_epoch
_INSTALLED = False


@dataclass
class _BootstrapCudaSession:
    model: object
    source_optimizer: torch.optim.Optimizer
    cuda_model: object
    sequence_gru: torch.nn.GRU
    optimizer: torch.optim.Optimizer
    pairs: list[tuple[torch.Tensor, torch.Tensor]]
    device: torch.device


_SESSION: _BootstrapCudaSession | None = None


def _copy_pairs_to_device(expert_pairs, device: torch.device):
    return [
        (
            observations.to(device=device, dtype=torch.float32),
            actions.to(device=device, dtype=torch.float32),
        )
        for observations, actions in expert_pairs
    ]


def _new_session(model, expert_pairs, optimizer) -> _BootstrapCudaSession:
    if optimizer.state:
        raise RuntimeError(
            "bootstrap CUDA session must start before the persistent Adam optimizer has CPU state"
        )

    device = cuda_bc.cuda_device()
    cuda_model = copy.deepcopy(model).to(device)
    sequence_gru = cuda_bc._packed_sequence_gru(cuda_model, device)
    parameters = cuda_bc._cuda_policy_parameters(cuda_model, sequence_gru)
    cuda_optimizer = cuda_bc._fresh_cuda_optimizer(optimizer, parameters)
    pairs = _copy_pairs_to_device(expert_pairs, device)
    return _BootstrapCudaSession(
        model=model,
        source_optimizer=optimizer,
        cuda_model=cuda_model,
        sequence_gru=sequence_gru,
        optimizer=cuda_optimizer,
        pairs=pairs,
        device=device,
    )


def _session_for(model, expert_pairs, optimizer) -> _BootstrapCudaSession:
    global _SESSION
    if (
        _SESSION is None
        or _SESSION.model is not model
        or _SESSION.source_optimizer is not optimizer
    ):
        _SESSION = _new_session(model, expert_pairs, optimizer)
    elif len(_SESSION.pairs) != len(expert_pairs):
        raise RuntimeError("bootstrap expert trajectory count changed inside one CUDA session")
    return _SESSION


def _copy_epoch_back(session: _BootstrapCudaSession) -> None:
    cuda_bc._copy_packed_gru_back(session.cuda_model, session.sequence_gru)
    cpu_state = {
        key: value.detach().cpu().clone()
        for key, value in session.cuda_model.state_dict().items()
    }
    session.model.load_state_dict(cpu_state)


def _bootstrap_multi_epoch_cuda(
    model,
    expert_pairs,
    *,
    optimizer,
    chunk_steps: int,
    reverse_order: bool,
) -> float:
    if not expert_pairs:
        raise ValueError("at least one bootstrap expert trajectory is required")
    if chunk_steps <= 0:
        raise ValueError("chunk_steps must be positive")
    if not cuda_bc.cuda_requested():
        return _BASE_BOOTSTRAP_MULTI_EPOCH(
            model,
            expert_pairs,
            optimizer=optimizer,
            chunk_steps=chunk_steps,
            reverse_order=reverse_order,
        )

    session = _session_for(model, expert_pairs, optimizer)
    ordered = list(reversed(session.pairs)) if reverse_order else session.pairs
    total_chunks = sum(
        (int(observations.shape[0]) + int(chunk_steps) - 1) // int(chunk_steps)
        for observations, _ in ordered
    )
    emit_progress(
        "bc_start",
        sequences=len(ordered),
        chunks=total_chunks,
        reverse=bool(reverse_order),
    )

    parameters = cuda_bc._cuda_policy_parameters(session.cuda_model, session.sequence_gru)
    frame_count = 0
    weighted_loss = torch.zeros((), dtype=torch.float32, device=session.device)
    global_chunk = 0
    displayed_loss = 0.0
    started = time.perf_counter()

    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        for sequence_index, (observations, actions) in enumerate(ordered, 1):
            frames = int(observations.shape[0])
            sequence_chunks = (frames + int(chunk_steps) - 1) // int(chunk_steps)
            emit_progress(
                "bc_sequence_start",
                index=sequence_index,
                total=len(ordered),
                chunks=sequence_chunks,
                frames=frames,
                source=f"bootstrap-expert-{sequence_index}",
                reverse=bool(reverse_order),
            )

            state = session.cuda_model.initial_state(session.device)
            chunk_index = 0
            for start in range(0, frames, int(chunk_steps)):
                end = min(frames, start + int(chunk_steps))
                state = state.detach()
                means, _values, state = cuda_bc._forward_sequence_cuda(
                    session.cuda_model,
                    session.sequence_gru,
                    observations[start:end],
                    state,
                )
                predicted = torch.tanh(means)
                loss = v054._actuation_loss(predicted, actions[start:end])

                session.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                session.optimizer.step()

                chunk_frames = end - start
                weighted_loss.add_(loss.detach() * chunk_frames)
                frame_count += chunk_frames
                chunk_index += 1
                global_chunk += 1
                if global_chunk == 1 or global_chunk % 32 == 0 or global_chunk == total_chunks:
                    displayed_loss = float(loss.detach().item())
                emit_progress(
                    "bc_chunk",
                    sequence=sequence_index,
                    sequence_total=len(ordered),
                    chunk=chunk_index,
                    chunk_total=sequence_chunks,
                    global_chunk=global_chunk,
                    global_total=total_chunks,
                    loss=displayed_loss,
                    source=f"bootstrap-expert-{sequence_index}",
                    reverse=bool(reverse_order),
                )

            emit_progress(
                "bc_sequence_done",
                index=sequence_index,
                total=len(ordered),
                source=f"bootstrap-expert-{sequence_index}",
            )

        torch.cuda.synchronize(session.device)
        final_loss = float((weighted_loss / max(1, frame_count)).item())
        _copy_epoch_back(session)
    except Exception:
        # Once the persistent Adam state has advanced on CUDA, silently falling
        # back to the untouched CPU optimizer would change bootstrap semantics.
        # Surface the failure instead of continuing with mismatched moments.
        global _SESSION
        _SESSION = None
        raise
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32

    elapsed = time.perf_counter() - started
    parity = "rev" if reverse_order else "fwd"
    print(
        f"bootstrap-cuda: version={BOOTSTRAP_CUDA_VERSION} parity={parity} "
        f"total={elapsed:.2f}s chunks={global_chunk} device={torch.cuda.get_device_name(session.device)}"
    )
    emit_progress("bc_done", loss=final_loss, reverse=bool(reverse_order))
    return final_loss


def install_bootstrap_cuda_acceleration() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    v062._bootstrap_multi_epoch = _bootstrap_multi_epoch_cuda
    _INSTALLED = True
    backend = "cuda" if cuda_bc.cuda_requested() else "cpu-fallback"
    print(f"bootstrap-cuda={BOOTSTRAP_CUDA_VERSION} backend={backend} persistent-adam=on")
