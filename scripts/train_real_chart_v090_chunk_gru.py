from __future__ import annotations

"""Execution-only BC acceleration using batched GRU sequence kernels.

This keeps the existing sequence order, chunk boundaries, optimizer steps,
weighted loss, gradient clipping, progress events, and checkpoint semantics.
Only the per-frame Python ``forward_step`` loop inside each BC chunk is replaced
with ``RecurrentActorCritic.forward_sequence`` so PyTorch executes the recurrent
chunk in its native GRU kernel.
"""

import torch

import train_real_chart_v056 as v056
import train_real_chart_v057 as v057
from dmdod.training_progress import emit_progress


CHUNK_GRU_VERSION = "v090-batched-gru-chunks-v1"
_INSTALLED = False


def _batched_progress_train_one_epoch(
    model,
    sequences,
    *,
    optimizer,
    chunk_steps: int,
    reverse_order: bool,
) -> float:
    if not sequences:
        raise ValueError("at least one stable DAgger sequence is required")
    if chunk_steps <= 0:
        raise ValueError("chunk_steps must be positive")

    parameters = v057._policy_parameters(model)
    ordered = list(reversed(sequences)) if reverse_order else sequences
    total_chunks = sum(
        (int(stable.sequence.frames) + int(chunk_steps) - 1) // int(chunk_steps)
        for stable in ordered
    )
    emit_progress(
        "bc_start",
        sequences=len(ordered),
        chunks=total_chunks,
        reverse=bool(reverse_order),
    )

    loss_sum = 0.0
    weighted_elements = 0.0
    global_chunk = 0

    for sequence_index, stable in enumerate(ordered, 1):
        sequence = stable.sequence
        sequence_chunks = (int(sequence.frames) + int(chunk_steps) - 1) // int(chunk_steps)
        source = str(getattr(sequence, "source", f"sequence-{sequence_index}"))
        emit_progress(
            "bc_sequence_start",
            index=sequence_index,
            total=len(ordered),
            chunks=sequence_chunks,
            frames=int(sequence.frames),
            source=source,
            reverse=bool(reverse_order),
        )

        state = model.initial_state(sequence.observations.device)
        chunk_index = 0
        for start in range(0, sequence.frames, chunk_steps):
            end = min(sequence.frames, start + chunk_steps)
            state = state.detach()

            means, _values, state = model.forward_sequence(
                sequence.observations[start:end],
                state,
            )
            predicted = torch.tanh(means)
            target = sequence.teacher_actions[start:end]
            weights = stable.loss_weights[start:end]
            loss = v056._weighted_actuation_loss(predicted, target, weights)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()

            weight = max(float(weights.sum().item()), 1.0)
            loss_value = float(loss.detach().item())
            loss_sum += loss_value * weight
            weighted_elements += weight
            chunk_index += 1
            global_chunk += 1
            emit_progress(
                "bc_chunk",
                sequence=sequence_index,
                sequence_total=len(ordered),
                chunk=chunk_index,
                chunk_total=sequence_chunks,
                global_chunk=global_chunk,
                global_total=total_chunks,
                loss=loss_value,
                source=source,
                reverse=bool(reverse_order),
            )

        emit_progress(
            "bc_sequence_done",
            index=sequence_index,
            total=len(ordered),
            source=source,
        )

    final_loss = loss_sum / max(1.0, weighted_elements)
    emit_progress("bc_done", loss=final_loss, reverse=bool(reverse_order))
    return final_loss


def install_chunk_gru_acceleration() -> None:
    """Install the batched chunk path before v0.7 captures the BC implementation."""

    global _INSTALLED
    if _INSTALLED:
        return
    v057._train_one_epoch = _batched_progress_train_one_epoch
    _INSTALLED = True
    print(f"bc-chunk-gru={CHUNK_GRU_VERSION} kernel=torch._VF.gru progress=on")
