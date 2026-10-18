from __future__ import annotations

"""Adapt truth-free TimingFrameUpdate to frozen 65D Fingering Network context.

Network v0.1 expects a lightweight object with episode_time_s; in this
versioned adapter that property is an ALIAS of the token's local clock due,
not the scorer's episode/target clock. No Motor observations are synthesized.
"""

from dataclasses import dataclass

import torch

from dmdod.motor.contact_16k import Coupled16ContactMotor
from dmdod.planning.timing_fingering_contract_v02 import TimingFrameUpdate
from dmdod.training import fingering_network_v01 as network


@dataclass(frozen=True, slots=True)
class _LocalLookahead:
    local_id: int
    episode_time_s: float  # pure local-policy clock alias for frozen encoder
    chord_size: int


def encode_context_from_timing_update(
    motor: Coupled16ContactMotor,
    update: TimingFrameUpdate,
    occupied: set[str] | frozenset[str],
    *,
    starting_at_local_id: int | None = None,
) -> torch.Tensor:
    """Exactly preserve network.encode_context's existing feature layout.

    Treat the current and any future local forecast events as ordinary
    lookahead tokens. Unconfirmed events remain visible to the encoder but
    can never create a new reservation unless confirmed separately.
    """
    if not isinstance(update, TimingFrameUpdate):
        raise TypeError("expected versioned TimingFrameUpdate")
    pending = update.pending
    if starting_at_local_id is not None:
        chosen = next((x for x in pending
                       if x.local_id == starting_at_local_id), None)
        if chosen is None:
            raise KeyError(starting_at_local_id)
        pending = tuple(x for x in pending
                        if x.due_local_s >= chosen.due_local_s)
    look = tuple(_LocalLookahead(
        local_id=x.local_id,
        episode_time_s=x.due_local_s,
        chord_size=x.chord_size,
    ) for x in pending)
    return network.encode_context(
        motor, look, update.local_clock_s, set(occupied)
    )
