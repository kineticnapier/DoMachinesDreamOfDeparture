from __future__ import annotations

"""Opt-in v1.5 lifecycle reconciliation for quantized control frames."""

from typing import Callable

from dmdod.planning.hud_timing_tracker_transition_v14 import (
    CausalTimingTracker,
    WindowForecast,
)


class QuantizedLifecycleCausalTimingTracker(CausalTimingTracker):
    """Finalize near-boundary events instead of canceling accurate IDs.

    When a new eligible HUD forecast has visibly advanced past an event whose
    due is within the existing identity-match tolerance, classify that token
    as elapsed rather than canceled.  Abstentions cannot trigger the rule. It
    changes only bookkeeping; it never alters a due time or creates an event.
    """

    def __init__(self, *, horizon_s: float = .030,
                 control_dt_s: float = .010,
                 match_tolerance_s: float = .00075,
                 predictor: Callable[[tuple[float, ...]], WindowForecast] | None = None) -> None:
        super().__init__(
            horizon_s=horizon_s,
            control_dt_s=control_dt_s,
            match_tolerance_s=match_tolerance_s,
            predictor=predictor,
            elapsed_tolerance_s=match_tolerance_s,
        )
