from __future__ import annotations

"""Development-selected confidence gate for hit-epoch JRP Timing."""

from dmdod.planning.jrp_cbpm_anchor_v2 import CausalJrpTimingBridgeV2


VERSION='stage-t-jrp-hit-epoch-confidence-v3'
MIN_CONFIDENCE=.5


class CausalJrpTimingBridgeV3(CausalJrpTimingBridgeV2):
    """Accept only the first half of the causal two-frame lag window."""

    def __init__(self, **kwargs):
        if 'min_confidence' in kwargs:
            raise ValueError('v3 confidence threshold is frozen')
        super().__init__(min_confidence=MIN_CONFIDENCE,**kwargs)
