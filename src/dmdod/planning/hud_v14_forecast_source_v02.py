from __future__ import annotations

"""Causal Timing v1.4 -> versioned forecast adapter.

The original v1.4 tracker internally retires expired predictions even while
abstaining. This layer conservatively keeps prior local forecast tokens in
the outward contract (unconfirmed) until an eligible update can explicitly
retire them. It never uses target ordinals, chart time, or evaluator state.
"""

from dataclasses import replace

from dmdod.planning import hud_timing_tracker_transition_v14 as v14
from dmdod.planning.contact_timing_hud_v01 import (
    ContactTimingHUD263Adapter, ContactTimingHUD263Frame,
)
from dmdod.planning.timing_fingering_contract_v02 import (
    ABSTAIN, CONTRACT_VERSION, ELIGIBLE, ELIGIBLE_PARTIAL,
    TimingEventToken, TimingFrameSequenceValidator, TimingFrameUpdate,
)

SOURCE = "hud-timing-v1.4-contact263"
DEFAULT_HORIZON_S = .080


class HUDV14ForecastSource:
    """A nonprivileged forecast stream; v1.4's event tokens carry no chord truth.

    v1.4 currently predicts ordinary one-contact events. It abstains around
    MultiPlanet/midspin and cannot invent an unseen chord size.
    """

    def __init__(self, *, horizon_s: float = DEFAULT_HORIZON_S) -> None:
        self.hud = ContactTimingHUD263Adapter(horizon_s=horizon_s)
        self.horizon_s = horizon_s
        self._previous_hud: tuple[float, ...] | None = None
        self._live: dict[int, TimingEventToken] = {}
        self._validator = TimingFrameSequenceValidator()

    def reset(self) -> None:
        self.hud.reset()
        self._previous_hud = None
        self._live.clear()
        self._validator = TimingFrameSequenceValidator()

    def observe(self, frame: ContactTimingHUD263Frame) -> TimingFrameUpdate:
        """Emit exactly one monotone local step; verify status before commit."""
        if self._validator.last_step is not None and frame.step != self._validator.last_step + 1:
            raise ValueError("nonconsecutive HUD-only Timing step")
        # This is the same deterministic forecast used by CausalTimingTracker;
        # compute only the verified coverage (not chart/evaluator state).
        forecast = v14.forecast_visible_window(
            frame.features, horizon_s=self.horizon_s,
            previous_hud=self._previous_hud,
        )
        old = dict(self._live)
        raw = self.hud.observe(frame)
        if raw.status != forecast.status:
            raise RuntimeError("v1.4 tracker/forecast coverage mismatch")
        now = frame.step * .010
        if raw.status in (ELIGIBLE, ELIGIBLE_PARTIAL):
            tokens = tuple(
                TimingEventToken(
                    local_id=item.local_id, due_local_s=item.due_s,
                    chord_size=1, confirmations=item.confirmations,
                    confirmed_now=item.confirmed_now,
                ) for item in raw.pending
            )
            current_ids = {x.local_id for x in tokens}
            released = set(old) - current_ids
            canceled = tuple(sorted(
                x for x in released if old[x].due_local_s > now + v14.EPS
            ))
            elapsed = tuple(sorted(x for x in released if x not in canceled))
            covered_until = now + (
                self.horizon_s if forecast.covered_until_s is None
                else forecast.covered_until_s
            )
            status = raw.status
            reason = None
        else:
            # v1.4's raw expired IDs are *not* a claim that the physically
            # commanded finger reset. Preserve these tokens across abstention.
            tokens = tuple(replace(x, confirmed_now=False) for x in old.values())
            current_ids = set(old)
            canceled, elapsed = (), ()
            covered_until = now
            status = ABSTAIN
            reason = raw.status

        tokens = tuple(sorted(tokens, key=lambda x: (x.due_local_s, x.local_id)))
        new_ids = tuple(sorted(current_ids - set(old)))
        reused_ids = tuple(sorted(current_ids & set(old)))
        update = TimingFrameUpdate(
            step=frame.step, status=status, covered_until_s=covered_until,
            pending=tokens, source=SOURCE, version=CONTRACT_VERSION,
            new_ids=new_ids, reused_ids=reused_ids,
            canceled_ids=canceled, elapsed_ids=elapsed,
            abstention_reason=reason,
        )
        self._validator.accept(update)
        self._live = {x.local_id: x for x in tokens}
        self._previous_hud = frame.features
        return update
