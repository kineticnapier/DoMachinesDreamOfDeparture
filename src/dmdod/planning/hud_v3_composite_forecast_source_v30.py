from __future__ import annotations

"""HUD v3 composite -> frozen Timing/Fingering contract adapter.

The adapter is deliberately truth-free. Composite string IDs are mapped once
to monotonically increasing integer IDs. The midpoint and half-width preserve
the complete due interval in the existing TimingEventToken representation.
"""

from dataclasses import replace

from dmdod.features.countdown_hud_v3 import (
    CountdownStudentObservation,
    encode_hud_v3,
)
from dmdod.planning.contact_timing_hud_v01 import (
    ContactTimingHUD263Adapter,
    ContactVisibleTimingHUD,
    MOD_VISIBLE_BPM_CONTRACT,
)
from dmdod.planning.hud_timing_composite_v20 import (
    HUDTimingComposite,
    TimingFrameUpdateV20,
)
from dmdod.planning.hud_timing_tracker_lifecycle_v15 import (
    QuantizedLifecycleCausalTimingTracker,
)
from dmdod.planning.timing_fingering_contract_v02 import (
    ABSTAIN,
    CONTRACT_VERSION,
    ELIGIBLE,
    ELIGIBLE_PARTIAL,
    TimingEventToken,
    TimingFrameSequenceValidator,
    TimingFrameUpdate,
)


SOURCE = "hud-v3-countdown-composite-contact278"
LIFECYCLE_V14 = "v1.4"
LIFECYCLE_V15 = "v1.5"
LIFECYCLE_VERSIONS = frozenset((LIFECYCLE_V14, LIFECYCLE_V15))
ELIGIBLE_COMPOSITE_STATUSES = frozenset((
    "eligible",
    "eligible_partial",
    "eligible_lead_guaranteed",
    "eligible_insufficient_lead",
))


class HUDTimingCompositeAdapter:
    """Translate one verified composite stream without target matching."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._next_step = 0
        self._next_integer_id = 0
        self._id_map: dict[str, int] = {}
        self._live: set[str] = set()
        self._validator = TimingFrameSequenceValidator()

    @property
    def id_map(self) -> dict[str, int]:
        return dict(self._id_map)

    def _integer_id(self, local_id: str, *, new: bool) -> int:
        if type(local_id) is not str or not local_id:
            raise ValueError("composite local ID must be nonempty text")
        if new:
            if local_id in self._id_map:
                raise ValueError("composite local ID cannot be created twice")
            self._id_map[local_id] = self._next_integer_id
            self._next_integer_id += 1
        elif local_id not in self._id_map:
            raise ValueError("unknown reused/retired composite local ID")
        return self._id_map[local_id]

    def observe(self, composite: TimingFrameUpdateV20) -> TimingFrameUpdate:
        if not isinstance(composite, TimingFrameUpdateV20):
            raise TypeError("expected TimingFrameUpdateV20")
        if composite.step != self._next_step:
            raise ValueError("composite adapter steps must be consecutive")

        pending_by_id = {item.local_id: item for item in composite.pending}
        if len(pending_by_id) != len(composite.pending):
            raise ValueError("duplicate composite pending local ID")
        pending_ids = set(pending_by_id)
        new = set(composite.new_ids)
        reused = set(composite.reused_ids)
        canceled = {item.local_id for item in composite.canceled}
        elapsed = {item.local_id for item in composite.elapsed}
        if new & reused or canceled & elapsed:
            raise ValueError("overlapping composite lifecycle groups")
        if new | reused != pending_ids:
            raise ValueError("composite new/reused IDs must partition pending")
        if reused - self._live:
            raise ValueError("composite reused ID is not live")
        if (canceled | elapsed) - self._live:
            raise ValueError("composite retired ID is not live")
        if self._live - reused != canceled | elapsed:
            raise ValueError("composite ID disappeared without retirement")

        for local_id in sorted(new):
            self._integer_id(local_id, new=True)
        for local_id in sorted(reused | canceled | elapsed):
            self._integer_id(local_id, new=False)

        tokens = tuple(sorted((
            TimingEventToken(
                local_id=self._id_map[item.local_id],
                due_local_s=item.due_local_s,
                chord_size=1,
                confirmations=item.confirmations,
                confirmed_now=item.confirmed_now,
                confidence=None,
                uncertainty_s=item.uncertainty_s,
            )
            for item in composite.pending
        ), key=lambda item: (item.due_local_s, item.local_id)))

        has_lifecycle_change = bool(new or canceled or elapsed)
        raw_eligible = composite.status in ELIGIBLE_COMPOSITE_STATUSES
        if raw_eligible:
            status = (
                ELIGIBLE_PARTIAL
                if composite.status == "eligible_partial" else ELIGIBLE
            )
            reason = None
        elif not has_lifecycle_change:
            status = ABSTAIN
            reason = composite.abstention_reason or composite.status
            tokens = tuple(replace(item, confirmed_now=False) for item in tokens)
        else:
            # Existing contract forbids lifecycle mutation on ABSTAIN. Publish
            # the explicit retirement as partial coverage, never as eligibility
            # for a new/reconfirmed event.
            status = ELIGIBLE_PARTIAL
            reason = composite.abstention_reason or composite.status
            tokens = tuple(replace(item, confirmed_now=False) for item in tokens)

        now_s = composite.step * .010
        covered_until_s = max(
            (item.due_interval_s.upper_s for item in composite.pending),
            default=now_s,
        )
        update = TimingFrameUpdate(
            step=composite.step,
            status=status,
            covered_until_s=covered_until_s,
            pending=tokens,
            source=SOURCE,
            version=CONTRACT_VERSION,
            new_ids=tuple(sorted(self._id_map[item] for item in new)),
            reused_ids=tuple(sorted(self._id_map[item] for item in reused)),
            canceled_ids=tuple(sorted(self._id_map[item] for item in canceled)),
            elapsed_ids=tuple(sorted(self._id_map[item] for item in elapsed)),
            abstention_reason=reason,
        )
        self._validator.accept(update)
        self._live = pending_ids
        self._next_step += 1
        return update


class HUDV3CompositeForecastSource:
    """Build HUD278 from current visible fields and expose motor contract."""

    def __init__(self, *, horizon_s: float = .080,
                 lifecycle_version: str = LIFECYCLE_V14,
                 reconcile_countdown_v14_identity: bool = True) -> None:
        if lifecycle_version not in LIFECYCLE_VERSIONS:
            raise ValueError(
                f"unsupported lifecycle version: {lifecycle_version!r}"
            )
        tracker = (
            None if lifecycle_version == LIFECYCLE_V14
            else QuantizedLifecycleCausalTimingTracker(horizon_s=horizon_s)
        )
        self.composite = HUDTimingComposite(
            bpm_observation_contract=MOD_VISIBLE_BPM_CONTRACT,
            horizon_s=horizon_s,
            v14_tracker=tracker,
            reconcile_countdown_v14_identity=(
                reconcile_countdown_v14_identity
            ),
        )
        self.adapter = HUDTimingCompositeAdapter()
        self.lifecycle_version = lifecycle_version

    def reset(self) -> None:
        self.composite.reset()
        self.adapter.reset()

    def observe(
        self,
        step: int,
        visible_hud: ContactVisibleTimingHUD,
        countdown: CountdownStudentObservation,
    ) -> tuple[TimingFrameUpdate, TimingFrameUpdateV20]:
        legacy = ContactTimingHUD263Adapter.encode(step, visible_hud).features
        raw = self.composite.observe(encode_hud_v3(legacy, countdown), countdown)
        return self.adapter.observe(raw), raw
