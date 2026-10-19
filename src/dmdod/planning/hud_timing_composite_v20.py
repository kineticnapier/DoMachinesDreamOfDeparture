from __future__ import annotations

"""Causal HUD v3 countdown + frozen Timing v1.4 composite."""

from dataclasses import dataclass, replace
from math import isfinite

from dmdod.features.countdown_hud_v3 import (
    CountdownHUDV3Frame,
    CountdownStudentObservation,
    HUD263_DIM,
    encode_countdown_features,
)
from dmdod.planning.countdown_first_event_v01 import (
    CountdownFirstEventEstimate,
    CountdownFirstEventEstimator,
)
from dmdod.planning.countdown_phase_tracker_v01 import (
    CountdownPhaseTracker,
    CountdownPhaseUpdate,
    TimeInterval,
)
from dmdod.planning.contact_timing_hud_v01 import MOD_VISIBLE_BPM_CONTRACT
from dmdod.planning.hud_timing_tracker_transition_v14 import (
    CausalTimingTracker,
    TimingUpdate,
)


COMPOSITE_VERSION = "hud-timing-composite-v2.0"
CONTROL_DT_S = .010
EPS = 1e-9


@dataclass(frozen=True, slots=True)
class TimingEventToken:
    local_id: str
    due_interval_s: TimeInterval
    first_seen_local_s: float
    source: str
    confirmations: int
    confirmed_now: bool
    lead_guaranteed: bool

    @property
    def due_local_s(self) -> float:
        return (
            self.due_interval_s.lower_s + self.due_interval_s.upper_s
        ) * .5

    @property
    def uncertainty_s(self) -> float:
        return self.due_interval_s.width_s * .5


@dataclass(frozen=True, slots=True)
class TimingFrameUpdateV20:
    step: int
    status: str
    pending: tuple[TimingEventToken, ...]
    elapsed: tuple[TimingEventToken, ...]
    canceled: tuple[TimingEventToken, ...]
    new_ids: tuple[str, ...]
    reused_ids: tuple[str, ...]
    abstention_reason: str | None
    countdown_phase: CountdownPhaseUpdate
    countdown_estimate: CountdownFirstEventEstimate
    v14_update: TimingUpdate
    version: str = COMPOSITE_VERSION


class HUDTimingComposite:
    """Keep both estimators independent and reconcile only public forecasts."""

    def __init__(self, *, bpm_observation_contract: str,
                 horizon_s: float = .080,
                 v14_tracker: CausalTimingTracker | None = None,
                 reconcile_countdown_v14_identity: bool = True) -> None:
        if bpm_observation_contract != MOD_VISIBLE_BPM_CONTRACT:
            raise ValueError(
                "HUDTimingComposite requires numeric Tile/Real BPM actually "
                "displayed by the Mod-visible HUD contract; stock-only input "
                "cannot treat chart BPM as an observation"
            )
        if v14_tracker is not None and not isinstance(
            v14_tracker, CausalTimingTracker
        ):
            raise TypeError("v14_tracker must be CausalTimingTracker")
        if type(reconcile_countdown_v14_identity) is not bool:
            raise TypeError("reconcile_countdown_v14_identity must be bool")
        self._v14 = v14_tracker or CausalTimingTracker(horizon_s=horizon_s)
        self._reconcile_countdown_v14_identity = (
            reconcile_countdown_v14_identity
        )
        self._phase = CountdownPhaseTracker()
        self._countdown = CountdownFirstEventEstimator()
        self.reset()

    def reset(self) -> None:
        self._v14.reset()
        self._phase.reset()
        self._countdown.reset()
        self._next_step = 0
        self._countdown_token: TimingEventToken | None = None
        self._countdown_v14_alias: str | None = None
        self._countdown_closed = False
        self._countdown_context_seen = False
        self._previous_pending: dict[str, TimingEventToken] = {}

    @staticmethod
    def _validate_boundary(
        frame: CountdownHUDV3Frame,
        observation: CountdownStudentObservation,
    ) -> tuple[float, ...]:
        if not isinstance(frame, CountdownHUDV3Frame):
            raise TypeError("expected CountdownHUDV3Frame")
        if not isinstance(observation, CountdownStudentObservation):
            raise TypeError("expected CountdownStudentObservation")
        if frame.step != observation.step:
            raise ValueError("HUD and countdown observation step mismatch")
        if abs(frame.observed_at_s - observation.observed_at_s) > EPS:
            raise ValueError("HUD and countdown observation time mismatch")
        if frame.features[HUD263_DIM:] != encode_countdown_features(observation):
            raise ValueError("HUD v3 countdown suffix does not match observation")
        return frame.legacy_hud263

    @staticmethod
    def _v14_token(event, now_s: float) -> TimingEventToken:
        return TimingEventToken(
            local_id=f"v14:{event.local_id}",
            due_interval_s=TimeInterval(event.due_s, event.due_s),
            first_seen_local_s=now_s,
            source="timing_v1.4",
            confirmations=event.confirmations,
            confirmed_now=event.confirmed_now,
            lead_guaranteed=event.due_s - now_s + EPS >= .044,
        )

    def _update_countdown_token(
        self, estimate: CountdownFirstEventEstimate, now_s: float
    ) -> str | None:
        if self._countdown_closed:
            return None
        if not estimate.eligible:
            if (
                self._countdown_token is not None
                and self._countdown_token.due_interval_s.upper_s > now_s + EPS
            ):
                self._countdown_token = replace(
                    self._countdown_token, confirmed_now=False
                )
            return None
        if self._countdown_token is None:
            self._countdown_token = TimingEventToken(
                local_id="countdown:0",
                due_interval_s=estimate.due_interval_s,
                first_seen_local_s=now_s,
                source="countdown_first_event",
                confirmations=1,
                confirmed_now=True,
                lead_guaranteed=estimate.lead_guaranteed,
            )
        else:
            prior = self._countdown_token.due_interval_s
            current = estimate.due_interval_s
            if not self._overlaps(prior, current):
                self._countdown_token = None
                self._countdown_closed = True
                return "countdown_interval_conflict"
            narrowed = TimeInterval(
                *self._intersection_bounds(prior, current),
            )
            self._countdown_token = replace(
                self._countdown_token,
                due_interval_s=narrowed,
                confirmations=self._countdown_token.confirmations + 1,
                confirmed_now=True,
                lead_guaranteed=estimate.lead_guaranteed,
            )
        return None

    @staticmethod
    def _overlaps(left: TimeInterval, right: TimeInterval) -> bool:
        return (
            max(left.lower_s, right.lower_s)
            <= min(left.upper_s, right.upper_s) + EPS
        )

    @staticmethod
    def _intersection_bounds(
        left: TimeInterval, right: TimeInterval
    ) -> tuple[float, float]:
        lower = max(left.lower_s, right.lower_s)
        upper = min(left.upper_s, right.upper_s)
        if lower > upper:
            # _overlaps permits only composite-scale floating-point roundoff.
            lower = upper = (lower + upper) * .5
        return lower, upper

    def observe(
        self, frame: CountdownHUDV3Frame,
        countdown_observation: CountdownStudentObservation,
    ) -> TimingFrameUpdateV20:
        if frame.step != self._next_step:
            raise ValueError("composite frames must be consecutive")
        expected_time = frame.step * CONTROL_DT_S
        if abs(frame.observed_at_s - expected_time) > EPS:
            raise ValueError("composite requires the frozen 10ms local clock")
        legacy = self._validate_boundary(frame, countdown_observation)

        v14 = self._v14.observe(frame.step, legacy)
        phase = self._phase.observe(countdown_observation)
        estimate = self._countdown.observe(phase, legacy)
        now_s = expected_time
        if phase.status in ("ready", "number", "go"):
            self._countdown_context_seen = True
        countdown_conflict = self._update_countdown_token(estimate, now_s)

        elapsed: list[TimingEventToken] = []
        if (
            self._countdown_token is not None
            and self._countdown_token.due_interval_s.upper_s <= now_s + EPS
        ):
            elapsed.append(self._countdown_token)
            self._countdown_token = None
            self._countdown_v14_alias = None
            self._countdown_closed = True
            self._countdown_context_seen = False

        # Visible Ready/number means these orbit cycles are countdown motion,
        # not playable floor hits. Keep the frozen raw update for diagnostics,
        # but do not publish it as a playable candidate until Go/normal play.
        suppress_v14_countdown_motion = (
            phase.status in ("ready", "number")
            or (not phase.valid and self._countdown_context_seen)
        )
        v14_tokens = (
            [] if suppress_v14_countdown_motion else
            [self._v14_token(item, now_s) for item in v14.pending]
        )
        pending: list[TimingEventToken] = []
        conflict = countdown_conflict is not None
        countdown_token = self._countdown_token
        if countdown_token is not None:
            aliased = [
                item for item in v14_tokens
                if item.local_id == self._countdown_v14_alias
            ]
            nonaliased = [
                item for item in v14_tokens if item not in aliased
            ]
            if (
                self._reconcile_countdown_v14_identity
                and self._countdown_v14_alias is not None
                and len(aliased) == 1
                and not nonaliased
            ):
                # Identity was established on an earlier frame by one unique
                # interval overlap. The frozen v1.4 tracker guarantees that a
                # reused local ID denotes that same observed event. Preserve
                # the already published Countdown interval: point-estimate
                # drift is not evidence for shrinking or retiming it.
                countdown_token = replace(
                    countdown_token,
                    source="countdown_first_event+timing_v1.4",
                    confirmations=(
                        countdown_token.confirmations
                        + aliased[0].confirmations
                    ),
                    confirmed_now=True,
                )
                self._countdown_token = countdown_token
                nonoverlaps = []
            else:
                overlaps = [
                    item for item in v14_tokens
                    if self._overlaps(countdown_token.due_interval_s,
                                      item.due_interval_s)
                ]
                nonoverlaps = [
                    item for item in v14_tokens if item not in overlaps
                ]
                if aliased:
                    # A second visible v1.4 candidate appeared while the
                    # established alias was live. Keep the conservative
                    # ambiguity response; do not infer event order here.
                    conflict = True
                    nonoverlaps = []
            if (
                not aliased
                and len(overlaps) == 1
                and not nonoverlaps
            ):
                lower = max(
                    [countdown_token.due_interval_s.lower_s]
                    + [item.due_interval_s.lower_s for item in overlaps]
                )
                upper = min(
                    [countdown_token.due_interval_s.upper_s]
                    + [item.due_interval_s.upper_s for item in overlaps]
                )
                if lower > upper:
                    # Both sources agree within the composite reconciliation
                    # epsilon; collapse floating-point roundoff to one point.
                    lower = upper = (lower + upper) * .5
                countdown_token = replace(
                    countdown_token,
                    due_interval_s=TimeInterval(lower, upper),
                    source="countdown_first_event+timing_v1.4",
                    confirmations=(
                        countdown_token.confirmations
                        + sum(item.confirmations for item in overlaps)
                    ),
                    confirmed_now=True,
                )
                self._countdown_token = countdown_token
                if self._reconcile_countdown_v14_identity:
                    self._countdown_v14_alias = overlaps[0].local_id
            elif v14_tokens:
                if not aliased:
                    conflict = True
            if not conflict:
                pending.append(countdown_token)
                pending.extend(nonoverlaps)
        else:
            pending.extend(v14_tokens)

        previous = self._previous_pending
        elapsed_ids = {item.local_id for item in elapsed}
        # Translate v1.4 elapsed tokens without changing its raw update.
        for event in v14.elapsed:
            token_id = f"v14:{event.local_id}"
            if token_id in previous and token_id not in elapsed_ids:
                elapsed.append(previous[token_id])
                elapsed_ids.add(token_id)

        countdown_active = (
            countdown_token is not None
            or phase.status in ("ready", "number", "go")
        )
        if conflict:
            status = countdown_conflict or "source_conflict"
            abstention = (
                countdown_conflict
                or "countdown_and_v14_candidates_are_ambiguous_or_disjoint"
            )
            pending = []
            self._countdown_token = None
            self._countdown_v14_alias = None
            self._countdown_closed = True
            self._countdown_context_seen = False
        elif countdown_active:
            status = estimate.status
            abstention = None if estimate.eligible else estimate.status
        else:
            # Exact normal-play status compatibility is intentionally exposed.
            status = v14.status
            abstention = None if v14.pending else v14.status

        current = {item.local_id: item for item in pending}
        new_ids = tuple(sorted(set(current) - set(previous)))
        reused_ids = tuple(sorted(set(current) & set(previous)))
        canceled = [
            item for key, item in previous.items()
            if key not in current and key not in elapsed_ids
        ]
        canceled = [
            item for item in canceled
            if item.local_id not in {x.local_id for x in elapsed}
        ]

        self._previous_pending = current
        self._next_step += 1
        return TimingFrameUpdateV20(
            step=frame.step, status=status,
            pending=tuple(sorted(pending, key=lambda x: x.due_local_s)),
            elapsed=tuple(elapsed), canceled=tuple(canceled),
            new_ids=new_ids, reused_ids=reused_ids,
            abstention_reason=abstention,
            countdown_phase=phase, countdown_estimate=estimate,
            v14_update=v14,
        )
