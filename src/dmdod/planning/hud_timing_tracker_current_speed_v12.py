from __future__ import annotations

"""Experimental stateful, causal HUD-only event tracker (not a Student policy).

Uses no target IDs, chart time, precompiled truth, hidden direction, or motor
commands. The stream clock counts 10ms HUD frames from the first observation.
Only *visible* ordinary-floor geometry is supported; all other cases abstain.
"""

from dataclasses import dataclass
from math import atan2, isfinite, pi
from typing import Callable

from dmdod.features.real_chart import (
    DEFAULT_REAL_CHART_FEATURE_CONFIG as FLOORS, REAL_CHART_FLOOR_FEATURE_DIM,
)
from dmdod.features.hud_features import HUD_BPM_REFERENCE

HUD_DIM = 263
MOTOR_FEATURES = 24
SLOT_START = MOTOR_FEATURES + 2
CURRENT_SLOT = FLOORS.behind_floors
FLOOR_DIM = REAL_CHART_FLOOR_FEATURE_DIM
HUD_OFFSET = SLOT_START + FLOORS.floor_slots * FLOOR_DIM
SKIP_MARKER_FROM, SKIP_MARKER_TO = 8, 14


@dataclass(frozen=True)
class WindowForecast:
    floor_offsets: tuple[int, ...]
    times_s: tuple[float, ...]
    status: str
    covered_until_s: float | None = None

    @property
    def eligible(self) -> bool:
        return self.status in ("eligible", "eligible_partial")


def visible_slot(hud: tuple[float, ...], slot: int) -> tuple[float, ...]:
    start = SLOT_START + slot*FLOOR_DIM
    return hud[start:start+FLOOR_DIM]

MIN_HORIZON_S = .010
MAX_HORIZON_S = .080
ANG_EPS = 1e-5
EPS = 1e-7


def _direction(hud: tuple[float, ...], sign: int, bpm: float, horizon_s: float) -> WindowForecast:
    orbit = atan2(hud[MOTOR_FEATURES], hud[MOTOR_FEATURES + 1])
    speed = pi * bpm / 60.
    elapsed = 0.
    offsets: list[int] = []
    times: list[float] = []
    for slot in range(CURRENT_SLOT, CURRENT_SLOT + FLOORS.ahead_floors + 1):
        floor = visible_slot(hud, slot)
        if floor[-1] < .5:
            return WindowForecast((), (), "insufficient_lookahead")
        relative = slot-CURRENT_SLOT
        if floor[6] > .5 or floor[7] > .5:
            return WindowForecast((), (), "midspin_or_multiplanet")
        markers = [j for j in range(SKIP_MARKER_FROM, SKIP_MARKER_TO)
                   if floor[j] > .5]
        if any(j not in (SKIP_MARKER_FROM, SKIP_MARKER_FROM+1) for j in markers) or (relative > 0 and markers):
            return WindowForecast((), (), "special_event")
        entry = atan2(floor[2], floor[3])
        exit_angle = atan2(floor[4], floor[5])
        arc = (sign*(exit_angle-entry))%(2*pi)
        if arc < ANG_EPS or 2*pi-arc < ANG_EPS:
            return WindowForecast((), (), "full_turn_or_zero_arc")
        if relative == 0:
            progress = (sign*(orbit-entry))%(2*pi)
            if progress < ANG_EPS or 2*pi-progress < ANG_EPS:
                progress = 0.
            if progress > arc + ANG_EPS:
                return WindowForecast((), (), "incompatible_direction")
            if arc-progress < ANG_EPS:
                # The current floor ends at this control-frame boundary.
                # Its exit event is not future (>0), so predict only exits
                # of subsequent visibly ordinary floors. Both direction
                # hypotheses must still agree on their future schedule.
                if abs(horizon_s - MIN_HORIZON_S) < 1e-12:
                    return WindowForecast((), (), "boundary_ambiguous")
                continue
            remaining = max(0., arc-progress)
        else:
            remaining = arc
        until = remaining/speed
        if elapsed+until > horizon_s+EPS:
            return WindowForecast(tuple(offsets), tuple(times), "eligible")
        next_slot = slot+1
        next_floor = visible_slot(hud, next_slot) if next_slot < FLOORS.floor_slots else ()
        if not next_floor or next_floor[-1] < .5:
            return WindowForecast((), (), "insufficient_lookahead")
        next_is_special = (any(next_floor[j] > .5 for j in (6, 7)) or any(
            next_floor[j] > .5 for j in range(SKIP_MARKER_FROM, SKIP_MARKER_TO)
        ))
        # SetSpeed/Pause may change the arrival time itself. Do not predict
        # across these transitions using the current Tile BPM.
        if any(next_floor[j] > .5 for j in (6, 7, 9, 10, 11, 12, 13)):
            return WindowForecast((), (), "special_future_floor")
        elapsed += until
        if elapsed <= EPS:
            return WindowForecast((), (), "boundary_ambiguous")
        offsets.append(relative+1)
        times.append(elapsed)
        if next_is_special:
            # An incoming Twirl note has a predictable arrival time, but
            # the new direction must be re-observed before planning beyond.
            return WindowForecast(tuple(offsets), tuple(times), "eligible_partial", elapsed)
        if elapsed >= horizon_s-EPS:
            return WindowForecast(tuple(offsets), tuple(times), "eligible")
    return WindowForecast((), (), "insufficient_lookahead")


def forecast_visible_window(hud: tuple[float, ...], *, horizon_s: float = .030) -> WindowForecast:
    """Forecast ordered events from *one* 263D HUD; reject ambiguous directions.

    10ms predictions preserve the original Stage T boundary abstention.
    Longer horizons can skip an event precisely at the current boundary.
    """
    if not MIN_HORIZON_S <= horizon_s <= MAX_HORIZON_S or not isfinite(horizon_s):
        raise ValueError("horizon must be finite and between 10 and 80ms")
    if len(hud) != HUD_DIM:
        return WindowForecast((), (), "invalid_hud_dimension")
    if not all(isfinite(v) for v in hud):
        return WindowForecast((), (), "nonfinite_hud")
    try:
        bpm = HUD_BPM_REFERENCE*(2.**hud[HUD_OFFSET])
    except OverflowError:
        return WindowForecast((), (), "invalid_bpm")
    if not isfinite(bpm) or bpm <= 0:
        return WindowForecast((), (), "invalid_bpm")
    if max(abs(v) for v in hud[MOTOR_FEATURES:MOTOR_FEATURES+2]) < .5:
        return WindowForecast((), (), "invalid_orbit")
    candidates = [_direction(hud, sign, bpm, horizon_s) for sign in (1, -1)]
    viable = [c for c in candidates if c.eligible]
    if not viable:
        reasons = [c.status for c in candidates]
        for reason in ("special_future_floor", "special_event", "midspin_or_multiplanet",
                       "boundary_ambiguous", "full_turn_or_zero_arc", "insufficient_lookahead"):
            if reason in reasons:
                return WindowForecast((), (), reason)
        return WindowForecast((), (), "no_consistent_direction")
    if len(viable) == 1:
        other = next(c for c in candidates if not c.eligible)
        return viable[0] if other.status == "incompatible_direction" else WindowForecast((), (), other.status)
    a, b = viable
    if a.floor_offsets != b.floor_offsets or len(a.times_s) != len(b.times_s) or any(
        abs(x-y) > EPS for x, y in zip(a.times_s, b.times_s)
    ):
        return WindowForecast((), (), "ambiguous_direction")
    return min((a,b), key=lambda x: x.covered_until_s if x.covered_until_s is not None else horizon_s)


@dataclass(frozen=True)
class TrackedEvent:
    local_id: int  # Own monotonically increasing ID; NOT a chart floor/target ID.
    due_s: float   # Relative to frame zero, not a privileged chart clock.
    confirmations: int
    confirmed_now: bool


@dataclass(frozen=True)
class TimingUpdate:
    frame: int
    status: str
    pending: tuple[TrackedEvent, ...]
    elapsed: tuple[TrackedEvent, ...]
    canceled: tuple[TrackedEvent, ...]
    new_ids: tuple[int, ...]
    reused_ids: tuple[int, ...]


class CausalTimingTracker:
    """Reconcile overlapping HUD predictions without true note identity.

    An eligible forecast is trusted only within its visible horizon. A later
    eligible forecast can withdraw a stale forecast; an abstention adds no
    events and marks carried predictions unconfirmed. Elapsed means *time
    passed*, not a physical hit or a proven floor transition.
    """

    def __init__(self, *, horizon_s: float = .030, control_dt_s: float = .010,
                 match_tolerance_s: float = .00075,
                 predictor: Callable[[tuple[float, ...]], WindowForecast] | None = None):
        if not MIN_HORIZON_S <= horizon_s <= MAX_HORIZON_S:
            raise ValueError("unsupported horizon")
        if abs(control_dt_s-.010) > 1e-12:
            raise ValueError("current HUD observation contract requires 10ms ticks")
        if not 0 <= match_tolerance_s <= .002:
            raise ValueError("invalid match tolerance")
        self.horizon_s = horizon_s
        self.dt = control_dt_s
        self.tolerance = match_tolerance_s
        self.predict = predictor or (lambda hud: forecast_visible_window(hud, horizon_s=horizon_s))
        self.reset()

    def reset(self):
        self._next_frame = 0
        self._next_id = 0
        self._pending: list[TrackedEvent] = []

    def observe(self, frame: int, hud: tuple[float, ...]) -> TimingUpdate:
        if type(frame) is not int or frame != self._next_frame:
            raise ValueError("HUD frames must be sequential, starting at zero")
        # No private truth is used here; relative time is just the number of
        # observed 10ms control frames.
        now = frame*self.dt
        prediction = self.predict(hud)
        if prediction.eligible and (
            len(prediction.floor_offsets) != len(prediction.times_s)
            or any(not isfinite(t) or t <= 0 or t > self.horizon_s+EPS
                   for t in prediction.times_s)
            or (prediction.status == "eligible_partial" and
                (not prediction.times_s or prediction.covered_until_s is None or
                 not isfinite(prediction.covered_until_s) or
                 abs(prediction.covered_until_s-prediction.times_s[-1]) > EPS))
            or any(b <= a for a, b in zip(prediction.times_s, prediction.times_s[1:]))
            or any(b <= a for a, b in zip(prediction.floor_offsets, prediction.floor_offsets[1:]))
        ):
            raise ValueError("malformed eligible forecast")
        elapsed = [x for x in self._pending if x.due_s <= now+EPS]
        survivors = [x for x in self._pending if x.due_s > now+EPS]
        canceled: list[TrackedEvent] = []
        reused: list[int] = []
        created: list[int] = []
        if prediction.eligible:
            available = list(survivors)
            pending: list[TrackedEvent] = []
            for delta in prediction.times_s:
                due = now+delta
                close = [p for p in available if abs(p.due_s-due) <= self.tolerance]
                if close:
                    # Avoid silently collapsing two equally plausible events.
                    close.sort(key=lambda p: (abs(p.due_s-due), p.local_id))
                    if (len(close)>1 and abs(abs(close[0].due_s-due)-abs(close[1].due_s-due)) <= EPS):
                        raise ValueError("ambiguous identity matching")
                    old = close[0]
                    available.remove(old)
                    reused.append(old.local_id)
                    pending.append(TrackedEvent(old.local_id, due, old.confirmations+1, True))
                else:
                    new_id = self._next_id
                    self._next_id += 1
                    created.append(new_id)
                    pending.append(TrackedEvent(new_id, due, 1, True))
            # Remove contradicted events WITHIN the currently observable
            # window, but retain out-of-window plans if any.
            covered = (self.horizon_s if prediction.covered_until_s is None
                       else prediction.covered_until_s)
            window_end = now+covered+EPS
            canceled = [p for p in available if p.due_s <= window_end]
            pending.extend(TrackedEvent(p.local_id,p.due_s,p.confirmations,False)
                           for p in available if p.due_s > window_end)
        else:
            pending = [TrackedEvent(p.local_id,p.due_s,p.confirmations,False) for p in survivors]
        self._pending = sorted(pending, key=lambda p: (p.due_s,p.local_id))
        self._next_frame += 1
        return TimingUpdate(frame, prediction.status, tuple(self._pending),
                            tuple(elapsed), tuple(canceled), tuple(created),tuple(reused))
