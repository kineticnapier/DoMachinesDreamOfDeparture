from __future__ import annotations

"""HUD/Oracle-independent Timing -> Fingering -> 12-axis Motor policy.

The policy receives ONLY a TimingFrameUpdate and its CURRENT physical Motor.
No chart, target ordinal, judge, true timestamp, or evaluator object is
accepted. Source switching is exclusively handled by the integration runner.
"""

from dataclasses import dataclass
from math import log1p

from dmdod.motor.contact_16k import (
    ContactAction, Coupled16ContactMotor, PHYSICAL_AXES, ThumbMode,
)
from dmdod.planning.contact_fingering_candidates_v02 import (
    center_first, finger_options, is_eligible, to_physical_fingers,
)
from dmdod.planning.persistent_fingering_scheduler_v01 import (
    MotorPhysicalSnapshot, PersistentFingeringScheduler, ReservationPhase,
)
from dmdod.planning.timing_fingering_context_v02 import (
    encode_context_from_timing_update,
)
from dmdod.planning.timing_fingering_contract_v02 import TimingFrameUpdate
from dmdod.training import fingering_network_v01 as network

NORMAL_LEAD_S = .044
DECISION_LEAD_S = .069
MAX_EXTRA_LEAD_S = .025
NEGATIVE_ACTIVATION_GATE = -.010
PREVIEW_TAIL_S = .035


@dataclass(frozen=True, slots=True)
class FingeringDecision:
    local_id: int
    step: int
    due_local_s: float
    choice: str | None
    reason: str
    extra_lead_s: float


class TimingFingeringMotorPolicy:
    """Frozen Net or CenterFirst on a shared stateful physical reservation."""

    def __init__(self, *, model=None, style_alpha: float = .5,
                 selector: str = "network", early_lead: bool = True):
        if selector not in ("network", "center_first"):
            raise ValueError("unsupported selector")
        if selector == "network" and model is None:
            raise ValueError("frozen Network model required")
        if not 0.0 <= style_alpha <= 1.0:
            raise ValueError("style alpha must lie in [0,1]")
        self.model = model
        self.style_alpha = style_alpha
        self.selector = selector
        self.early_lead = early_lead
        self.reset()

    def reset(self) -> None:
        self.scheduler = PersistentFingeringScheduler()
        self._early_by_id: dict[int, float] = {}
        self._decided_ids: set[int] = set()
        self.decisions: list[FingeringDecision] = []
        self.insufficient_observable_lead = 0
        self.unassigned = 0
        self.hud_abstentions = 0
        self.finger_collisions = 0
        self.inflight_cancellations = 0

    def _extra_lead(self, motor: Coupled16ContactMotor,
                    allocation: tuple) -> float:
        if not self.early_lead or len(allocation) != 1:
            return 0.0
        axis = allocation[0].axis
        if int(axis.rsplit("_", 1)[1]) > 4:
            return 0.0
        body = motor.motor
        index = PHYSICAL_AXES.index(axis)
        if body.observe().pressed(axis):
            return 0.0
        state = body.body.states[index]
        if state.activation >= NEGATIVE_ACTIVATION_GATE:
            return 0.0
        tau = body.body.configs[index].activation_tau_s
        if tau <= 0:
            raise ValueError("invalid Motor activation time constant")
        return min(MAX_EXTRA_LEAD_S, tau * log1p(-state.activation))

    def _choose(self, update: TimingFrameUpdate,
                token_id: int, motor: Coupled16ContactMotor,
                snapshot: MotorPhysicalSnapshot) -> tuple | None:
        token = next(x for x in update.pending if x.local_id == token_id)
        candidates = finger_options(token.chord_size)
        allowed = tuple(is_eligible(option, self.scheduler, motor, snapshot)
                        for option in candidates)
        if not any(allowed):
            return None
        if self.selector == "center_first":
            selected = center_first(candidates, allowed)
        else:
            context = encode_context_from_timing_update(
                motor, update, set(self.scheduler.occupied_axes()),
                starting_at_local_id=token.local_id,
            )
            selected, _, _ = network.choose_candidate_styled(
                self.model, context, candidates, allowed, alpha=self.style_alpha,
            )
        return candidates[selected] if selected is not None else None

    def command(self, update: TimingFrameUpdate,
                motor: Coupled16ContactMotor) -> ContactAction:
        """One local 10ms step. Call observe_physics AFTER motor.step()."""
        snapshot = MotorPhysicalSnapshot.from_motor(motor)
        self.scheduler.apply_forecast(update, snapshot)
        now = update.local_clock_s
        if update.status == "abstain":
            self.hud_abstentions += 1

        for token in update.pending:
            if token.local_id in self._decided_ids:
                continue
            # Stable acquisition is only at the frozen 69ms decision window,
            # not as soon as an Oracle can see an arbitrarily distant event.
            remaining = token.due_local_s - now
            if not token.confirmed_now or remaining > DECISION_LEAD_S + 1e-9:
                continue
            if remaining < NORMAL_LEAD_S - 1e-9:
                self.insufficient_observable_lead += 1
                self._decided_ids.add(token.local_id)
                self.decisions.append(FingeringDecision(
                    token.local_id, update.step, token.due_local_s,
                    None, "insufficient_observable_lead", 0.0
                ))
                continue
            allocation = self._choose(update, token.local_id, motor, snapshot)
            if allocation is None:
                self.unassigned += 1
                self._decided_ids.add(token.local_id)
                self.decisions.append(FingeringDecision(
                    token.local_id, update.step, token.due_local_s,
                    None, "no_eligible_axis", 0.0
                ))
                continue
            try:
                self.scheduler.reserve(
                    token.local_id, to_physical_fingers(allocation), snapshot
                )
            except ValueError:
                # Concurrent availability contention must not invent keys.
                self.finger_collisions += 1
                self._decided_ids.add(token.local_id)
                self.decisions.append(FingeringDecision(
                    token.local_id, update.step, token.due_local_s,
                    None, "reservation_collision", 0.0
                ))
                continue
            extra = self._extra_lead(motor, allocation)
            self._early_by_id[token.local_id] = extra
            self._decided_ids.add(token.local_id)
            self.decisions.append(FingeringDecision(
                token.local_id, update.step, token.due_local_s,
                " + ".join(x.label for x in allocation), "assigned", extra
            ))

        observed = motor.motor.observe()
        commands = {axis: (-1.0 if observed.pressed(axis) else 0.0)
                    for axis in PHYSICAL_AXES}
        modes = list(motor._modes)
        edges = list(motor._edge)
        for reservation in tuple(self.scheduler.reservations):
            local_id = reservation.local_id
            if reservation.phase is ReservationPhase.RESERVED:
                extra = self._early_by_id.get(local_id, 0.0)
                start = reservation.due_local_s - NORMAL_LEAD_S - extra
                if (reservation.confirmed and now + 1e-9 >= start
                        and now <= reservation.due_local_s + PREVIEW_TAIL_S):
                    self.scheduler.arm(local_id, snapshot)
                    reservation = self.scheduler.reservation(local_id)
            if reservation.phase is ReservationPhase.ARMED_IN_FLIGHT:
                if now > reservation.due_local_s + PREVIEW_TAIL_S:
                    self.scheduler.cancel_armed(local_id, "past_motor_window")
                    self.inflight_cancellations += 1
                    reservation = self.scheduler.reservation(local_id)
                elif reservation.confirmed:
                    for finger in reservation.fingers:
                        axis = finger.axis
                        if axis in reservation.down_axes or observed.pressed(axis):
                            continue
                        hand, depth = axis.rsplit("_", 1)
                        index = 0 if hand == "left" else 1
                        if depth == "5":
                            wanted = (ThumbMode.FLAT if finger.contact_count == 2
                                      else ThumbMode.TIP)
                            modes[index] = wanted
                        elif depth == "6":
                            edges[index] = True
                        commands[axis] = 1.0
                    continue
            if reservation.phase in (ReservationPhase.RELEASING,
                                     ReservationPhase.CANCELED_IN_FLIGHT,
                                     ReservationPhase.KEYDOWN_OBSERVED):
                for axis in reservation.axes:
                    commands[axis] = -1.0
            elif reservation.phase is ReservationPhase.ARMED_IN_FLIGHT:
                # Unconfirmed event during Pause or ambiguous observations:
                # physically brake, but retain reservation until reset.
                for axis in reservation.axes:
                    commands[axis] = -1.0
        return ContactAction(
            tuple(commands[axis] for axis in PHYSICAL_AXES),
            tuple(modes), tuple(edges),
        )

    def observe_physics(self, motor: Coupled16ContactMotor) -> None:
        """Only twelve physical switch transitions, NEVER scorer Hit/Miss."""
        transition = motor.motor.last_transition
        if transition is None:
            raise RuntimeError("Motor must step before observing its physical edges")
        physical_edges = tuple((event.key, event.event)
                               for event in transition.evaluator_events)
        snapshot = MotorPhysicalSnapshot.from_motor(motor)
        self.scheduler.observe_physics(snapshot, physical_edges)
