from __future__ import annotations

"""Auditable rhythm-phrase fingering hypothesis, NOT learned human behaviour.

Uses exact onset times for offline research. This is not a HUD-only policy and
makes no physical Motor feasibility or human-likeness claim. Physical safety
must be verified by the live Motor adapter.
"""

from dataclasses import dataclass
from math import isfinite
from statistics import median
from typing import Sequence


@dataclass(frozen=True, slots=True)
class PhraseEvent:
    event_index: int
    due_s: float
    chord_size: int = 1


@dataclass(frozen=True, slots=True)
class RhythmPhrase:
    start_event_index: int
    end_event_index: int
    event_count: int
    rhythm_class: str
    gap_motif_s: tuple[float, ...]
    boundary_reason: str
    template: str = "unplanned"
    mirrored: bool = False
    phase: int = 0


@dataclass(frozen=True, slots=True)
class PhrasePlan:
    feasible: bool
    assignments: tuple[tuple[str, ...], ...]
    phrases: tuple[RhythmPhrase, ...]
    failure_reason: str | None
    states_evaluated: int
    metrics: dict
    model_contract: dict


# The only keys admitted here map to eight uncoupled ordinary axes under the
# fixed PatternMotorMapping.default_eight() topology. The values are axis
# identifiers, not validated finger anatomy, strength or human demonstration.
ORDINARY_AXES = (
    "left_1", "right_1", "left_2", "right_2",
    "left_3", "right_3", "left_4", "right_4",
)

# Alternative gesture descriptions, not a universal 8-key circular optimum.
# Each motif is a preference that must be tested against real demonstrations.
TEMPLATES = {
    "interleaved": (0, 1, 2, 3, 4, 5, 6, 7),
    "paired_hand": (0, 2, 1, 3, 4, 6, 5, 7),
    "hand_roll": (0, 2, 4, 6, 1, 3, 5, 7),
}


def _rhythm_class(events: tuple[PhraseEvent, ...]) -> tuple[str, tuple[float, ...]]:
    if any(event.chord_size > 1 for event in events):
        return "chord", ()
    gaps = tuple(b.due_s - a.due_s for a, b in zip(events, events[1:]))
    if len(gaps) < 2:
        return "short_or_isolated", ()
    typical = median(gaps)
    # Discover repeating inter-onset intervals, not guessed finger motions.
    # An N-gap motif spans N+1 notes; require >=2 complete repetitions.
    for period in range(1, min(8, len(gaps) // 2) + 1):
        representative = tuple(median(gaps[phase::period])
                               for phase in range(period))
        tolerance = max(.001, .06 * max(representative))
        if all(abs(gap - representative[i % period]) <= tolerance
               for i, gap in enumerate(gaps)):
            if period == 1:
                return ("even_dense" if typical <= .045 else "even_spaced"), (
                    round(representative[0], 6),)
            if period == 2 and max(representative) >= 1.8 * min(representative):
                return "alternating_gaps", tuple(round(x, 6) for x in representative)
            return f"periodic_{period}", tuple(round(x, 6) for x in representative)
    # Relative short/long separation survives gradual BPM drift.
    if len(gaps) >= 5:
        evens, odds = gaps[::2], gaps[1::2]
        low, high = median(evens), median(odds)
        if (min(low, high) > 0 and (
                max(evens) * 1.8 < min(odds)
                or max(odds) * 1.8 < min(evens))):
            return "alternating_gaps", (round(low, 6), round(high, 6))
    return "irregular", ()


class HumanPhrasePlannerV07:
    """Bounded beam search over rhythm chunks and declared gesture templates.

    Phrase boundaries are timing-derived planning windows, not verified
    musical phrases. A rhythm label is not evidence of human motor intent.
    """

    def __init__(self, *, minimum_reuse_s: float = .080,
                 max_phrase_events: int = 16, restart_gap_s: float = .450,
                 beam_width: int = 32) -> None:
        if not isfinite(minimum_reuse_s) or minimum_reuse_s < 0:
            raise ValueError("minimum_reuse_s must be finite and nonnegative")
        if not isfinite(restart_gap_s) or restart_gap_s <= 0:
            raise ValueError("restart_gap_s must be finite and positive")
        if max_phrase_events < 4 or max_phrase_events > 64:
            raise ValueError("max_phrase_events must be 4..64")
        if beam_width < 1 or beam_width > 256:
            raise ValueError("beam_width must be 1..256")
        self.minimum_reuse_s = float(minimum_reuse_s)
        self.max_phrase_events = int(max_phrase_events)
        self.restart_gap_s = float(restart_gap_s)
        self.beam_width = int(beam_width)

    def _contract(self) -> dict:
        return {
            "source": "offline_exact_event_timestamps_and_chord_sizes",
            "status": "untrained_rhythm_phrase_style_hypothesis",
            "human_demonstrations_used": False,
            "human_likeness_verified": False,
            "motor_feasibility_verified": False,
            "student_hud_only": False,
            "minimum_reuse_s": self.minimum_reuse_s,
            "max_phrase_events": self.max_phrase_events,
            "restart_gap_s": self.restart_gap_s,
            "beam_width": self.beam_width,
            "beam_optimality": "heuristic_not_globally_optimal",
            "physical_topology": "ordinary_axes_only; mapping fixed; no thumb/side",
        }

    def _windows(self, events: tuple[PhraseEvent, ...]) -> tuple[tuple[tuple[PhraseEvent, ...], str], ...]:
        if not events:
            return ()
        windows: list[tuple[tuple[PhraseEvent, ...], str]] = []
        start = 0
        reason = "chart_start"
        for i in range(1, len(events)):
            gap = events[i].due_s - events[i - 1].due_s
            pause = gap >= self.restart_gap_s
            chord_change = events[i].chord_size != events[i - 1].chord_size
            horizon = i - start >= self.max_phrase_events
            if pause or chord_change or horizon:
                windows.append((events[start:i], reason))
                start = i
                reason = "pause" if pause else "chord_change" if chord_change else "planning_horizon"
        windows.append((events[start:], reason))
        return tuple(windows)

    @staticmethod
    def _template_order(rhythm_class: str) -> tuple[str, ...]:
        if rhythm_class.startswith("periodic_"):
            period = int(rhythm_class.split("_")[1])
            return (("hand_roll", "paired_hand", "interleaved") if period in (3, 4)
                    else ("paired_hand", "hand_roll", "interleaved"))
        return {
            "alternating_gaps": ("paired_hand", "interleaved", "hand_roll"),
            "even_dense": ("hand_roll", "interleaved", "paired_hand"),
            "even_spaced": ("interleaved", "paired_hand", "hand_roll"),
            "chord": ("interleaved", "paired_hand", "hand_roll"),
            "short_or_isolated": ("interleaved", "paired_hand", "hand_roll"),
            "irregular": ("paired_hand", "interleaved", "hand_roll"),
        }[rhythm_class]

    @staticmethod
    def _candidate(window: tuple[PhraseEvent, ...], template: str,
                   mirrored: bool, phase: int) -> tuple[tuple[str, ...], ...]:
        numbers = TEMPLATES[template]
        if mirrored:
            numbers = tuple(value ^ 1 for value in numbers)
        numbers = numbers[phase:] + numbers[:phase]
        cursor = 0
        result = []
        for event in window:
            keys = tuple(ORDINARY_AXES[numbers[(cursor + offset) % 8]]
                         for offset in range(event.chord_size))
            result.append(keys)
            cursor += event.chord_size
        return tuple(result)

    def plan(self, sequence: Sequence[PhraseEvent]) -> PhrasePlan:
        events = tuple(sequence)
        contract = self._contract()

        def failed(reason: str, count: int = 0) -> PhrasePlan:
            return PhrasePlan(False, (), (), reason, count, {}, contract)

        if any(not isfinite(event.due_s) for event in events):
            return failed("non_finite_time")
        if any(event.chord_size < 1 or event.chord_size > len(ORDINARY_AXES)
               for event in events):
            return failed("invalid_chord_size")
        if any(right.due_s < left.due_s for left, right in zip(events, events[1:])):
            return failed("nonmonotone_times")
        if len({event.event_index for event in events}) != len(events):
            return failed("duplicate_event_id")
        if not events:
            return PhrasePlan(True, (), (), None, 0, {"phrase_count": 0}, contract)

        # A finite beam preserves determinism, not global optimality.
        # state = (objective, last_axis_uses, choices, phrases, previous_template).
        states = [((0, 0, ()), {}, (), (), None)]
        evaluated = 0
        for window, reason in self._windows(events):
            rhythm, motif = _rhythm_class(window)
            preferred = self._template_order(rhythm)
            next_states = []
            for objective, prev_use, selected, annotations, previous_template in states:
                for rank, template in enumerate(preferred):
                    for mirrored in (False, True):
                        for phase in range(8):
                            candidate = self._candidate(window, template, mirrored, phase)
                            evaluated += 1
                            recent = prev_use.copy()
                            valid = True
                            for event, axes in zip(window, candidate):
                                if len(set(axes)) != len(axes):
                                    valid = False
                                    break
                                for axis in axes:
                                    if (axis in recent and
                                            event.due_s - recent[axis] <= 1e-12):
                                        valid = False
                                        break
                                    if (axis in recent and event.due_s - recent[axis] + 1e-12
                                            < self.minimum_reuse_s):
                                        valid = False
                                        break
                                    recent[axis] = event.due_s
                                if not valid:
                                    break
                            if not valid:
                                continue
                            change = int(previous_template is not None
                                         and previous_template != template)
                            cost = (objective[0] + rank,
                                    objective[1] + change,
                                    objective[2] + ((template, mirrored, phase),))
                            phrase = RhythmPhrase(
                                window[0].event_index, window[-1].event_index,
                                len(window), rhythm, motif, reason,
                                template, mirrored, phase)
                            next_states.append((cost, recent, selected + candidate,
                                                annotations + (phrase,), template))
            if not next_states:
                return failed("no_abstract_reuse_safe_gesture_in_beam", evaluated)
            next_states.sort(key=lambda item: item[0])
            states = next_states[:self.beam_width]

        objective, _, assignments, phrases, _ = states[0]
        last_use: dict[str, float] = {}
        min_gap = float("inf")
        for event, axes in zip(events, assignments):
            for axis in axes:
                if axis in last_use:
                    min_gap = min(min_gap, event.due_s - last_use[axis])
                last_use[axis] = event.due_s
        return PhrasePlan(True, assignments, phrases, None, evaluated, {
            "phrase_count": len(phrases),
            "rhythm_classes": {name: sum(p.rhythm_class == name for p in phrases)
                               for name in sorted({p.rhythm_class for p in phrases})},
            "gesture_template_changes": sum(a.template != b.template
                                            for a, b in zip(phrases, phrases[1:])),
            "minimum_abstract_axis_reuse_ms": (
                None if min_gap == float("inf") else round(min_gap * 1000, 6)),
            "abstract_reuse_violations": 0,
            "style_cost": objective[:2],
        }, contract)
