from __future__ import annotations

from dataclasses import dataclass

from dmdod.motor.body import BodyConfig, TwoFingerBody
from dmdod.motor.keyboard import KeyEvent
from dmdod.envs.simulator import Simulation


@dataclass(frozen=True)
class RateResult:
    rate_hz: float
    presses: int
    duration_s: float
    interval_ms: float


@dataclass(frozen=True)
class FeedbackRateResult:
    """Rate produced by the threshold-feedback maximum-speed controller."""

    rate_hz: float
    presses: int
    duration_s: float


def _estimate_single_cycle_steps(config: BodyConfig, *, probe_s: float = 1.0) -> int:
    """Estimate one settled left-finger DOWN->DOWN cycle for phase seeding.

    Cross-hand fingers are physically independent in the current model.  We use
    a short probe only to obtain the body's own cycle length, then start the
    second hand half a cycle later.  This avoids reintroducing a fixed duty cycle
    or waiting for the previous hand's DOWN before the other hand may move.
    """
    sim = Simulation(body=TwoFingerBody(config=config))
    steps = max(1, round(probe_s / sim.config.dt_s))
    pressing = True
    downs: list[int] = []

    for step in range(steps):
        result = sim.step(1.0 if pressing else -1.0, 0.0)
        for finger, event in result.events:
            if finger != "left":
                continue
            if event is KeyEvent.DOWN:
                pressing = False
                downs.append(step)
            elif event is KeyEvent.UP:
                pressing = True

    if len(downs) < 2:
        raise RuntimeError("Could not estimate a single-finger feedback cycle")

    intervals = [b - a for a, b in zip(downs[-5:-1], downs[-4:])]
    if not intervals:
        intervals = [downs[-1] - downs[-2]]
    return max(2, round(sum(intervals) / len(intervals)))


def measure_feedback_rate(
    duration_s: float,
    *,
    mode: str = "single",
    body_config: BodyConfig | None = None,
    warmup_s: float = 1.0,
) -> FeedbackRateResult:
    """Measure maximum tapping using key DOWN/UP events as controller feedback.

    Single-finger control presses until actuation, releases until reset, and
    immediately presses again.

    Same-hand alternation remains sequential because the shared hand/coordination
    model is intended to constrain finger switching.  Cross-hand alternation is
    different: the two independent hands run their own feedback loops in
    parallel.  The right hand is seeded half of the body's measured single-finger
    cycle after the left, approximating a 180-degree phase offset without a fixed
    50:50 square wave.
    """
    if duration_s <= 0.0 or warmup_s < 0.0:
        raise ValueError("duration_s must be positive and warmup_s non-negative")
    if mode not in {"single", "alternate"}:
        raise ValueError("mode must be 'single' or 'alternate'")

    config = body_config or BodyConfig()
    sim = Simulation(body=TwoFingerBody(config=config))
    warmup_steps = round(warmup_s / sim.config.dt_s)
    measure_steps = round(duration_s / sim.config.dt_s)
    total_steps = warmup_steps + measure_steps
    presses = 0

    single_press = True
    expected = "left"

    # Independent-hand controller state.  The short phase probe is performed in
    # a separate simulation, so it does not alter fatigue in the measured run.
    parallel_cross_hand = mode == "alternate" and not config.same_hand
    left_press = True
    right_press = True
    right_start_step = 0
    if parallel_cross_hand:
        period_steps = _estimate_single_cycle_steps(config)
        right_start_step = max(1, round(period_steps / 2.0))

    for step in range(total_steps):
        if mode == "single":
            left_command = 1.0 if single_press else -1.0
            right_command = 0.0
        elif parallel_cross_hand:
            left_command = 1.0 if left_press else -1.0
            if step < right_start_step:
                right_command = 0.0
            else:
                right_command = 1.0 if right_press else -1.0
        else:
            # Same-hand alternation: the selected finger presses while the other
            # releases.  Shared hand coordination is therefore part of the rate.
            if expected == "left":
                left_command = -1.0 if sim.keyboard.left.pressed else 1.0
                right_command = -1.0
            else:
                left_command = -1.0
                right_command = -1.0 if sim.keyboard.right.pressed else 1.0

        result = sim.step(left_command, right_command)

        for finger, event in result.events:
            if mode == "single" and finger == "left":
                if event is KeyEvent.DOWN:
                    single_press = False
                    if step >= warmup_steps:
                        presses += 1
                elif event is KeyEvent.UP:
                    single_press = True
            elif parallel_cross_hand:
                if finger == "left":
                    if event is KeyEvent.DOWN:
                        left_press = False
                        if step >= warmup_steps:
                            presses += 1
                    elif event is KeyEvent.UP:
                        left_press = True
                elif finger == "right" and step >= right_start_step:
                    if event is KeyEvent.DOWN:
                        right_press = False
                        if step >= warmup_steps:
                            presses += 1
                    elif event is KeyEvent.UP:
                        right_press = True
            elif event is KeyEvent.DOWN and finger == expected:
                if step >= warmup_steps:
                    presses += 1
                expected = "right" if expected == "left" else "left"

    return FeedbackRateResult(presses / duration_s, presses, duration_s)


def measure_periodic_rate(
    interval_ms: float,
    duration_s: float,
    *,
    mode: str = "single",
    body_config: BodyConfig | None = None,
    warmup_s: float = 1.0,
) -> RateResult:
    """Drive the body periodically and measure steady-state physical key presses.

    Retained as a diagnostic for fixed-rate tests. Calibration should normally
    use measure_feedback_rate so the controller does not define the speed limit.
    """
    if interval_ms <= 0.0 or duration_s <= 0.0 or warmup_s < 0.0:
        raise ValueError("interval_ms/duration_s must be positive and warmup_s non-negative")
    if mode not in {"single", "alternate"}:
        raise ValueError("mode must be 'single' or 'alternate'")

    sim = Simulation(body=TwoFingerBody(config=body_config or BodyConfig()))
    dt_ms = sim.config.dt_s * 1000.0
    half_steps = max(1, round((interval_ms / 2.0) / dt_ms))
    warmup_steps = round(warmup_s / sim.config.dt_s)
    measure_steps = round(duration_s / sim.config.dt_s)
    total_steps = warmup_steps + measure_steps
    presses = 0

    for step in range(total_steps):
        phase = (step // half_steps) % 2
        if mode == "single":
            left = 1.0 if phase == 0 else -1.0
            right = 0.0
        else:
            left = 1.0 if phase == 0 else -1.0
            right = -left

        result = sim.step(left, right)
        if step >= warmup_steps:
            presses += sum(1 for _, event in result.events if event is KeyEvent.DOWN)

    return RateResult(presses / duration_s, presses, duration_s, interval_ms)


def _sustainable(result: RateResult, required_fraction: float) -> bool:
    expected = result.duration_s * 1000.0 / result.interval_ms
    required = max(0.0, expected * required_fraction - 1.0)
    return result.presses >= required


def find_fastest_sustainable_rate(
    *,
    mode: str = "single",
    duration_s: float = 5.0,
    min_interval_ms: float = 20.0,
    max_interval_ms: float = 250.0,
    resolution_ms: float = 1.0,
    required_fraction: float = 0.98,
    body_config: BodyConfig | None = None,
    warmup_s: float = 1.0,
) -> RateResult:
    """Legacy fixed-period search, retained for diagnostics and comparisons."""
    if min_interval_ms <= 0 or max_interval_ms <= min_interval_ms or resolution_ms <= 0:
        raise ValueError("invalid interval search range")

    cache: dict[float, RateResult] = {}

    def evaluate(interval: float) -> RateResult:
        interval = max(min_interval_ms, min(max_interval_ms, interval))
        interval = round(interval / resolution_ms) * resolution_ms
        if interval not in cache:
            cache[interval] = measure_periodic_rate(
                interval,
                duration_s,
                mode=mode,
                body_config=body_config,
                warmup_s=warmup_s,
            )
        return cache[interval]

    fastest = evaluate(min_interval_ms)
    if _sustainable(fastest, required_fraction):
        return fastest

    fail = min_interval_ms
    candidate = min_interval_ms
    step = max(8.0 * resolution_ms, 8.0)
    passed: float | None = None
    while candidate < max_interval_ms:
        candidate = min(max_interval_ms, candidate + step)
        result = evaluate(candidate)
        if _sustainable(result, required_fraction):
            passed = candidate
            break
        fail = candidate
        step *= 1.7

    if passed is None:
        raise RuntimeError("No sustainable rate found in the requested interval range")

    lo, hi = fail, passed
    while hi - lo > resolution_ms:
        mid = (lo + hi) / 2.0
        result = evaluate(mid)
        if _sustainable(result, required_fraction):
            hi = result.interval_ms
        else:
            lo = result.interval_ms
        if hi - lo <= resolution_ms:
            break

    start = max(min_interval_ms, hi - 3.0 * resolution_ms)
    end = min(max_interval_ms, hi + 3.0 * resolution_ms)
    interval = start
    best: RateResult | None = None
    while interval <= end + 1e-9:
        result = evaluate(interval)
        if _sustainable(result, required_fraction):
            if best is None or result.interval_ms < best.interval_ms:
                best = result
        interval += resolution_ms

    if best is None:
        raise RuntimeError("Sustainable-rate refinement failed")
    return best
