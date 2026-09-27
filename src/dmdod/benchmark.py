from __future__ import annotations

from dataclasses import dataclass

from .body import BodyConfig, TwoFingerBody
from .keyboard import KeyEvent
from .simulator import Simulation


@dataclass(frozen=True)
class RateResult:
    rate_hz: float
    presses: int
    duration_s: float
    interval_ms: float


def measure_periodic_rate(
    interval_ms: float,
    duration_s: float,
    *,
    mode: str = "single",
    body_config: BodyConfig | None = None,
    warmup_s: float = 1.0,
) -> RateResult:
    """Drive the body periodically and measure steady-state physical key presses.

    The controller runs continuously through warmup and measurement.  Only DOWN
    events after warmup count, avoiding startup transients in the reported rate.
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
    # Expected count is compared with tolerance for one boundary press.  This
    # avoids short/long tests disagreeing merely because their windows cut a
    # periodic sequence at different phases.
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
    """Find the fastest sustainable requested rate using a bounded search."""
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
