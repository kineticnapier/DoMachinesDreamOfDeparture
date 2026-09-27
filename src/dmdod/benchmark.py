from __future__ import annotations

from dataclasses import dataclass

from .keyboard import KeyEvent
from .simulator import Simulation


@dataclass(frozen=True)
class RateResult:
    rate_hz: float
    presses: int
    duration_s: float
    interval_ms: float


def _count_down(sim: Simulation, left: float, right: float, steps: int) -> int:
    count = 0
    for _ in range(steps):
        result = sim.step(left, right)
        count += sum(1 for _, event in result.events if event is KeyEvent.DOWN)
    return count


def measure_periodic_rate(
    interval_ms: float,
    duration_s: float,
    *,
    mode: str = "single",
) -> RateResult:
    """Drive the body with a square-wave motor command and count real key actuations.

    interval_ms is the requested time between digital presses.  The benchmark
    never counts requested actions: only keyboard DOWN events produced by the
    physical model count.
    """
    if interval_ms <= 0.0 or duration_s <= 0.0:
        raise ValueError("interval_ms and duration_s must be positive")
    if mode not in {"single", "alternate"}:
        raise ValueError("mode must be 'single' or 'alternate'")

    sim = Simulation()
    dt_ms = sim.config.dt_s * 1000.0
    half_steps = max(1, round((interval_ms / 2.0) / dt_ms))
    total_steps = round(duration_s / sim.config.dt_s)
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
        presses += sum(1 for _, event in result.events if event is KeyEvent.DOWN)

    return RateResult(presses / duration_s, presses, duration_s, interval_ms)


def find_fastest_sustainable_rate(
    *,
    mode: str = "single",
    duration_s: float = 5.0,
    min_interval_ms: float = 20.0,
    max_interval_ms: float = 250.0,
    resolution_ms: float = 1.0,
    required_fraction: float = 0.98,
) -> RateResult:
    """Find the fastest requested rate the body can physically sustain.

    A candidate passes when at least required_fraction of its expected presses
    become real keyboard DOWN events.  Search proceeds from fast to slow so the
    first passing candidate is the physical limit under this controller.
    """
    interval = min_interval_ms
    while interval <= max_interval_ms + 1e-9:
        result = measure_periodic_rate(interval, duration_s, mode=mode)
        expected_rate = 1000.0 / interval
        if result.rate_hz >= expected_rate * required_fraction:
            return result
        interval += resolution_ms
    raise RuntimeError("No sustainable rate found in the requested interval range")
