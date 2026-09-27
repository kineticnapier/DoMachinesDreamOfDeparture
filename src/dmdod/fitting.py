from __future__ import annotations

from dataclasses import dataclass, replace
from math import log

from .benchmark import find_fastest_sustainable_rate
from .body import BodyConfig, FingerConfig


@dataclass(frozen=True)
class CalibrationTargets:
    short_single_hz: float = 8.9
    sustained_single_hz: float = 8.2
    alternate_hz: float = 12.5


@dataclass(frozen=True)
class CalibrationResult:
    config: BodyConfig
    loss: float
    short_single_hz: float
    sustained_single_hz: float
    alternate_hz: float


def _evaluate(config: BodyConfig, targets: CalibrationTargets, *, coarse: bool) -> CalibrationResult:
    resolution = 4.0 if coarse else 1.0
    short_duration = 3.0 if coarse else 5.0
    long_duration = 8.0 if coarse else 20.0
    alt_duration = 5.0 if coarse else 10.0

    short = find_fastest_sustainable_rate(
        mode="single", duration_s=short_duration, resolution_ms=resolution, body_config=config
    ).rate_hz
    sustained = find_fastest_sustainable_rate(
        mode="single", duration_s=long_duration, resolution_ms=resolution, body_config=config
    ).rate_hz
    alternate = find_fastest_sustainable_rate(
        mode="alternate", duration_s=alt_duration, resolution_ms=resolution, body_config=config
    ).rate_hz

    # Relative squared error prevents the larger alternating rate from dominating.
    loss = (
        ((short - targets.short_single_hz) / targets.short_single_hz) ** 2
        + ((sustained - targets.sustained_single_hz) / targets.sustained_single_hz) ** 2
        + ((alternate - targets.alternate_hz) / targets.alternate_hz) ** 2
    )
    return CalibrationResult(config, loss, short, sustained, alternate)


def _scaled(config: BodyConfig, *, tau: float = 1.0, force: float = 1.0, damping: float = 1.0,
            fatigue: float = 1.0, recovery: float = 1.0, coupling: float = 1.0) -> BodyConfig:
    def finger(f: FingerConfig) -> FingerConfig:
        return replace(
            f,
            activation_tau_s=f.activation_tau_s * tau,
            max_force_n=f.max_force_n * force,
            damping_n_s_m=f.damping_n_s_m * damping,
            fatigue_gain_s=f.fatigue_gain_s * fatigue,
            fatigue_recovery_s=f.fatigue_recovery_s * recovery,
        )

    return BodyConfig(
        left=finger(config.left),
        right=finger(config.right),
        left_affected_by_right=min(0.95, config.left_affected_by_right * coupling),
        right_affected_by_left=min(0.95, config.right_affected_by_left * coupling),
    )


def fit_body_config(targets: CalibrationTargets = CalibrationTargets()) -> CalibrationResult:
    """Fit a provisional two-finger body to the initial human observations.

    This intentionally uses a small deterministic coordinate search rather than
    a heavy optimizer dependency.  It is calibration of a model, not a claim
    that the resulting parameters are direct physiological measurements.
    """
    base = BodyConfig()

    # Broad search: response delay is the main rate limiter, while fatigue and
    # coupling control sustained and two-finger behavior. Force/damping remain
    # available because the key must still physically cross actuation/reset.
    candidates: list[BodyConfig] = []
    for tau in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
        for force in (0.6, 0.8, 1.0):
            for damping in (1.0, 1.5, 2.0):
                for fatigue in (1.0, 3.0, 6.0):
                    for coupling in (1.0, 2.5, 5.0):
                        candidates.append(
                            _scaled(base, tau=tau, force=force, damping=damping,
                                    fatigue=fatigue, coupling=coupling)
                        )

    best = min((_evaluate(c, targets, coarse=True) for c in candidates), key=lambda r: r.loss)

    # Local multiplicative refinement around the best coarse point.
    current = best.config
    for scale in (1.25, 1.12, 1.06):
        neighborhood = [current]
        for name in ("tau", "force", "damping", "fatigue", "recovery", "coupling"):
            for factor in (1.0 / scale, scale):
                kwargs = {name: factor}
                neighborhood.append(_scaled(current, **kwargs))
        refined = min((_evaluate(c, targets, coarse=True) for c in neighborhood), key=lambda r: r.loss)
        current = refined.config

    return _evaluate(current, targets, coarse=False)
