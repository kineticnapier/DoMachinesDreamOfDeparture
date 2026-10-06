from __future__ import annotations

from dmdod.training.budget import TrustRadiusController


def test_reject_tries_floor_exactly_once() -> None:
    radius = TrustRadiusController.create(
        initial=0.01,
        minimum=0.001,
        reject_shrink=0.5,
        safe_grow=1.25,
    )

    assert radius.reject()
    assert radius.current == 0.005
    assert radius.reject()
    assert radius.current == 0.0025
    assert radius.reject()
    assert radius.current == 0.00125
    assert radius.reject()
    assert radius.current == 0.001

    # The caller now gets one trial at the exact floor. Only a rejection at
    # that floor reports that there is nowhere smaller to go.
    assert not radius.reject()
    assert radius.current == 0.001


def test_safe_growth_never_exceeds_initial_radius() -> None:
    radius = TrustRadiusController.create(
        initial=0.01,
        minimum=1e-6,
        reject_shrink=0.5,
        safe_grow=2.0,
    )
    radius.current = 0.008
    assert radius.accept() == 0.01
    assert radius.accept() == 0.01
