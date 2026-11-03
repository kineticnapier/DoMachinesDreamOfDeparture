from __future__ import annotations

import pytest

from dmdod.training.budget import NonBestRestartController, TrustRadiusController


@pytest.mark.parametrize("field", ("initial", "minimum", "reject_shrink", "safe_grow"))
@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_trust_radius_rejects_nonfinite_parameters(field: str, bad: float) -> None:
    values = dict(initial=0.01, minimum=0.001, reject_shrink=0.5, safe_grow=1.25)
    values[field] = bad
    with pytest.raises(ValueError, match=rf"{field} must be finite"):
        TrustRadiusController.create(**values)


class _FloatOnlyRadius:
    def __init__(self, value: float) -> None:
        self.value = value

    def __float__(self) -> float:
        return self.value


class _SpoofedFloatRadius(float):
    def __float__(self) -> float:
        return float("nan")


class _SpoofedIntRadius(int):
    def __float__(self) -> float:
        return float("nan")


@pytest.mark.parametrize("field", ("initial", "minimum", "reject_shrink", "safe_grow"))
@pytest.mark.parametrize("wrapper", (_FloatOnlyRadius, _SpoofedFloatRadius))
def test_trust_radius_normalizes_numeric_objects_before_comparison(field, wrapper) -> None:
    values = dict(initial=0.01, minimum=0.001, reject_shrink=0.5, safe_grow=1.25)
    values[field] = wrapper(values[field])
    radius = TrustRadiusController.create(**values)
    assert getattr(radius, field) == (
        float.__float__(values[field]) if isinstance(values[field], float)
        else values[field].value
    )
    assert type(getattr(radius, field)) is float
    assert radius.current == 0.01


def test_trust_radius_uses_stored_integer_payload_not_overridden_float() -> None:
    radius = TrustRadiusController.create(
        initial=0.01, minimum=0.001, reject_shrink=0.5,
        safe_grow=_SpoofedIntRadius(2),
    )
    assert radius.safe_grow == 2.0
    assert type(radius.safe_grow) is float


@pytest.mark.parametrize("field", ("initial", "minimum", "reject_shrink", "safe_grow"))
@pytest.mark.parametrize("bad", (True, "0.5"))
def test_trust_radius_rejects_boolean_and_string_parameters(field, bad) -> None:
    values = dict(initial=0.01, minimum=0.001, reject_shrink=0.5, safe_grow=1.25)
    values[field] = bad
    with pytest.raises(ValueError, match=rf"{field} must be finite"):
        TrustRadiusController.create(**values)


class _FalseComparisonRadius(float):
    def __float__(self) -> float:
        return 2.0

    def __lt__(self, other) -> bool:
        return False


def test_trust_radius_rejects_invalid_stored_float_despite_overrides() -> None:
    values = dict(initial=0.01, minimum=0.001, reject_shrink=0.5, safe_grow=1.25)
    values["safe_grow"] = _FalseComparisonRadius(-1.0)
    with pytest.raises(ValueError, match="safe_grow must be >= 1"):
        TrustRadiusController.create(**values)


def test_trust_radius_preserves_finite_parameter_behavior() -> None:
    radius = TrustRadiusController.create(
        initial=0.01, minimum=0.001, reject_shrink=0.5, safe_grow=1.25,
    )
    assert radius.current == 0.01
    assert radius.reject()
    assert radius.current == 0.005
    assert radius.accept() == 0.00625


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


def test_nonbest_restart_controller_restarts_at_limit() -> None:
    controller = NonBestRestartController.create(limit=3)

    assert not controller.observe(selected_best=False)
    assert controller.streak == 1
    assert not controller.observe(selected_best=False)
    assert controller.streak == 2
    assert controller.observe(selected_best=False)
    assert controller.streak == 0
    assert controller.restarts == 1


def test_nonbest_restart_controller_best_resets_streak() -> None:
    controller = NonBestRestartController.create(limit=2)

    assert not controller.observe(selected_best=False)
    assert not controller.observe(selected_best=True)
    assert controller.streak == 0
    assert controller.restarts == 0


@pytest.mark.parametrize(
    "invalid", (0, -1, 2.5, 3.0, True, "3", float("nan"), float("inf")),
)
def test_nonbest_restart_controller_rejects_noninteger_limit(invalid) -> None:
    with pytest.raises(ValueError, match="non-best restart limit must be a positive integer"):
        NonBestRestartController.create(limit=invalid)


def test_nonbest_restart_controller_keeps_exact_integer_limit() -> None:
    controller = NonBestRestartController.create(limit=3)
    assert controller.limit == 3
    assert [controller.observe(selected_best=False) for _ in range(3)] == [
        False, False, True,
    ]
