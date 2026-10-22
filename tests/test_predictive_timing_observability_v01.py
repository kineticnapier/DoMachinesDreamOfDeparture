from __future__ import annotations

from dmdod.planning.predictive_timing_observability_v01 import (
    AcceptableStrikeWindow,
    DueInterval,
    EarlyDueStatus,
    LeadStatus,
    VisiblePrefixFrame,
    assess_shared_safe_strike,
    compare_visible_prefixes,
)


def _frames(last: int, *, changed_after: int | None = None):
    return tuple(
        VisiblePrefixFrame(i / 100.0, (float(i), 1.0 if changed_after is not None and i >= changed_after else 0.0), ("hidden",))
        for i in range(last + 1)
    )


def test_prefix_decision_is_invariant_to_future_frames() -> None:
    left = _frames(20)
    right = _frames(20, changed_after=18)
    short = compare_visible_prefixes(left, right, deadline_s=0.17)
    left_future = tuple(VisiblePrefixFrame(0.21 + i / 100.0, (9.0, 0.0), ("future",)) for i in range(5))
    right_future = tuple(VisiblePrefixFrame(0.21 + i / 100.0, (-9.0, 1.0), ("different",)) for i in range(5))
    extended = compare_visible_prefixes(left + left_future, right + right_future, deadline_s=0.17)
    assert short == extended
    assert short.status is EarlyDueStatus.NONIDENTIFIABLE


def test_counterfactual_equal_prefix_is_nonidentifiable() -> None:
    result = compare_visible_prefixes(_frames(10), _frames(10), deadline_s=0.10)
    assert result.status is EarlyDueStatus.NONIDENTIFIABLE
    assert result.maximum_feature_difference == 0.0


def test_44ms_boundary_is_inclusive_for_point_due() -> None:
    due = DueInterval(1.0, 1.0)
    window = (AcceptableStrikeWindow(0.95, 1.05),)
    result = assess_shared_safe_strike(observed_at_s=0.956, due_interval=due, acceptable_windows=window)
    assert result.status is LeadStatus.FEASIBLE
    late = assess_shared_safe_strike(observed_at_s=0.9560001, due_interval=due, acceptable_windows=window)
    assert late.status is LeadStatus.INFEASIBLE


def test_disjoint_counterfactual_hit_windows_have_no_safe_reservation() -> None:
    result = assess_shared_safe_strike(
        observed_at_s=3.17,
        due_interval=DueInterval(3.2205882724, 3.5294118055),
        acceptable_windows=(
            AcceptableStrikeWindow(3.1555882724, 3.2855882724),
            AcceptableStrikeWindow(3.4117647467, 3.6470588643),
        ),
    )
    assert result.status is LeadStatus.INFEASIBLE
    assert result.shared_safe_strike_window is None


def test_observability_types_have_no_oracle_identity_fields() -> None:
    frame = VisiblePrefixFrame(0.0, (1.0,))
    due = DueInterval(1.0, 1.1)
    assert not hasattr(frame, "target_index")
    assert not hasattr(due, "floor_index")
