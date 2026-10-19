from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import textwrap

import pytest

from dmdod.planning.countdown_first_event_v01 import CountdownFirstEventEstimate
from dmdod.planning.countdown_phase_tracker_v01 import (
    CountdownPhaseUpdate,
    TimeInterval,
)
from dmdod.planning.hud_timing_composite_v20 import (
    TimingEventToken as CompositeToken,
    TimingFrameUpdateV20,
)
from dmdod.planning.hud_timing_tracker_transition_v14 import TimingUpdate
from dmdod.planning.hud_v3_composite_forecast_source_v30 import (
    HUDTimingCompositeAdapter,
    HUDV3CompositeForecastSource,
    LIFECYCLE_V14,
    LIFECYCLE_V15,
    SOURCE,
)
from dmdod.planning.hud_timing_tracker_lifecycle_v15 import (
    QuantizedLifecycleCausalTimingTracker,
)
from dmdod.planning.hud_timing_tracker_transition_v14 import CausalTimingTracker
from dmdod.planning.timing_fingering_contract_v02 import (
    ABSTAIN,
    ELIGIBLE,
    ELIGIBLE_PARTIAL,
    TimingFrameSequenceValidator,
)


def raw(
    step: int,
    *,
    status: str,
    pending=(),
    new=(),
    reused=(),
    canceled=(),
    elapsed=(),
    reason=None,
):
    now = step * .010
    phase = CountdownPhaseUpdate(
        step=step,
        observed_at_s=now,
        status="hidden",
        displayed_digit=None,
        transition_interval_s=None,
        measured_tick_period_s=None,
        remaining_transitions_to_go=None,
        transition_confirmations=0,
        evidence_age_s=0.0,
        valid=True,
    )
    estimate = CountdownFirstEventEstimate(
        step=step,
        observed_at_s=now,
        status="fixture",
        due_interval_s=None,
        lead_interval_s=None,
        lead_guaranteed=False,
    )
    v14 = TimingUpdate(step, "fixture", (), (), (), (), ())
    return TimingFrameUpdateV20(
        step=step,
        status=status,
        pending=tuple(pending),
        elapsed=tuple(elapsed),
        canceled=tuple(canceled),
        new_ids=tuple(new),
        reused_ids=tuple(reused),
        abstention_reason=reason,
        countdown_phase=phase,
        countdown_estimate=estimate,
        v14_update=v14,
    )


def token(
    local_id: str,
    lower: float,
    upper: float,
    *,
    confirmed=True,
    confirmations=1,
):
    return CompositeToken(
        local_id=local_id,
        due_interval_s=TimeInterval(lower, upper),
        first_seen_local_s=0.0,
        source="fixture",
        confirmations=confirmations,
        confirmed_now=confirmed,
        lead_guaranteed=lower >= .044,
    )


def test_string_ids_map_once_and_lifecycle_remains_valid() -> None:
    adapter = HUDTimingCompositeAdapter()
    first = token("countdown:0", .090, .110)
    unconfirmed = replace(first, confirmed_now=False)
    frames = (
        raw(0, status="eligible_lead_guaranteed", pending=(first,),
            new=("countdown:0",)),
        raw(1, status="phase_invalid_or_stale_frame", pending=(unconfirmed,),
            reused=("countdown:0",), reason="phase_invalid_or_stale_frame"),
        raw(2, status="source_conflict", canceled=(unconfirmed,),
            reason="countdown_and_v14_candidates_are_ambiguous_or_disjoint"),
    )
    updates = tuple(adapter.observe(frame) for frame in frames)
    TimingFrameSequenceValidator().accept(updates[0])
    validator = TimingFrameSequenceValidator()
    for update in updates:
        validator.accept(update)
    assert updates[0].source == SOURCE
    assert updates[0].new_ids == (0,)
    assert updates[0].status == ELIGIBLE
    assert updates[1].status == ABSTAIN
    assert updates[1].reused_ids == (0,)
    assert not updates[1].pending[0].confirmed_now
    assert updates[2].status == ELIGIBLE_PARTIAL
    assert updates[2].canceled_ids == (0,)
    assert adapter.id_map == {"countdown:0": 0}


def test_due_interval_midpoint_and_uncertainty_are_preserved() -> None:
    update = HUDTimingCompositeAdapter().observe(raw(
        0,
        status="eligible_lead_guaranteed",
        pending=(token("countdown:0", 3.979833385, 3.999833385),),
        new=("countdown:0",),
    ))
    translated = update.pending[0]
    assert translated.due_local_s == pytest.approx(3.989833385)
    assert translated.uncertainty_s == pytest.approx(.010)
    assert translated.remaining_lower_s(0) == pytest.approx(3.979833385)


def test_countdown_and_v14_text_ids_never_collide_or_reuse() -> None:
    adapter = HUDTimingCompositeAdapter()
    countdown = token("countdown:0", .090, .100)
    first = adapter.observe(raw(
        0, status="eligible_lead_guaranteed", pending=(countdown,),
        new=("countdown:0",),
    ))
    adapter.observe(raw(1, status="eligible", elapsed=(countdown,)))
    ordinary = token("v14:0", .150, .150)
    third = adapter.observe(raw(
        2, status="eligible", pending=(ordinary,), new=("v14:0",),
    ))
    assert first.new_ids == (0,)
    assert third.new_ids == (1,)
    assert adapter.id_map == {"countdown:0": 0, "v14:0": 1}


def test_abstention_cannot_reconfirm_carried_composite_event() -> None:
    adapter = HUDTimingCompositeAdapter()
    active = token("countdown:0", .090, .100)
    adapter.observe(raw(
        0, status="eligible_lead_guaranteed", pending=(active,),
        new=("countdown:0",),
    ))
    carried = adapter.observe(raw(
        1, status="phase_invalid_or_stale_frame", pending=(active,),
        reused=("countdown:0",), reason="phase_invalid_or_stale_frame",
    ))
    assert carried.status == ABSTAIN
    assert not carried.pending[0].confirmed_now


def test_adapter_rejects_silent_disappearance_and_reused_unknown_id() -> None:
    adapter = HUDTimingCompositeAdapter()
    active = token("countdown:0", .090, .100)
    adapter.observe(raw(
        0, status="eligible_lead_guaranteed", pending=(active,),
        new=("countdown:0",),
    ))
    with pytest.raises(ValueError, match="disappeared"):
        adapter.observe(raw(1, status="hidden"))

    fresh = HUDTimingCompositeAdapter()
    with pytest.raises(ValueError, match="not live"):
        fresh.observe(raw(
            0, status="eligible", pending=(active,), reused=("countdown:0",)
        ))


def test_adapter_policy_surface_has_no_truth_fields() -> None:
    source = inspect.getsource(HUDTimingCompositeAdapter.observe)
    attributes = {
        node.attr for node in ast.walk(ast.parse(textwrap.dedent(source)))
        if isinstance(node, ast.Attribute)
    }
    assert attributes.isdisjoint({
        "targets", "target_time_s", "episode_time_s", "chart_time_s",
        "target_ordinal", "_used", "_missed",
    })


def test_forecast_source_lifecycle_selection_is_explicit_and_default_stable() -> None:
    default = HUDV3CompositeForecastSource()
    legacy = HUDV3CompositeForecastSource(lifecycle_version=LIFECYCLE_V14)
    improved = HUDV3CompositeForecastSource(lifecycle_version=LIFECYCLE_V15)
    assert default.lifecycle_version == LIFECYCLE_V14
    assert type(default.composite._v14) is CausalTimingTracker
    assert type(legacy.composite._v14) is CausalTimingTracker
    assert isinstance(
        improved.composite._v14, QuantizedLifecycleCausalTimingTracker
    )
    with pytest.raises(ValueError, match="unsupported lifecycle version"):
        HUDV3CompositeForecastSource(lifecycle_version="latest")


def test_source_identity_reconciliation_selection_is_forwarded() -> None:
    enabled = HUDV3CompositeForecastSource()
    disabled = HUDV3CompositeForecastSource(
        reconcile_countdown_v14_identity=False
    )
    assert enabled.composite._reconcile_countdown_v14_identity is True
    assert disabled.composite._reconcile_countdown_v14_identity is False
