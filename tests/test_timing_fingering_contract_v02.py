from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace

import pytest

from dmdod.planning import causal_contract as legacy
from dmdod.planning import timing_fingering_contract_v02 as contract


def token(local_id: int = 0, due: float = .055, **overrides):
    return contract.TimingEventToken(
        local_id=local_id, due_local_s=due,
        chord_size=overrides.pop("chord_size", 1), **overrides
    )


def update(step: int = 0, *, pending=(), status=contract.ELIGIBLE,
           source="hud_v1.4", covered=None, **updates):
    if covered is None:
        covered = (step + 8) * contract.CONTROL_DT_S
    return contract.TimingFrameUpdate(
        step=step, status=status, covered_until_s=covered,
        pending=tuple(pending), source=source, **updates
    )


def test_local_clock_uses_only_policy_steps_and_signed_remaining():
    t = token(due=.025)
    assert t.remaining_s(0) == pytest.approx(.025)
    assert t.remaining_s(3) == pytest.approx(-.005)
    f = update(2, pending=(t,), new_ids=(0,))
    assert f.local_clock_s == pytest.approx(.020)
    assert f.remaining_s(0) == pytest.approx(.005)
    with pytest.raises(KeyError):
        f.remaining_s(5)


def test_interval_lower_remaining_uses_uncertainty_without_changing_v14_default():
    interval = token(due=.069, uncertainty_s=.010)
    point = token(due=.069)
    assert interval.remaining_lower_s(0) == pytest.approx(.059)
    assert point.remaining_lower_s(0) == pytest.approx(.069)


def test_two_events_inside_one_control_interval_stay_distinct():
    a = token(0, .055)
    b = token(1, .058)
    frame = update(0, pending=(a, b), new_ids=(0, 1))
    assert frame.pending_ids == (0, 1)
    contract.validate_timing_frames((frame,))


def test_simultaneous_events_must_be_one_chord_token():
    chord = token(5, .040, chord_size=4)
    assert update(0, pending=(chord,), new_ids=(5,)).pending[0].chord_size == 4
    with pytest.raises(ValueError, match="merge exactly simultaneous"):
        update(0, pending=(token(0, .04), token(1, .04)),
               new_ids=(0, 1))


@pytest.mark.parametrize("field,value", [
    ("local_id", -1),
    ("local_id", 1.5),
    ("due_local_s", -0.001),
    ("due_local_s", float("nan")),
    ("due_local_s", float("inf")),
    ("chord_size", 0),
    ("chord_size", 9),
    ("chord_size", True),
    ("confirmations", -1),
    ("confirmations", 1.5),
    ("confirmed_now", 1),
    ("confidence", -0.1),
    ("confidence", 1.1),
    ("confidence", float("nan")),
    ("uncertainty_s", -0.1),
    ("uncertainty_s", float("inf")),
])
def test_invalid_token_fields_rejected(field, value):
    with pytest.raises(ValueError):
        replace(token(), **{field: value})


def test_default_confidence_is_explicitly_unknown_not_invented():
    t = token()
    assert t.confidence is None
    assert t.uncertainty_s is None
    assert t.confirmations == 1 and t.confirmed_now is True


def test_frozen_token_and_frame_are_immutable():
    t = token()
    f = update(0, pending=(t,), new_ids=(0,))
    with pytest.raises(FrozenInstanceError):
        t.local_id = 100
    with pytest.raises(FrozenInstanceError):
        f.step = 100


def test_explicit_lifecycle_create_reuse_abstain_retire_and_new():
    validator = contract.TimingFrameSequenceValidator()
    f0 = update(0, pending=(token(0, .055),), new_ids=(0,))
    f1 = update(1, pending=(token(0, .057, confirmations=2),
                            token(1, .059)), new_ids=(1,), reused_ids=(0,))
    f2 = update(
        2, status=contract.ABSTAIN, covered=.01,
        pending=(token(0, .057, confirmations=2, confirmed_now=False),
                 token(1, .059, confirmed_now=False)),
        reused_ids=(0, 1), abstention_reason="pause_unobservable"
    )
    f3 = update(3, pending=(), canceled_ids=(0,), elapsed_ids=(1,))
    f4 = update(4, pending=(token(2, .140),), new_ids=(2,))
    for frame in (f0, f1, f2, f3, f4):
        validator.accept(frame)
    assert validator.last_step == 4
    assert validator.live_ids == frozenset((2,))
    contract.validate_timing_frames((f0, f1, f2, f3, f4))


def test_abstention_cannot_create_or_cancel_or_reconfirm():
    with pytest.raises(ValueError, match="cannot create"):
        update(0, status=contract.ABSTAIN, pending=(token(),), new_ids=(0,),
               abstention_reason="pause")
    with pytest.raises(ValueError, match="cannot create"):
        update(0, status=contract.ABSTAIN, canceled_ids=(0,),
               abstention_reason="pause")
    with pytest.raises(ValueError, match="cannot reconfirm"):
        update(0, status=contract.ABSTAIN,
               pending=(token(),), reused_ids=(0,),
               abstention_reason="pause")
    with pytest.raises(ValueError, match="requires an explicit reason"):
        update(0, status=contract.ABSTAIN, pending=())


def test_pending_and_id_partitions_are_strict():
    a, b = token(0, .05), token(1, .06)
    with pytest.raises(ValueError, match="partition pending"):
        update(0, pending=(a, b), new_ids=(0,))
    with pytest.raises(ValueError, match="overlapping lifecycle"):
        update(0, pending=(a,), new_ids=(0,), canceled_ids=(0,))
    with pytest.raises(ValueError, match="sorted unique"):
        update(0, pending=(a, b), new_ids=(1, 0))
    with pytest.raises(ValueError, match="duplicate local event"):
        update(0, pending=(a, replace(a, due_local_s=.06)),
               new_ids=(0,))
    with pytest.raises(ValueError, match="must be sorted"):
        update(0, pending=(b, a), new_ids=(0, 1))


def test_nonconsecutive_frames_and_source_change_fail():
    validator = contract.TimingFrameSequenceValidator()
    validator.accept(update(4))
    with pytest.raises(ValueError, match="consecutive"):
        validator.accept(update(6))
    with pytest.raises(ValueError, match="source cannot change"):
        validator.accept(update(5, source="oracle"))


def test_reuse_of_retired_id_and_id_rewind_fail():
    validator = contract.TimingFrameSequenceValidator()
    validator.accept(update(0, pending=(token(10),), new_ids=(10,)))
    validator.accept(update(1, canceled_ids=(10,)))
    with pytest.raises(ValueError, match="monotonically"):
        validator.accept(update(2, pending=(token(10),), new_ids=(10,)))
    with pytest.raises(ValueError, match="no live prior"):
        validator.accept(update(2, pending=(token(10),), reused_ids=(10,)))
    validator.accept(update(2, pending=(token(11),), new_ids=(11,)))


def test_cannot_silently_drop_forecast_when_hud_abstains():
    validator = contract.TimingFrameSequenceValidator()
    validator.accept(update(0, pending=(token(0),), new_ids=(0,)))
    bad = update(1, pending=(), status=contract.ABSTAIN,
                 abstention_reason="ambiguous_pause")
    with pytest.raises(ValueError, match="disappeared"):
        validator.accept(bad)
    good = update(1, pending=(token(0, confirmed_now=False),),
                  reused_ids=(0,), status=contract.ABSTAIN,
                  abstention_reason="ambiguous_pause")
    validator.accept(good)
    assert validator.live_ids == frozenset((0,))


def test_only_prior_live_ids_can_be_expired_or_canceled():
    validator = contract.TimingFrameSequenceValidator()
    with pytest.raises(ValueError, match="non-live"):
        validator.accept(update(0, canceled_ids=(0,)))
    validator.accept(update(0, pending=(token(),), new_ids=(0,)))
    with pytest.raises(ValueError, match="non-live"):
        validator.accept(update(1, elapsed_ids=(3,)))
    validator.accept(update(1, elapsed_ids=(0,)))


def test_invalid_frames_do_not_advance_validator_state():
    validator = contract.TimingFrameSequenceValidator()
    validator.accept(update(0, pending=(token(),), new_ids=(0,)))
    with pytest.raises(ValueError):
        validator.accept(update(1, pending=(token(2),), new_ids=(2,)))
    assert validator.live_ids == frozenset((0,))
    assert validator.last_step == 0
    validator.accept(update(1, pending=(token(),), reused_ids=(0,)))


def test_no_privileged_evaluator_or_chart_fields():
    forbidden = (
        "target_index", "target_ordinal", "floor_index", "target_time_s",
        "chart_time", "judgement", "_used", "_missed", "segment.targets",
    )
    for typ in (contract.TimingEventToken, contract.TimingFrameUpdate):
        names = {field.name for field in fields(typ)}
        assert not names.intersection(forbidden)
    assert contract.CONTROL_DT_S == .010
    assert legacy.HUD_DIM == 263 and legacy.KEY_COUNT == 8


def test_legacy_contract_not_reinterpreted():
    assert contract.CONTRACT_VERSION == "timing-fingering-v0.2"
    assert contract.MAX_CHORD_SIZE == 8
    assert legacy.CONTROL_DT_S == contract.CONTROL_DT_S
    assert tuple(field.name for field in fields(legacy.TimingPrediction)) == ("events",)


@pytest.mark.parametrize("kwargs", [
    {"step": -1},
    {"step": 0.5},
    {"status": "unknown"},
    {"covered_until_s": float("nan")},
    {"covered_until_s": -1.0},
    {"source": ""},
    {"source": "  "},
    {"version": "unversioned"},
    {"abstention_reason": "not allowed"},
])
def test_frame_rejects_invalid_metadata(kwargs):
    with pytest.raises(ValueError):
        replace(update(), **kwargs)


def test_eligible_coverage_cannot_end_before_current_frame():
    with pytest.raises(ValueError, match="coverage"):
        update(10, covered=.01)
    assert update(10, status=contract.ABSTAIN, covered=.01,
                  abstention_reason="no_motion").status == contract.ABSTAIN
