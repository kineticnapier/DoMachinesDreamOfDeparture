from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from dmdod.adofai.playable import PlayableChartSegment, PlayableChartTarget
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher
from dmdod.training import prestige1_autoplay_sequence_student as student


def _teacher(*times: float) -> AutoplayTimingTeacher:
    targets = tuple(
        PlayableChartTarget(
            ordinal=i, floor_index=i + 1, chart_time_s=value,
            episode_time_s=value, bpm=120.0,
        )
        for i, value in enumerate(times)
    )
    return AutoplayTimingTeacher.from_segment(PlayableChartSegment(
        chart=SimpleNamespace(), targets=targets, start_chart_s=0.0,
        end_chart_s=max(times) + 1.0, pitch_ratio=1.0,
    ))


def _sequence(rows: list[student.FutureEventLabels]) -> student.MultiEventTimingSequence:
    offsets, present, chords, overflow = student.pack_label_rows(rows)
    return student.MultiEventTimingSequence(
        chart_sha256="a" * 64,
        visible=torch.randn(len(rows), 263),
        offsets_s=offsets,
        present=present,
        chord_sizes=chords,
        overflow=overflow,
        exact_event_times_s=(0.01234567, 0.01456789),
        special_event_tags=((), ("Twirl",)),
    )


def test_exact_future_labels_keep_sub_millisecond_times_and_multiple_events():
    exact = _teacher(0.01234567, 0.01456789, 0.01987654, 0.10000003)
    label = student.future_labels(exact, 0.01000001)
    assert label.offsets_s == pytest.approx(
        (0.00234566, 0.00456788, 0.00987653), abs=1e-10
    )
    assert all((offset * 1000) % 1 != 0 for offset in label.offsets_s)
    assert not label.overflow


def test_more_than_one_event_inside_one_control_interval_is_preserved():
    exact = _teacher(0.002, 0.005, 0.008, 0.011)
    label = student.future_labels(exact, 0.0)
    assert label.offsets_s[:3] == pytest.approx((0.002, 0.005, 0.008))
    assert sum(offset <= student.CONTROL_DT_S for offset in label.offsets_s) == 3


def test_events_at_now_are_not_misreported_as_useful_future_lead():
    exact = _teacher(0.0100000005, 0.020123456)
    label = student.future_labels(exact, 0.0100000005)
    assert label.offsets_s == pytest.approx((0.0101234555,))


def test_model_is_causal_prefix_invariant_and_outputs_ordered_sequence():
    torch.manual_seed(7)
    model = student.MultiEventTimingStudent(hidden=24).eval()
    visible = torch.randn(9, 263)
    with torch.no_grad():
        short_offsets, short_presence = model(visible[:5])
        full_offsets, full_presence = model(visible)
    assert torch.allclose(short_offsets, full_offsets[:5], atol=1e-6, rtol=1e-6)
    assert torch.allclose(short_presence, full_presence[:5], atol=1e-6, rtol=1e-6)
    assert bool(torch.all(full_offsets[:, 1:] > full_offsets[:, :-1]))
    assert bool(torch.all((full_offsets > 0) & (full_offsets < student.HORIZON_S)))
    with pytest.raises(ValueError, match="263"):
        model(torch.zeros(4, 264))


def test_loss_uses_all_present_slots_and_backpropagates():
    rows = [
        student.FutureEventLabels((0.0023, 0.0058, 0.0091), (1, 1, 1), False),
        student.FutureEventLabels((0.0123,), (2,), False),
        student.FutureEventLabels((), (), False),
    ]
    sequence = _sequence(rows)
    model = student.MultiEventTimingStudent(hidden=16)
    offsets, logits = model(sequence.visible)
    loss = student.sequence_loss(offsets, logits, sequence, 0, 3)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_fixed_train_loop_and_threshold_selection_use_only_supplied_sequences():
    torch.manual_seed(17)
    fit = [_sequence([
        student.FutureEventLabels((0.012, 0.031), (1, 1), False),
        student.FutureEventLabels((0.022,), (1,), False),
        student.FutureEventLabels((), (), False),
    ])]
    model = student.MultiEventTimingStudent(hidden=8)
    history = student.train_student(model, fit, epochs=2, seed=3)
    threshold, rows = student.select_threshold(model, fit)
    assert len(history) == 2
    assert threshold in student.THRESHOLD_CANDIDATES
    assert [row["threshold"] for row in rows] == list(student.THRESHOLD_CANDIDATES)
    assert all(0 <= row["slot_f1"] <= 1 for row in rows)


def test_decoder_confidence_prefix_and_abstention():
    offsets = torch.linspace(0.004, 0.076, student.MAX_EVENTS)
    logits = torch.full((student.MAX_EVENTS,), -8.0)
    assert student.decode_forecast(
        offsets, logits, confidence_threshold=0.8
    ).abstained
    logits[:3] = torch.tensor([5.0, 4.0, 3.0])
    forecast = student.decode_forecast(offsets, logits, confidence_threshold=0.8)
    assert not forecast.abstained
    assert len(forecast.offsets_s) == 3
    assert forecast.confidences[0] > forecast.confidences[2] > 0.8


def test_sequence_rejects_oracle_channel_or_invalid_slot_order():
    rows = [student.FutureEventLabels((0.020, 0.030), (1, 1), False)]
    sequence = _sequence(rows)
    with pytest.raises(ValueError, match="sequence"):
        student.MultiEventTimingSequence(
            sequence.chart_sha256,
            torch.zeros(1, 264),
            sequence.offsets_s,
            sequence.present,
            sequence.chord_sizes,
            sequence.overflow,
            sequence.exact_event_times_s,
            sequence.special_event_tags,
        )
    bad = sequence.offsets_s.clone()
    bad[0, 1] = bad[0, 0]
    with pytest.raises(ValueError, match="ordered"):
        student.MultiEventTimingSequence(
            sequence.chart_sha256, sequence.visible, bad, sequence.present,
            sequence.chord_sizes, sequence.overflow,
            sequence.exact_event_times_s, sequence.special_event_tags,
        )
