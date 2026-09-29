from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v080 as v080  # noqa: E402
import train_real_chart_v080_fast as fast  # noqa: E402


def test_anchor_batches_are_two_wide_and_preserve_order():
    items = list(range(7))
    assert list(fast._anchor_batches(items, 2)) == [
        (0, [0, 1]),
        (2, [2, 3]),
        (4, [4, 5]),
        (6, [6]),
    ]


def test_anchor_batches_reject_nonpositive_size():
    try:
        list(fast._anchor_batches([1], 0))
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("nonpositive anchor batch size must be rejected")


def test_bootstrap_state_evaluates_anchor_and_validation_in_one_wave(monkeypatch):
    anchor_segments = [object(), object(), object()]
    validation_segments = [object(), object()]
    expected = ["a0", "a1", "a2", "v0", "v1"]
    calls = []

    def fake_evaluate(model, state, segments, *, same_hand, control_dt_s):
        calls.append((model, state, list(segments), same_hand, control_dt_s))
        return list(expected)

    monkeypatch.setattr(v080, "_evaluate_one_state_many", fake_evaluate)
    anchors, validations = fast._evaluate_bootstrap_state(
        "model",
        "state",
        anchor_segments,
        validation_segments,
        same_hand=True,
        control_dt_s=0.01,
    )

    assert len(calls) == 1
    assert calls[0][2] == [*anchor_segments, *validation_segments]
    assert anchors == expected[:3]
    assert validations == expected[3:]


def test_fast_wrapper_keeps_v080_checkpoint_identity():
    assert v080.TRAINER_VERSION == "0.8.0-multichart-hud"
    assert v080.CHECKPOINT_FORMAT_VERSION == 15
    assert fast.FAST_EVAL_VERSION.startswith("v080-")
