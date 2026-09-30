from __future__ import annotations

import pytest

from dmdod.modern_cli_live import LiveModernTrainerConsole
from dmdod.training_progress import TrainingProgressEvent


class _FakeBar:
    def __init__(self, total: int) -> None:
        self.total = float(total)
        self.n = 0.0
        self.postfix = ""
        self.closed = False

    def update(self, delta: float) -> None:
        self.n += float(delta)

    def refresh(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def set_postfix_str(self, text: str) -> None:
        self.postfix = str(text)


def _console() -> LiveModernTrainerConsole:
    console = LiveModernTrainerConsole(
        rounds=48,
        round_epochs=12,
        bootstrap_epochs=64,
    )
    console.enabled = True
    factory = lambda *, total, desc, colour, position: _FakeBar(total)
    console._bar = factory  # type: ignore[method-assign]
    console._live_bar = factory  # type: ignore[method-assign]
    return console


def test_bc_uses_single_global_bar_without_sequence_detail() -> None:
    console = _console()

    console._on_progress(
        TrainingProgressEvent(
            "bc_start",
            {"sequences": 142, "chunks": 500, "reverse": False},
        )
    )
    assert console._stage is not None
    assert console._detail is None
    assert console._stage.total == 500

    console._on_progress(
        TrainingProgressEvent(
            "bc_sequence_start",
            {
                "index": 12,
                "total": 142,
                "source": "Example Very Long Chart Name",
                "reverse": False,
            },
        )
    )
    assert console._detail is None
    assert "seq=12/142" in console._stage.postfix
    assert "Example Very Long Chart Name" in console._stage.postfix

    console._on_progress(
        TrainingProgressEvent(
            "bc_chunk",
            {
                "global_chunk": 123,
                "global_total": 500,
                "sequence": 12,
                "sequence_total": 142,
                "loss": 0.01234,
                "source": "Example Very Long Chart Name",
                "reverse": False,
            },
        )
    )
    assert console._stage.n == 123
    assert console._detail is None
    assert "seq=12/142" in console._stage.postfix
    assert "loss=0.0123" in console._stage.postfix


def test_bc_done_closes_single_bar() -> None:
    console = _console()
    console._on_progress(
        TrainingProgressEvent(
            "bc_start",
            {"sequences": 2, "chunks": 10, "reverse": True},
        )
    )
    bar = console._stage
    assert bar is not None

    console._on_progress(
        TrainingProgressEvent("bc_sequence_done", {"index": 1, "total": 2})
    )
    assert console._stage is bar
    assert console._detail is None

    console._on_progress(
        TrainingProgressEvent("bc_done", {"loss": 0.0042, "reverse": True})
    )
    assert bar.n == 10
    assert bar.closed
    assert console._stage is None
    assert console._detail is None


def test_child_progress_flows_fractionally_into_epoch_and_round() -> None:
    console = _console()
    console._round = _FakeBar(48)
    console._epoch = _FakeBar(12)
    console._current_round = 5

    console._on_progress(
        TrainingProgressEvent(
            "bc_start",
            {"sequences": 2, "chunks": 10, "reverse": False},
        )
    )
    console._on_progress(
        TrainingProgressEvent(
            "bc_chunk",
            {
                "global_chunk": 5,
                "global_total": 10,
                "sequence": 1,
                "sequence_total": 2,
                "loss": 0.01,
                "source": "chart",
                "reverse": False,
            },
        )
    )
    # BC owns the first 10% of one epoch, so halfway through BC is 0.05 epoch.
    assert console._epoch.n == pytest.approx(0.05)
    assert console._round.n == pytest.approx(4.0 + 0.05 / 12.0)

    console._on_progress(TrainingProgressEvent("bc_done", {"loss": 0.01}))
    console._on_progress(
        TrainingProgressEvent(
            "guard_start",
            {"alphas": 3, "validation_total": 10, "anchor_total": 20},
        )
    )
    console._on_progress(
        TrainingProgressEvent(
            "guard_phase_step",
            {"phase": "anchor", "current": 10, "total": 20, "states": 3},
        )
    )

    # Halfway through the third guard phase => 2.5 / 3 through Guard.
    expected_epoch = 0.10 + 0.90 * (2.5 / 3.0)
    assert console._stage is not None
    assert console._stage.n == pytest.approx(2.5)
    assert console._epoch.n == pytest.approx(expected_epoch)
    assert console._round.n == pytest.approx(4.0 + expected_epoch / 12.0)

    console._on_progress(TrainingProgressEvent("guard_done", {"accepted": True, "alpha": 0.5}))
    assert console._epoch.n == pytest.approx(1.0)
    assert console._round.n == pytest.approx(4.0 + 1.0 / 12.0)


def test_fractional_advance_does_not_overshoot_integer_boundary() -> None:
    console = _console()
    bar = _FakeBar(12)
    bar.n = 0.95

    console._advance_to(bar, 1)

    assert bar.n == pytest.approx(1.0)
