from __future__ import annotations

from dmdod.modern_cli_live import LiveModernTrainerConsole
from dmdod.training_progress import TrainingProgressEvent


class _FakeBar:
    def __init__(self, total: int) -> None:
        self.total = int(total)
        self.n = 0
        self.postfix = ""
        self.closed = False

    def update(self, delta: int) -> None:
        self.n += int(delta)

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
    console._live_bar = lambda *, total, desc, colour, position: _FakeBar(total)  # type: ignore[method-assign]
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
