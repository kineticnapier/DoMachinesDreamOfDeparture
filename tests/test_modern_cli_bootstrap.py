from __future__ import annotations

import pytest

from dmdod.modern_cli_bootstrap import (
    BootstrapLiveModernTrainerConsole,
    _stable_parent_rate,
)
from dmdod.training_progress import TrainingProgressEvent


class _FakeBar:
    def __init__(self, total: int) -> None:
        self.total = float(total)
        self.n = 0.0
        self.postfix = ""
        self.closed = False
        self.eta_resets = 0

    def update(self, delta: float) -> None:
        self.n += float(delta)

    def refresh(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def set_postfix_str(self, text: str) -> None:
        self.postfix = str(text)

    def reset_eta_origin(self) -> None:
        self.eta_resets += 1


def _console() -> BootstrapLiveModernTrainerConsole:
    console = BootstrapLiveModernTrainerConsole(
        rounds=48,
        round_epochs=12,
        bootstrap_epochs=64,
    )
    console.enabled = True
    factory = lambda *, total, desc, colour, position: _FakeBar(total)
    console._bar = factory  # type: ignore[method-assign]
    console._live_bar = factory  # type: ignore[method-assign]
    return console


def test_bootstrap_eval_owns_single_stage_bar() -> None:
    console = _console()

    console._on_progress(
        TrainingProgressEvent(
            "bootstrap_eval_start",
            {
                "total": 152,
                "waves": 13,
                "start_micro": 48,
                "batch_size": 12,
            },
        )
    )
    assert console._bootstrap_eval_active
    assert console._stage is not None
    assert console._stage.total == 152
    assert "wave=0/13" in console._stage.postfix
    assert "start=48" in console._stage.postfix

    console._on_progress(
        TrainingProgressEvent(
            "bootstrap_eval_step",
            {
                "current": 24,
                "total": 152,
                "wave": 2,
                "waves": 13,
                "phase": "start-micro",
                "evaluated_targets": 96,
                "total_targets": 2000,
                "overloaded": False,
            },
        )
    )
    assert console._stage.n == 24
    assert "wave=2/13" in console._stage.postfix
    assert "start-micro" in console._stage.postfix
    assert "targets=96/2000" in console._stage.postfix


def test_bootstrap_eval_suppresses_generic_eval_and_closes_on_done() -> None:
    console = _console()
    console._on_progress(
        TrainingProgressEvent(
            "bootstrap_eval_start",
            {"total": 24, "waves": 2, "start_micro": 12, "batch_size": 12},
        )
    )
    bar = console._stage
    assert bar is not None

    console._on_progress(
        TrainingProgressEvent(
            "eval_start",
            {"total": 12, "current": 0, "label": "Eval", "cached": 0, "workers": 12},
        )
    )
    assert console._stage is bar
    assert console._stage.total == 24

    console._on_progress(
        TrainingProgressEvent(
            "bootstrap_eval_done",
            {"current": 12, "total": 24, "status": "PRUNE", "reason": "safety"},
        )
    )
    assert bar.n == 12
    assert bar.closed
    assert console._stage is None
    assert not console._bootstrap_eval_active


def test_bootstrap_eval_fraction_flows_into_outer_bootstrap_bar() -> None:
    console = _console()
    console._bootstrap = _FakeBar(64)
    console._bootstrap.n = 7.0
    console._fraction_parent_name = "_bootstrap"
    console._fraction_parent_base = 7.0

    console._on_progress(
        TrainingProgressEvent(
            "bootstrap_eval_start",
            {"total": 100, "waves": 10, "start_micro": 48, "batch_size": 12},
        )
    )
    console._on_progress(
        TrainingProgressEvent(
            "bootstrap_eval_step",
            {
                "current": 50,
                "total": 100,
                "wave": 5,
                "waves": 10,
                "phase": "anchors",
                "evaluated_targets": 500,
                "total_targets": 1000,
                "overloaded": False,
            },
        )
    )

    # BC owns 10%; halfway through eval is 10% + 45% = 55% of epoch 8.
    assert console._bootstrap.n == pytest.approx(7.55)

    console._on_progress(
        TrainingProgressEvent(
            "bootstrap_eval_done",
            {"current": 50, "total": 100, "status": "PRUNE", "reason": "completion"},
        )
    )
    assert console._bootstrap.n == pytest.approx(8.0)


def test_parent_eta_waits_for_one_logical_unit() -> None:
    assert _stable_parent_rate(0.0, 10.0) is None
    assert _stable_parent_rate(0.999, 100.0) is None
    assert _stable_parent_rate(1.0, 100.0) == pytest.approx(0.01)


def test_parent_eta_uses_cumulative_fractional_rate_not_last_update() -> None:
    # 2.5 logical epochs over 250 seconds is 0.01 epoch/s regardless of how
    # tiny or bursty the individual child-bar updates were.
    assert _stable_parent_rate(2.5, 250.0) == pytest.approx(0.01)
    assert _stable_parent_rate(2.5, 0.0) is None


def test_parent_eta_origin_can_be_reset_after_resume() -> None:
    console = _console()
    bar = _FakeBar(48)
    bar.n = 17.0
    console._round = bar

    console._reset_parent_eta("_round")

    assert bar.eta_resets == 1
