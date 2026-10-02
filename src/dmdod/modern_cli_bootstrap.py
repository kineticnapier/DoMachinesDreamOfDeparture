from __future__ import annotations

"""Bootstrap-aware live terminal dashboard for the v1.0 trainer."""

from typing import Any

from .modern_cli_live import (
    LiveModernTrainerConsole,
    _INTEGER_COUNT_BAR_FORMAT,
    _IntegerCountTqdm,
)


PARENT_ETA_VERSION = "cumulative-fractional-v1"


def _stable_parent_rate(progress: float, elapsed_s: float) -> float | None:
    """Return a cumulative parent-bar rate after one logical unit is observed.

    Parent bars receive many tiny fractional updates from BC/guard/bootstrap
    child bars. tqdm's EMA treats those tiny deltas as independent samples, so
    its instantaneous ETA can jump wildly when a child phase changes speed.
    Using cumulative logical progress avoids that phase-boundary distortion.
    The first unit intentionally has no ETA: before one complete epoch/round we
    do not have enough evidence to extrapolate the remaining run.
    """

    progress = float(progress)
    elapsed_s = float(elapsed_s)
    if progress < 1.0 - 1e-9 or elapsed_s <= 0.0:
        return None
    return progress / elapsed_s


if _IntegerCountTqdm is not None:
    class _StableParentTqdm(_IntegerCountTqdm):
        """Persistent tqdm whose ETA is stable under fractional child updates."""

        def __init__(self, *args, **kwargs) -> None:
            # tqdm may render during its own constructor, so make the override
            # safe before delegating to it.
            self._eta_origin_n: float | None = None
            self._eta_origin_elapsed = 0.0
            super().__init__(*args, **kwargs)
            self.reset_eta_origin()

        def reset_eta_origin(self) -> None:
            """Start a new ETA sample window from the bar's current position."""

            values = super().format_dict
            self._eta_origin_n = float(self.n)
            self._eta_origin_elapsed = float(values.get("elapsed", 0.0) or 0.0)

        @property
        def format_dict(self):
            values = super().format_dict
            origin_n = self._eta_origin_n
            if origin_n is not None:
                progress = max(0.0, float(self.n) - origin_n)
                elapsed = max(
                    0.0,
                    float(values.get("elapsed", 0.0) or 0.0) - self._eta_origin_elapsed,
                )
                values["rate"] = _stable_parent_rate(progress, elapsed)
            return values
else:  # pragma: no cover - modern TTY mode is disabled without tqdm
    _StableParentTqdm = None


class BootstrapLiveModernTrainerConsole(LiveModernTrainerConsole):
    """Show turbo bootstrap candidate evaluation on the live dashboard."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._bootstrap_eval_active = False

    def _bar(self, *, total: int, desc: str, colour: str, position: int):
        """Use cumulative-rate ETA for persistent parent bars only."""

        if _StableParentTqdm is None:
            return super()._bar(total=total, desc=desc, colour=colour, position=position)
        return _StableParentTqdm(
            total=max(1, int(total)),
            desc=desc,
            unit="step",
            dynamic_ncols=True,
            leave=True,
            colour=colour,
            position=position,
            file=self.stream,
            mininterval=0.10,
            miniters=0,
            bar_format=_INTEGER_COUNT_BAR_FORMAT,
        )

    def _reset_parent_eta(self, name: str) -> None:
        bar = getattr(self, name, None)
        reset = getattr(bar, "reset_eta_origin", None)
        if callable(reset):
            reset()

    def _handle(self, event, raw: str) -> bool:
        handled = super()._handle(event, raw)
        # A resumed Rounds bar is first advanced to historical progress.  That
        # history happened before this process started, so do not divide it by
        # the current process's tiny elapsed time when estimating ETA.
        if event.kind == "resume":
            self._reset_parent_eta("_round")
        return handled

    def _on_progress(self, event) -> None:
        if not self.enabled:
            return
        values = event.values
        kind = event.kind

        if kind == "bootstrap_eval_start":
            self._bootstrap_eval_active = True
            if self._fraction_parent_name != "_bootstrap":
                self._begin_fractional_parent_unit()
            self._set_second_phase_parent_progress(0.0)
            total = int(values.get("total", 1))
            self._new_stage(total=max(1, total), desc="BS eval", colour="cyan")
            self._stage.set_postfix_str(
                f"wave=0/{int(values.get('waves', 0))} "
                f"start={int(values.get('start_micro', 0))} "
                f"batch={int(values.get('batch_size', 0))}"
            )
            return

        if kind == "bootstrap_eval_step":
            total = max(1, int(values.get("total", 1)))
            current = int(values.get("current", 0))
            if self._stage is None:
                self._new_stage(total=total, desc="BS eval", colour="cyan")
            self._advance_to(self._stage, current)
            self._set_second_phase_parent_progress(current / total)
            over = " over" if values.get("overloaded") else ""
            self._stage.set_postfix_str(
                f"wave={int(values.get('wave', 0))}/{int(values.get('waves', 0))} "
                f"{values.get('phase', 'eval')} "
                f"targets={int(values.get('evaluated_targets', 0))}/"
                f"{int(values.get('total_targets', 0))}{over}"
            )
            return

        if kind == "bootstrap_eval_done":
            if self._stage is not None:
                self._advance_to(self._stage, int(values.get("current", 0)))
                status = str(values.get("status", "FULL"))
                reason = values.get("reason")
                self._stage.set_postfix_str(
                    status if not reason else f"{status}:{reason}"
                )
                self._close_live("_stage")
            self._finish_fractional_parent_unit()
            self._bootstrap_eval_active = False
            return

        # The wrapped evaluator may itself emit generic eval events for each
        # worker-sized wave. Hide those while the bootstrap-specific bar owns
        # the display, otherwise the bar would reset on every wave.
        if self._bootstrap_eval_active and kind in {"eval_start", "eval_step", "eval_done"}:
            return

        super()._on_progress(event)
