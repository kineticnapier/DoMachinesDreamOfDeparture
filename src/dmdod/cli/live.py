from __future__ import annotations

"""Fine-grained live dashboard layered on top of ``modern_cli``."""

from typing import Any

from . import modern_cli as base
from .modern import ModernTrainerConsole
from .training_progress import TrainingProgressEvent, subscribe, unsubscribe


# BC is normally much shorter than guard/evaluation.  This share is only a UI
# interpolation weight so parent tqdm bars can update their built-in ETA while
# a child stage is running; it never affects training semantics.
_BC_PARENT_SHARE = 0.10
_INTEGER_COUNT_BAR_FORMAT = (
    "{l_bar}{bar}| {display_n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)


if base.tqdm is not None:
    class _IntegerCountTqdm(base.tqdm):
        """tqdm that may track fractional work while displaying whole units."""

        @property
        def format_dict(self):
            values = super().format_dict
            values["display_n_fmt"] = str(int(float(self.n) + 1e-12))
            return values
else:  # pragma: no cover - modern TTY mode is disabled without tqdm
    _IntegerCountTqdm = None


class LiveModernTrainerConsole(ModernTrainerConsole):
    """Add current-stage and current-detail bars fed by trainer callbacks."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stage = None
        self._detail = None
        self._detail_phase: str | None = None
        self._guard_active = False
        self._fraction_parent_name: str | None = None
        self._fraction_parent_base = 0.0

    def _bar(self, *, total: int, desc: str, colour: str, position: int):
        assert _IntegerCountTqdm is not None
        return _IntegerCountTqdm(
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

    def _live_bar(self, *, total: int, desc: str, colour: str, position: int):
        assert _IntegerCountTqdm is not None
        return _IntegerCountTqdm(
            total=max(1, int(total)),
            desc=desc,
            unit="step",
            dynamic_ncols=True,
            leave=False,
            colour=colour,
            position=position,
            file=self.stream,
            mininterval=0.10,
            miniters=0,
            bar_format=_INTEGER_COUNT_BAR_FORMAT,
        )

    @staticmethod
    def _advance_to(bar, value: float) -> None:
        """Advance a tqdm to an exact (possibly fractional) logical position."""

        target = max(0.0, min(float(value), float(bar.total)))
        delta = target - float(bar.n)
        if delta > 1e-12:
            bar.update(delta)
        elif delta < -1e-12:
            bar.n = target
            bar.refresh()

    def _close_live(self, name: str) -> None:
        bar = getattr(self, name)
        if bar is not None:
            bar.close()
            setattr(self, name, None)
        if name == "_detail":
            self._detail_phase = None

    def _new_stage(self, *, total: int, desc: str, colour: str) -> None:
        self._close_live("_detail")
        self._close_live("_stage")
        self._stage = self._live_bar(total=total, desc=desc, colour=colour, position=2)

    def _new_detail(self, *, total: int, desc: str, colour: str, phase: str | None = None) -> None:
        self._close_live("_detail")
        self._detail = self._live_bar(total=total, desc=desc, colour=colour, position=3)
        self._detail_phase = phase

    @staticmethod
    def _short(text: object, width: int = 34) -> str:
        value = str(text)
        return value if len(value) <= width else value[-width:]

    def _begin_fractional_parent_unit(self) -> None:
        """Capture the integer base of the current bootstrap/round epoch."""

        if self._epoch is not None:
            self._fraction_parent_name = "_epoch"
            self._fraction_parent_base = float(int(float(self._epoch.n) + 1e-9))
            return

        # Bootstrap progress lines are only printed after a candidate finishes,
        # so create the outer bar as soon as the first BC phase begins.
        if self._bootstrap is None and self._current_round == 0 and self.bootstrap_epochs > 0:
            self._bootstrap = self._bar(
                total=self.bootstrap_epochs,
                desc="Bootstrap",
                colour="blue",
                position=0,
            )
        if self._bootstrap is not None:
            self._fraction_parent_name = "_bootstrap"
            self._fraction_parent_base = float(int(float(self._bootstrap.n) + 1e-9))
            return

        self._fraction_parent_name = None
        self._fraction_parent_base = 0.0

    def _sync_round_from_epoch(self) -> None:
        if self._round is None or self._epoch is None or self._current_round <= 0:
            return
        round_target = (
            float(self._current_round - 1)
            + float(self._epoch.n) / max(1.0, float(self.round_epochs))
        )
        self._advance_to(self._round, round_target)

    def _set_fractional_parent_progress(self, fraction: float) -> None:
        """Set progress inside the current parent unit and bubble it upward."""

        if self._fraction_parent_name is None:
            return
        parent = getattr(self, self._fraction_parent_name, None)
        if parent is None:
            return
        fraction = max(0.0, min(1.0, float(fraction)))
        self._advance_to(parent, self._fraction_parent_base + fraction)
        if self._fraction_parent_name == "_epoch":
            self._sync_round_from_epoch()

    def _set_bc_parent_progress(self, fraction: float) -> None:
        self._set_fractional_parent_progress(_BC_PARENT_SHARE * max(0.0, min(1.0, fraction)))

    def _set_second_phase_parent_progress(self, fraction: float) -> None:
        fraction = max(0.0, min(1.0, fraction))
        self._set_fractional_parent_progress(
            _BC_PARENT_SHARE + (1.0 - _BC_PARENT_SHARE) * fraction
        )

    def _finish_fractional_parent_unit(self) -> None:
        self._set_fractional_parent_progress(1.0)

    def _on_progress(self, event: TrainingProgressEvent) -> None:
        if not self.enabled:
            return
        values = event.values
        kind = event.kind

        if kind == "rollout_start":
            mode = str(values.get("mode", "rollout"))
            desc = "Expert rollout" if mode == "expert" else "Mix rollout"
            colour = "cyan" if mode == "expert" else "yellow"
            self._new_stage(total=int(values.get("total", 1)), desc=desc, colour=colour)
            parts = [self._short(values.get("source", ""), 28)]
            if values.get("teacher_fraction") is not None:
                parts.append(f"teacher={float(values['teacher_fraction']) * 100:.0f}%")
            self._stage.set_postfix_str(" ".join(part for part in parts if part))
            return

        if kind == "rollout_step" and self._stage is not None:
            self._advance_to(self._stage, int(values.get("current", 0)))
            return

        if kind == "rollout_done":
            self._close_live("_detail")
            self._close_live("_stage")
            return

        # BC sequences are short enough that a second per-sequence tqdm bar is
        # mostly terminal churn. Keep one global chunk bar and surface the
        # current sequence/source/loss in its postfix instead.
        if kind == "bc_start":
            chunks = int(values.get("chunks", 1))
            self._begin_fractional_parent_unit()
            self._set_bc_parent_progress(0.0)
            self._new_stage(total=chunks, desc="BC", colour="blue")
            parity = "rev" if values.get("reverse") else "fwd"
            self._stage.set_postfix_str(
                f"{parity} seq=0/{int(values.get('sequences', 0))}"
            )
            return

        if kind == "bc_sequence_start":
            if self._stage is not None:
                index = int(values.get("index", 0))
                total = int(values.get("total", 0))
                parity = "rev" if values.get("reverse") else "fwd"
                source = self._short(values.get("source", ""), 28)
                self._stage.set_postfix_str(
                    f"{parity} seq={index}/{total} {source}".rstrip()
                )
            return

        if kind == "bc_chunk":
            if self._stage is not None:
                current = int(values.get("global_chunk", 0))
                total = max(1, int(values.get("global_total", self._stage.total)))
                self._advance_to(self._stage, current)
                self._set_bc_parent_progress(current / total)
                parity = "rev" if values.get("reverse") else "fwd"
                source = self._short(values.get("source", ""), 24)
                self._stage.set_postfix_str(
                    f"{parity} seq={int(values.get('sequence', 0))}/"
                    f"{int(values.get('sequence_total', 0))} "
                    f"{source} loss={float(values.get('loss', 0.0)):.4f}".strip()
                )
            return

        if kind == "bc_sequence_done":
            # Deliberately no separate sequence bar to close.
            return

        if kind == "bc_done":
            self._set_bc_parent_progress(1.0)
            if self._stage is not None:
                self._advance_to(self._stage, float(self._stage.total))
                self._stage.set_postfix_str(f"loss={float(values.get('loss', 0.0)):.4f}")
            self._close_live("_detail")
            self._close_live("_stage")
            return

        if kind == "guard_start":
            self._guard_active = True
            # Continue the same epoch that BC started.  If instrumentation ever
            # enters Guard without BC, still establish a sane parent base.
            if self._fraction_parent_name != "_epoch":
                self._begin_fractional_parent_unit()
            self._set_second_phase_parent_progress(0.0)
            self._new_stage(total=3, desc="Guard", colour="yellow")
            self._stage.set_postfix_str(
                f"alphas={int(values.get('alphas', 0))} "
                f"V={int(values.get('validation_total', 0))} "
                f"A={int(values.get('anchor_total', 0))}"
            )
            return

        if kind == "guard_phase_start":
            phase = str(values.get("phase", "eval"))
            total = max(1, int(values.get("total", 1)))
            current = int(values.get("current", 0))
            if self._detail is None or self._detail_phase != phase:
                labels = {
                    "train": ("Train α", "yellow"),
                    "validation": ("Validation", "blue"),
                    "anchor": ("Anchors", "magenta"),
                }
                desc, colour = labels.get(phase, (phase.title(), "cyan"))
                self._new_detail(total=total, desc=desc, colour=colour, phase=phase)
            self._advance_to(self._detail, current)
            self._detail.set_postfix_str(f"states={int(values.get('states', 0))}")
            phase_base = {"train": 0.0, "validation": 1.0, "anchor": 2.0}.get(phase)
            if phase_base is not None and self._stage is not None:
                stage_target = phase_base + current / total
                self._advance_to(self._stage, stage_target)
                self._set_second_phase_parent_progress(stage_target / 3.0)
            return

        if kind == "guard_phase_step":
            phase = str(values.get("phase", "eval"))
            current = int(values.get("current", 0))
            total = max(1, int(values.get("total", 1)))
            if self._detail is not None and self._detail_phase == phase:
                self._advance_to(self._detail, current)
                self._detail.set_postfix_str(f"states={int(values.get('states', 0))}")
            phase_base = {"train": 0.0, "validation": 1.0, "anchor": 2.0}.get(phase)
            if phase_base is not None and self._stage is not None:
                stage_target = phase_base + current / total
                self._advance_to(self._stage, stage_target)
                self._set_second_phase_parent_progress(stage_target / 3.0)
            return

        if kind == "guard_done":
            if self._stage is not None:
                self._advance_to(self._stage, 3.0)
                if values.get("accepted"):
                    alpha = values.get("alpha")
                    self._stage.set_postfix_str(
                        "ACCEPT" if alpha is None else f"ACCEPT a={float(alpha):g}"
                    )
                else:
                    self._stage.set_postfix_str("ROLLBACK")
            self._finish_fractional_parent_unit()
            self._close_live("_detail")
            self._close_live("_stage")
            self._guard_active = False
            return

        if kind == "eval_start" and not self._guard_active:
            label = str(values.get("label", "Eval"))
            self._new_stage(total=int(values.get("total", 1)), desc=label, colour="cyan")
            self._advance_to(self._stage, int(values.get("current", 0)))
            self._stage.set_postfix_str(
                f"cache={int(values.get('cached', 0))} workers={int(values.get('workers', 1))}"
            )
            return

        if kind == "eval_step" and self._stage is not None and not self._guard_active:
            self._advance_to(self._stage, int(values.get("current", 0)))
            return

        if kind == "eval_done" and not self._guard_active:
            if self._stage is not None:
                self._advance_to(self._stage, int(values.get("total", self._stage.total)))
            self._close_live("_detail")
            self._close_live("_stage")
            return

    def _handle(self, event, raw: str) -> bool:
        if event.kind in {"round_done", "checkpoint_final", "final"}:
            self._close_live("_detail")
            self._close_live("_stage")

        handled = super()._handle(event, raw)

        # Plain trainer lines remain the source of truth for completed integer
        # work.  Re-sync the fractional hierarchy to those exact boundaries.
        if event.kind == "round_start":
            self._fraction_parent_name = None
            self._fraction_parent_base = 0.0
            self._sync_round_from_epoch()
        elif event.kind == "epoch_step":
            self._sync_round_from_epoch()
        elif event.kind == "epoch_skip":
            self._sync_round_from_epoch()
        elif event.kind == "round_done":
            self._fraction_parent_name = None
            self._fraction_parent_base = 0.0
        elif event.kind == "bootstrap_step" and self._bootstrap is not None:
            self._fraction_parent_name = None
            self._fraction_parent_base = 0.0

        return handled

    def __enter__(self):
        super().__enter__()
        if self.enabled:
            subscribe(self._on_progress)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            unsubscribe(self._on_progress)
        self._close_live("_detail")
        self._close_live("_stage")
        return super().__exit__(exc_type, exc, tb)
