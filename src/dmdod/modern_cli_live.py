from __future__ import annotations

"""Fine-grained live dashboard layered on top of ``modern_cli``."""

from typing import Any

from . import modern_cli as base
from .modern_cli import ModernTrainerConsole
from .training_progress import TrainingProgressEvent, subscribe, unsubscribe


class LiveModernTrainerConsole(ModernTrainerConsole):
    """Add current-stage and current-detail bars fed by trainer callbacks."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stage = None
        self._detail = None
        self._detail_phase: str | None = None
        self._guard_active = False

    def _live_bar(self, *, total: int, desc: str, colour: str, position: int):
        return base.tqdm(
            total=max(1, int(total)),
            desc=desc,
            unit="step",
            dynamic_ncols=True,
            leave=False,
            colour=colour,
            position=position,
            file=self.stream,
            mininterval=0.10,
        )

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
                self._advance_to(self._stage, int(values.get("global_chunk", 0)))
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
            if self._stage is not None:
                self._advance_to(self._stage, int(self._stage.total))
                self._stage.set_postfix_str(f"loss={float(values.get('loss', 0.0)):.4f}")
            self._close_live("_detail")
            self._close_live("_stage")
            return

        if kind == "guard_start":
            self._guard_active = True
            self._new_stage(total=3, desc="Guard", colour="yellow")
            self._stage.set_postfix_str(
                f"alphas={int(values.get('alphas', 0))} "
                f"V={int(values.get('validation_total', 0))} "
                f"A={int(values.get('anchor_total', 0))}"
            )
            return

        if kind == "guard_phase_start":
            phase = str(values.get("phase", "eval"))
            total = int(values.get("total", 1))
            if self._detail is None or self._detail_phase != phase:
                labels = {
                    "train": ("Train α", "yellow"),
                    "validation": ("Validation", "blue"),
                    "anchor": ("Anchors", "magenta"),
                }
                desc, colour = labels.get(phase, (phase.title(), "cyan"))
                self._new_detail(total=total, desc=desc, colour=colour, phase=phase)
            self._advance_to(self._detail, int(values.get("current", 0)))
            self._detail.set_postfix_str(f"states={int(values.get('states', 0))}")
            return

        if kind == "guard_phase_step":
            phase = str(values.get("phase", "eval"))
            if self._detail is not None and self._detail_phase == phase:
                self._advance_to(self._detail, int(values.get("current", 0)))
                self._detail.set_postfix_str(f"states={int(values.get('states', 0))}")
            phase_index = {"train": 1, "validation": 2, "anchor": 3}.get(phase)
            if phase_index is not None and self._stage is not None:
                if int(values.get("current", 0)) >= int(values.get("total", 1)):
                    self._advance_to(self._stage, phase_index)
            return

        if kind == "guard_done":
            if self._stage is not None:
                self._advance_to(self._stage, 3)
                if values.get("accepted"):
                    alpha = values.get("alpha")
                    self._stage.set_postfix_str(
                        "ACCEPT" if alpha is None else f"ACCEPT a={float(alpha):g}"
                    )
                else:
                    self._stage.set_postfix_str("ROLLBACK")
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
        return super()._handle(event, raw)

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
