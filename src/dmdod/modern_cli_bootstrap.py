from __future__ import annotations

"""Bootstrap-aware live terminal dashboard for the v1.0 trainer."""

from typing import Any

from .modern_cli_live import LiveModernTrainerConsole


class BootstrapLiveModernTrainerConsole(LiveModernTrainerConsole):
    """Show turbo bootstrap evaluation and keep fast BC on one compact bar."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._bootstrap_eval_active = False

    def _on_progress(self, event) -> None:
        if not self.enabled:
            return
        values = event.values
        kind = event.kind

        # BC sequences are short enough that a second per-sequence tqdm bar is
        # mostly terminal churn. Keep the global chunk bar and surface the
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
                self._stage.set_postfix_str(
                    f"loss={float(values.get('loss', 0.0)):.4f}"
                )
            self._close_live("_detail")
            self._close_live("_stage")
            return

        if kind == "bootstrap_eval_start":
            self._bootstrap_eval_active = True
            total = int(values.get("total", 1))
            self._new_stage(total=max(1, total), desc="BS eval", colour="cyan")
            self._stage.set_postfix_str(
                f"wave=0/{int(values.get('waves', 0))} "
                f"start={int(values.get('start_micro', 0))} "
                f"batch={int(values.get('batch_size', 0))}"
            )
            return

        if kind == "bootstrap_eval_step":
            if self._stage is None:
                self._new_stage(
                    total=max(1, int(values.get("total", 1))),
                    desc="BS eval",
                    colour="cyan",
                )
            self._advance_to(self._stage, int(values.get("current", 0)))
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
            self._bootstrap_eval_active = False
            return

        # The wrapped evaluator may itself emit generic eval events for each
        # worker-sized wave. Hide those while the bootstrap-specific bar owns
        # the display, otherwise the bar would reset on every wave.
        if self._bootstrap_eval_active and kind in {"eval_start", "eval_step", "eval_done"}:
            return

        super()._on_progress(event)
