from __future__ import annotations

"""Bootstrap-aware live terminal dashboard for the v1.0 trainer."""

from typing import Any

from .modern_cli_live import LiveModernTrainerConsole


class BootstrapLiveModernTrainerConsole(LiveModernTrainerConsole):
    """Show turbo bootstrap candidate evaluation on the live dashboard."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._bootstrap_eval_active = False

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
