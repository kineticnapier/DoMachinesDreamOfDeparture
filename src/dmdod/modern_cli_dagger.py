from __future__ import annotations

"""tqdm frontend for the v1.6+ N-key trust-DAgger text logs.

The trust-DAgger trainers intentionally keep plain ``print`` logs as their
source of truth.  This module only translates those existing lines into the
shared modern tqdm dashboard, so training/evaluation semantics and checkpoint
contents remain untouched.
"""

import re
import sys
from collections.abc import Callable
from typing import Any, TypeVar

from .modern_cli_bootstrap import BootstrapLiveModernTrainerConsole, _pop_ui_option


_T = TypeVar("_T")


class DaggerLiveModernTrainerConsole(BootstrapLiveModernTrainerConsole):
    """Render v1.6+ trust-DAgger progress with stage and epoch ETAs."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._dagger_anchor_total: int | None = None
        self._dagger_epochs_total: int | None = None
        self._dagger_alphas: list[float] = []
        self._dagger_epoch_bar = None
        self._dagger_stage_key: str | None = None
        self._dagger_current_epoch = 0

    def _close_dagger_stage(self) -> None:
        self._close_live("_stage")
        self._dagger_stage_key = None

    def _close_dagger_epoch_bar(self) -> None:
        if self._dagger_epoch_bar is not None:
            self._dagger_epoch_bar.close()
            self._dagger_epoch_bar = None

    def _ensure_dagger_epoch_bar(self) -> None:
        if self._dagger_epoch_bar is not None:
            return
        total = self._dagger_epochs_total
        if total is None or total <= 0:
            return
        self._dagger_epoch_bar = self._bar(
            total=total,
            desc="DAgger",
            colour="blue",
            position=0,
        )

    def _ensure_stage(
        self,
        *,
        key: str,
        total: int,
        desc: str,
        colour: str,
    ) -> None:
        if self._stage is not None and self._dagger_stage_key == key:
            return
        self._close_dagger_stage()
        self._dagger_stage_key = key
        self._new_stage(total=total, desc=desc, colour=colour)

    def _start_trust_alpha(self, alpha: float) -> None:
        total = self._dagger_anchor_total
        if total is None or total <= 0:
            return
        self._ensure_stage(
            key=f"trust:{self._dagger_current_epoch}:{alpha:g}",
            total=total,
            desc=f"Trust a={alpha:g}",
            colour="yellow",
        )

    def _next_alpha(self, alpha: float) -> float | None:
        for index, candidate in enumerate(self._dagger_alphas):
            if abs(candidate - alpha) <= max(1e-15, abs(alpha) * 1e-12):
                next_index = index + 1
                return (
                    self._dagger_alphas[next_index]
                    if next_index < len(self._dagger_alphas)
                    else None
                )
        return None

    def _handle_n_key_dagger_line(self, text: str) -> bool:
        config = re.match(
            r"anchors=(\d+)\s+validation=(\d+)\s+dagger-epochs=(\d+)\s+",
            text,
        )
        if config:
            self._dagger_anchor_total = int(config.group(1))
            self._dagger_epochs_total = int(config.group(3))
            return False

        alpha_config = re.match(r"trust-alphas=(.+)", text)
        if alpha_config:
            try:
                self._dagger_alphas = [
                    float(part) for part in alpha_config.group(1).split(",") if part
                ]
            except ValueError:
                self._dagger_alphas = []
            return False

        if text.startswith("=== pre-DAgger continuous Train"):
            total = self._dagger_anchor_total
            if total is not None:
                self._ensure_stage(
                    key="pre-train",
                    total=total,
                    desc="Pre-train",
                    colour="cyan",
                )
            return False

        pre_train = re.match(
            r"pre-train\s+(\d+)/(\d+)\s+(.+?):\s+H=", text
        )
        if pre_train:
            current, total = int(pre_train.group(1)), int(pre_train.group(2))
            self._ensure_stage(
                key="pre-train",
                total=total,
                desc="Pre-train",
                colour="cyan",
            )
            self._advance_to(self._stage, current)
            self._stage.set_postfix_str(pre_train.group(3)[-34:])
            if current >= total:
                self._close_dagger_stage()
            return True

        collect = re.match(
            r"collect-r(\d+)\s+(\d+)/(\d+)\s+(.+?):\s+frames=", text
        )
        if collect:
            round_index = int(collect.group(1))
            current, total = int(collect.group(2)), int(collect.group(3))
            self._ensure_stage(
                key=f"collect:{round_index}",
                total=total,
                desc=f"Collect r{round_index}",
                colour="magenta",
            )
            self._advance_to(self._stage, current)
            self._stage.set_postfix_str(collect.group(4)[-34:])
            if current >= total:
                self._close_dagger_stage()
            return True

        if re.match(r"(pre-train|collect-r\d+) aggregate:", text):
            self._close_dagger_stage()
            self._write(text)
            return True

        if text.startswith("aggregate-data generation="):
            self._ensure_dagger_epoch_bar()
            return False

        proposal = re.match(
            r"dagger-proposal\s+(\d+)/(\d+)\s+loss=([0-9.eE+-]+)", text
        )
        if proposal:
            self._dagger_current_epoch = int(proposal.group(1))
            if self._dagger_epochs_total is None:
                self._dagger_epochs_total = int(proposal.group(2))
            self._ensure_dagger_epoch_bar()
            if self._dagger_epoch_bar is not None:
                self._dagger_epoch_bar.set_postfix_str(
                    f"epoch={self._dagger_current_epoch} loss={float(proposal.group(3)):.6f}"
                )
            return True

        trust_header = re.match(
            r"=== Train trust line search epoch\s+(\d+)/(\d+)\s+===", text
        )
        if trust_header:
            self._dagger_current_epoch = int(trust_header.group(1))
            if self._dagger_alphas:
                self._start_trust_alpha(self._dagger_alphas[0])
            return True

        trust_eval = re.match(
            r"epoch-(\d+)-a([0-9.eE+-]+)\s+(\d+)/(\d+)\s+(.+?):\s+H=",
            text,
        )
        if trust_eval:
            epoch = int(trust_eval.group(1))
            alpha = float(trust_eval.group(2))
            current, total = int(trust_eval.group(3)), int(trust_eval.group(4))
            self._dagger_current_epoch = epoch
            self._ensure_stage(
                key=f"trust:{epoch}:{alpha:g}",
                total=total,
                desc=f"Trust a={alpha:g}",
                colour="yellow",
            )
            self._advance_to(self._stage, current)
            self._stage.set_postfix_str(trust_eval.group(5)[-34:])
            return True

        trust_result = re.match(r"trust alpha=([0-9.eE+-]+):\s+(.+)", text)
        if trust_result:
            alpha = float(trust_result.group(1))
            self._close_dagger_stage()
            self._write(text)
            next_alpha = self._next_alpha(alpha)
            if next_alpha is not None:
                self._start_trust_alpha(next_alpha)
            return True

        continuation = re.match(
            r"epoch-continuation:\s+(ACCEPT|KEEP)\s+(.+)", text
        )
        if continuation:
            self._close_dagger_stage()
            self._ensure_dagger_epoch_bar()
            if self._dagger_epoch_bar is not None:
                self._advance_to(self._dagger_epoch_bar, self._dagger_current_epoch)
                self._dagger_epoch_bar.set_postfix_str(
                    f"{continuation.group(1)} epoch={self._dagger_current_epoch}"
                )
            self._write(text)
            return True

        if text.startswith("epoch-loop: EARLY STOP"):
            self._close_dagger_stage()
            if self._dagger_epoch_bar is not None:
                self._dagger_epoch_bar.set_postfix_str(
                    f"EARLY STOP epoch={self._dagger_current_epoch}"
                )
            self._write(text)
            self._close_dagger_epoch_bar()
            return True

        if text.startswith("=== selected Train-safe trust-region checkpoint ==="):
            self._close_dagger_stage()
            self._close_dagger_epoch_bar()
            return False

        validation = re.match(
            r"validation\s+(\d+)/(\d+)\s+(.+?):\s+H=", text
        )
        if validation:
            current, total = int(validation.group(1)), int(validation.group(2))
            self._ensure_stage(
                key="dagger-validation",
                total=total,
                desc="Validation",
                colour="cyan",
            )
            self._advance_to(self._stage, current)
            self._stage.set_postfix_str(validation.group(3)[-34:])
            if current >= total:
                self._close_dagger_stage()
            return True

        if text.startswith("validation aggregate:"):
            self._close_dagger_stage()
            self._write(text)
            return True

        return False

    def _modern_print(self, *args, **kwargs) -> None:
        file = kwargs.get("file", self.stream)
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        if (
            self.enabled
            and file in (None, self.stream, sys.stdout)
            and end == "\n"
        ):
            text = sep.join(str(item) for item in args)
            if self._handle_n_key_dagger_line(text):
                return
        super()._modern_print(*args, **kwargs)

    def __exit__(self, exc_type, exc, tb):
        self._close_dagger_stage()
        self._close_dagger_epoch_bar()
        return super().__exit__(exc_type, exc, tb)


def run_with_dagger_modern_console(main: Callable[[], _T]) -> _T:
    """Run a trust-DAgger trainer with ``--ui auto|modern|plain`` support."""

    argv = list(sys.argv)
    mode = _pop_ui_option(argv)
    sys.argv = argv
    if mode == "plain":
        return main()
    with DaggerLiveModernTrainerConsole.from_argv(argv[1:]):
        return main()
