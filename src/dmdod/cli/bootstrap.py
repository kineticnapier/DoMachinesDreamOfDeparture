from __future__ import annotations

"""Bootstrap-aware live terminal dashboard for the v1.0 trainer."""

import re
import sys
from collections.abc import Callable
from typing import Any, TypeVar

from .live import (
    LiveModernTrainerConsole,
    _INTEGER_COUNT_BAR_FORMAT,
    _IntegerCountTqdm,
)


PARENT_ETA_VERSION = "cumulative-fractional-v1"
_T = TypeVar("_T")


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
            self._eta_origin_n: float | None = None
            self._eta_origin_elapsed = 0.0
            super().__init__(*args, **kwargs)
            self.reset_eta_origin()

        def reset_eta_origin(self) -> None:
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
    """Show bootstrap and evaluation progress on the live dashboard."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._bootstrap_eval_active = False
        self._n_key_role: str | None = None
        self._n_key_bootstrap_total: int | None = None

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
        if event.kind == "resume":
            self._reset_parent_eta("_round")
        return handled

    def _handle_n_key_bootstrap_line(self, text: str) -> bool:
        """Translate the v1.6.x N-key bootstrap's existing logs into tqdm bars.

        That trainer predates the fine-grained progress event bus and emits
        ``001/20`` style progress lines instead. Keep the trainer output as the
        source of truth while making the shared modern frontend useful for the
        v1.6.8/v1.7.0 wrappers as well.

        The bootstrap bar is created from the pre-training ``parameters`` line,
        not from the first completed epoch. This makes tqdm's elapsed/rate cover
        epoch 1 as well, so a one-epoch run reports real wall time instead of an
        effectively instantaneous post-hoc 0 -> 1 update.
        """

        config = re.match(
            r"anchors=\d+\s+validation=\d+\s+epochs=(\d+)\s+\|",
            text,
        )
        if config:
            self._n_key_bootstrap_total = int(config.group(1))
            return False

        teacher = re.match(r"teacher\s+(\d+)/(\d+)\s+(.+?)\s+H=", text)
        if teacher:
            current, total = int(teacher.group(1)), int(teacher.group(2))
            if self._teacher is None:
                self._teacher = self._bar(
                    total=total, desc="Teacher", colour="cyan", position=0
                )
            self._advance_to(self._teacher, current)
            self._teacher.set_postfix_str(teacher.group(3)[-36:])
            if current >= total:
                self._close_bar("_teacher")
            return True

        if text.startswith("parameters trainable="):
            total = self._n_key_bootstrap_total
            if total is not None and total > 0 and self._bootstrap is None:
                self._bootstrap = self._bar(
                    total=total, desc="Bootstrap", colour="blue", position=0
                )
            return False

        bootstrap = re.match(
            r"bootstrap\s+(\d+)/(\d+)\s+loss=([0-9.eE+-]+)", text
        )
        if bootstrap:
            current = int(bootstrap.group(1))
            total = int(bootstrap.group(2))
            loss = float(bootstrap.group(3))
            if self._bootstrap is None:
                self._bootstrap = self._bar(
                    total=total, desc="Bootstrap", colour="blue", position=0
                )
            self._advance_to(self._bootstrap, current)
            self._bootstrap.set_postfix_str(f"loss={loss:.6f}")
            if current >= total:
                self._close_bar("_bootstrap")
            return True

        role = re.match(
            r"(student-train|validation)\s+(\d+)/(\d+)\s+(.+?):\s+H=", text
        )
        if role:
            name = role.group(1)
            current, total = int(role.group(2)), int(role.group(3))
            if self._stage is None or self._n_key_role != name:
                self._close_live("_stage")
                self._n_key_role = name
                desc = "Train eval" if name == "student-train" else "Validation"
                colour = "yellow" if name == "student-train" else "cyan"
                self._new_stage(total=total, desc=desc, colour=colour)
            self._advance_to(self._stage, current)
            self._stage.set_postfix_str(role.group(4)[-34:])
            if current >= total:
                self._close_live("_stage")
                self._n_key_role = None
            return True

        aggregate = re.match(r"(student-train|validation) aggregate:", text)
        if aggregate:
            self._close_live("_stage")
            self._n_key_role = None
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
            if self._handle_n_key_bootstrap_line(text):
                return
        super()._modern_print(*args, **kwargs)

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

        if self._bootstrap_eval_active and kind in {"eval_start", "eval_step", "eval_done"}:
            return

        super()._on_progress(event)


def _pop_ui_option(argv: list[str]) -> str:
    """Consume the common --ui option before trainer-specific argparse sees it."""

    mode = "auto"
    for index, token in enumerate(list(argv)):
        if token == "--ui":
            if index + 1 >= len(argv):
                raise SystemExit(f"--ui requires one of: auto, modern, plain")
            mode = argv[index + 1]
            del argv[index : index + 2]
            break
        if token.startswith("--ui="):
            mode = token.split("=", 1)[1]
            del argv[index]
            break
    if mode not in {"auto", "modern", "plain"}:
        raise SystemExit("--ui must be one of: auto, modern, plain")
    return mode


def run_with_modern_console(main: Callable[[], _T]) -> _T:
    """Run any compatible trainer through the shared modern/plain UI switch."""

    argv = list(sys.argv)
    mode = _pop_ui_option(argv)
    sys.argv = argv
    if mode == "plain":
        return main()
    with BootstrapLiveModernTrainerConsole.from_argv(argv[1:]):
        return main()
