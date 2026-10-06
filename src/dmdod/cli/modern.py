from __future__ import annotations

"""TTY-only modern frontend for long DMDOD training runs.

The trainer remains the source of truth and continues emitting plain text.  This
module only interprets known progress lines when stdout is an interactive TTY,
turning them into coloured tqdm progress bars and live postfix metrics.  Pipes,
redirects, CI, and ``DMDOD_PLAIN_CLI=1`` keep the original byte-for-byte style of
plain log output.
"""

import builtins
import os
import re
import sys
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

try:  # tqdm is already part of the rl extra, but keep the frontend optional.
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only without the rl extra
    tqdm = None


RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
CYAN = "\x1b[36m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
MAGENTA = "\x1b[35m"
BLUE = "\x1b[34m"


@dataclass(frozen=True, slots=True)
class CliEvent:
    kind: str
    values: dict[str, Any] = field(default_factory=dict)


def _float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    return None if match is None else float(match.group(1))


def _int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text)
    return None if match is None else int(match.group(1))


def parse_training_line(text: str) -> CliEvent | None:
    """Parse one existing trainer log line without changing trainer semantics."""

    match = re.match(
        r"teacher-turbo: anchors=(\d+) cache-hit=(\d+) generate=(\d+) workers=(\d+)", text
    )
    if match:
        return CliEvent(
            "teacher_start",
            {
                "total": int(match.group(1)),
                "cache_hits": int(match.group(2)),
                "generate": int(match.group(3)),
                "workers": int(match.group(4)),
            },
        )

    match = re.match(r"teacher\s+(\d+)\s+(.+?)\s+[-+]?\d", text)
    if match:
        return CliEvent("teacher_step", {"index": int(match.group(1)), "chart": match.group(2)})

    match = re.match(r"bootstrap\s+(\d+):", text)
    if match:
        epoch = int(match.group(1))
        if "PRUNE[" in text:
            reason = re.search(r"PRUNE\[([^]]+)\]", text)
            evaluated = re.search(r"eval=(\d+)/(\d+)", text)
            return CliEvent(
                "bootstrap_step",
                {
                    "epoch": epoch,
                    "loss": _float(r"loss=([0-9.eE+-]+)", text),
                    "status": "PRUNE" if reason is None else f"PRUNE:{reason.group(1)}",
                    "evaluated": None if evaluated is None else int(evaluated.group(1)),
                    "eval_total": None if evaluated is None else int(evaluated.group(2)),
                },
            )
        return CliEvent(
            "bootstrap_step",
            {
                "epoch": epoch,
                "loss": _float(r"loss=([0-9.eE+-]+)", text),
                "safe": "safe=True" in text,
                "completion": _float(r"completion=([0-9.]+)%", text),
                "mean_x": _float(r"meanX=([0-9.]+)%", text),
                "status": "KEEP" if text.endswith(" KEEP") else "eval",
            },
        )

    if text.startswith("bootstrap selected:"):
        return CliEvent("bootstrap_done")

    match = re.search(r"completed-round=(\d+)/(\d+)", text)
    if text.startswith("resume=") and match:
        return CliEvent(
            "resume",
            {"completed": int(match.group(1)), "total": int(match.group(2))},
        )

    match = re.match(r"round\s+(\d{3})\s+chart=(.+?)\s+train=", text)
    if match:
        return CliEvent(
            "round_start",
            {
                "round": int(match.group(1)),
                "chart": match.group(2),
                "base_x": _float(r"base=.*?X([0-9.]+)%", text),
                "mix_x": _float(r"mix\d+=.*?X([0-9.]+)%", text),
            },
        )

    match = re.match(r"round\s+(\d{3})\s+e(\d{2}):\s+(ACCEPT|ROLLBACK)", text)
    if match:
        return CliEvent(
            "epoch_step",
            {
                "round": int(match.group(1)),
                "epoch": int(match.group(2)),
                "status": match.group(3),
                "alpha": _float(r"a=([0-9.eE+-]+)", text),
                "loss": _float(r"loss=([0-9.eE+-]+)", text),
                "x": _float(r"\sX([0-9.]+)%", text),
            },
        )

    match = re.match(r"round\s+(\d{3}): exact odd/even proposal fixed point; skip remaining (\d+) epochs", text)
    if match:
        return CliEvent(
            "epoch_skip",
            {"round": int(match.group(1)), "remaining": int(match.group(2))},
        )

    match = re.match(r"round\s+(\d{3})\s+summary:\s+accepted=(\d+)", text)
    if match:
        return CliEvent(
            "round_done",
            {
                "round": int(match.group(1)),
                "accepted": int(match.group(2)),
                "x": _float(r"\sX([0-9.]+)%", text),
            },
        )

    match = re.match(r"checkpoint round\s+(\d{3}):", text)
    if match:
        return CliEvent("checkpoint", {"round": int(match.group(1))})

    if text.startswith("FINAL holdout aggregate:"):
        return CliEvent("final", {"text": text})
    if text.startswith("final validation aggregate:"):
        return CliEvent("validation_final", {"text": text})
    if text.startswith("checkpoint final:"):
        return CliEvent("checkpoint_final", {"text": text})

    return None


def _supports_modern_tty(stream) -> bool:
    if tqdm is None:
        return False
    if os.environ.get("DMDOD_PLAIN_CLI"):
        return False
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(callable(isatty) and isatty())


def _option_int(argv: list[str], name: str, default: int) -> int:
    for index, token in enumerate(argv):
        if token == name and index + 1 < len(argv):
            try:
                return int(argv[index + 1])
            except ValueError:
                return default
        prefix = name + "="
        if token.startswith(prefix):
            try:
                return int(token[len(prefix) :])
            except ValueError:
                return default
    return default


class ModernTrainerConsole(AbstractContextManager):
    """Translate existing trainer prints into a compact live terminal dashboard."""

    def __init__(
        self,
        *,
        rounds: int,
        round_epochs: int,
        bootstrap_epochs: int,
        stream=None,
    ) -> None:
        self.stream = sys.stdout if stream is None else stream
        self.rounds = int(rounds)
        self.round_epochs = int(round_epochs)
        self.bootstrap_epochs = int(bootstrap_epochs)
        self.enabled = _supports_modern_tty(self.stream)
        self.color = self.enabled and not os.environ.get("NO_COLOR")
        self._original_print = builtins.print
        self._teacher = None
        self._bootstrap = None
        self._round = None
        self._epoch = None
        self._current_round = 0

    @classmethod
    def from_argv(cls, argv: list[str]) -> "ModernTrainerConsole":
        return cls(
            rounds=_option_int(argv, "--rounds", 48),
            round_epochs=_option_int(argv, "--round-epochs", 12),
            bootstrap_epochs=_option_int(argv, "--bootstrap-epochs", 64),
        )

    def _paint(self, text: str, colour: str, *, bold: bool = False) -> str:
        if not self.color:
            return text
        return f"{BOLD if bold else ''}{colour}{text}{RESET}"

    def _write(self, text: str) -> None:
        if tqdm is not None:
            tqdm.write(text, file=self.stream)
        else:  # pragma: no cover
            self._original_print(text, file=self.stream)

    def _bar(self, *, total: int, desc: str, colour: str, position: int):
        return tqdm(
            total=max(1, int(total)),
            desc=desc,
            unit="step",
            dynamic_ncols=True,
            leave=True,
            colour=colour,
            position=position,
            file=self.stream,
            mininterval=0.10,
        )

    @staticmethod
    def _advance_to(bar, value: int) -> None:
        target = max(0, min(int(value), int(bar.total)))
        delta = target - int(bar.n)
        if delta > 0:
            bar.update(delta)
        elif delta < 0:
            bar.n = target
            bar.refresh()

    def _close_bar(self, name: str) -> None:
        bar = getattr(self, name)
        if bar is not None:
            bar.close()
            setattr(self, name, None)

    def _handle(self, event: CliEvent, raw: str) -> bool:
        values = event.values

        if event.kind == "teacher_start":
            self._close_bar("_teacher")
            self._teacher = self._bar(total=values["total"], desc="Teacher", colour="cyan", position=0)
            self._teacher.set_postfix_str(
                f"cache={values['cache_hits']} gen={values['generate']} workers={values['workers']}"
            )
            return True

        if event.kind == "teacher_step" and self._teacher is not None:
            self._advance_to(self._teacher, values["index"])
            self._teacher.set_postfix_str(values["chart"][-36:])
            if values["index"] >= self._teacher.total:
                self._close_bar("_teacher")
            return True

        if event.kind == "bootstrap_step":
            if values["epoch"] == 0:
                self._write(self._paint(raw, BLUE))
                return True
            if self._bootstrap is None:
                self._bootstrap = self._bar(
                    total=self.bootstrap_epochs,
                    desc="Bootstrap",
                    colour="blue",
                    position=0,
                )
            self._advance_to(self._bootstrap, values["epoch"])
            postfix = []
            if values.get("loss") is not None:
                postfix.append(f"loss={values['loss']:.4f}")
            if values.get("completion") is not None:
                postfix.append(f"C={values['completion']:.1f}%")
            if values.get("mean_x") is not None:
                postfix.append(f"X={values['mean_x']:.1f}%")
            postfix.append(str(values.get("status", "")))
            self._bootstrap.set_postfix_str(" ".join(item for item in postfix if item))
            return True

        if event.kind == "bootstrap_done":
            if self._bootstrap is not None:
                self._advance_to(self._bootstrap, self.bootstrap_epochs)
                self._close_bar("_bootstrap")
            self._write(self._paint(raw, GREEN, bold=True))
            return True

        if event.kind == "resume":
            if self._round is None:
                self._round = self._bar(total=values["total"], desc="Rounds", colour="green", position=0)
            self._advance_to(self._round, values["completed"])
            self._write(self._paint(raw, CYAN))
            return True

        if event.kind == "round_start":
            round_index = values["round"]
            self._current_round = round_index
            if self._round is None:
                self._round = self._bar(total=self.rounds, desc="Rounds", colour="green", position=0)
            self._close_bar("_epoch")
            self._epoch = self._bar(
                total=self.round_epochs,
                desc=f"R{round_index:02d}",
                colour="magenta",
                position=1,
            )
            parts = [values["chart"][-30:]]
            if values.get("base_x") is not None:
                parts.append(f"baseX={values['base_x']:.1f}%")
            if values.get("mix_x") is not None:
                parts.append(f"mixX={values['mix_x']:.1f}%")
            self._epoch.set_postfix_str(" ".join(parts))
            return True

        if event.kind == "epoch_step" and self._epoch is not None:
            self._advance_to(self._epoch, values["epoch"])
            status = values["status"]
            parts = [status]
            if values.get("loss") is not None:
                parts.append(f"loss={values['loss']:.4f}")
            if values.get("alpha") is not None:
                parts.append(f"a={values['alpha']:g}")
            if values.get("x") is not None:
                parts.append(f"X={values['x']:.1f}%")
            self._epoch.set_postfix_str(" ".join(parts))
            return True

        if event.kind == "epoch_skip" and self._epoch is not None:
            self._advance_to(self._epoch, self.round_epochs)
            self._epoch.set_postfix_str(f"fixed-point skip={values['remaining']}")
            return True

        if event.kind == "round_done":
            if self._epoch is not None:
                self._advance_to(self._epoch, self.round_epochs)
                self._close_bar("_epoch")
            if self._round is None:
                self._round = self._bar(total=self.rounds, desc="Rounds", colour="green", position=0)
            self._advance_to(self._round, values["round"])
            postfix = f"accepted={values['accepted']}"
            if values.get("x") is not None:
                postfix += f" X={values['x']:.1f}%"
            self._round.set_postfix_str(postfix)
            return True

        if event.kind == "checkpoint":
            self._write(self._paint(f"✓ {raw}", MAGENTA))
            return True

        if event.kind in {"validation_final", "final", "checkpoint_final"}:
            colour = GREEN if event.kind != "checkpoint_final" else MAGENTA
            self._write(self._paint(raw, colour, bold=True))
            return True

        return False

    def _modern_print(self, *args, **kwargs) -> None:
        file = kwargs.get("file", self.stream)
        if file not in (None, self.stream, sys.stdout):
            self._original_print(*args, **kwargs)
            return

        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        if end != "\n":
            self._original_print(*args, **kwargs)
            return
        text = sep.join(str(item) for item in args)
        event = parse_training_line(text)
        if event is not None and self._handle(event, text):
            return

        if text.startswith("==="):
            self._write(self._paint(text, CYAN, bold=True))
        elif "fallback" in text.lower() or "error" in text.lower():
            self._write(self._paint(text, RED))
        elif "warning" in text.lower() or "PRUNE[" in text:
            self._write(self._paint(text, YELLOW))
        elif text.startswith(("dataset:", "input=", "train-window=", "flat-eval=", "fast-eval=", "turbo=", "press-persistence")):
            self._write(self._paint(text, BLUE))
        else:
            self._write(text)

    def __enter__(self):
        if self.enabled:
            builtins.print = self._modern_print
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            builtins.print = self._original_print
        for name in ("_teacher", "_bootstrap", "_epoch", "_round"):
            self._close_bar(name)
        return False
