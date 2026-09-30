from __future__ import annotations

"""Modern TTY frontend for the accelerated v0.9 turbo trainer."""

import sys

from dmdod.modern_cli import ModernTrainerConsole
import train_real_chart_v090_round_accel as round_accel
import train_real_chart_v090_turbo as turbo


def main() -> None:
    round_accel.install_round_acceleration()
    with ModernTrainerConsole.from_argv(sys.argv[1:]):
        turbo.main()


if __name__ == "__main__":
    main()
