from __future__ import annotations

"""Modern live frontend for the accelerated v0.9 turbo trainer."""

import sys

from dmdod.modern_cli_live import LiveModernTrainerConsole
import train_real_chart_v090_progress as progress
import train_real_chart_v090_round_accel as round_accel
import train_real_chart_v090_turbo as turbo


def main() -> None:
    with LiveModernTrainerConsole.from_argv(sys.argv[1:]):
        # Install scheduling first so progress wraps the final accelerated hot
        # paths, then let the turbo trainer install its v0.9/v0.8 runtime hooks.
        round_accel.install_round_acceleration()
        progress.install_progress_instrumentation()
        turbo.main()


if __name__ == "__main__":
    main()
