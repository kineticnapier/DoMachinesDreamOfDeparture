from __future__ import annotations

"""Modern live frontend for the accelerated v0.9 turbo trainer."""

import sys

from dmdod.cli.live import LiveModernTrainerConsole
import train_real_chart_v090_chunk_gru as chunk_gru
import train_real_chart_v090_progress as progress
import train_real_chart_v090_round_accel as round_accel
import train_real_chart_v090_turbo as turbo


def main() -> None:
    with LiveModernTrainerConsole.from_argv(sys.argv[1:]):
        # Install scheduling first, then observational progress, then replace
        # only the BC chunk's per-frame Python loop with the native GRU kernel.
        # v0.7 captures the final _train_one_epoch implementation when turbo
        # installs the mature training stack.
        round_accel.install_round_acceleration()
        progress.install_progress_instrumentation()
        chunk_gru.install_chunk_gru_acceleration()
        turbo.main()


if __name__ == "__main__":
    main()
