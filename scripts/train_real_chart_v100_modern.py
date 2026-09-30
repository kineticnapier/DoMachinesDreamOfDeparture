from __future__ import annotations

"""Modern live frontend for the v1.0 start-micro trainer."""

import sys

from dmdod.modern_cli_live import LiveModernTrainerConsole
import train_real_chart_v090_chunk_gru as chunk_gru
import train_real_chart_v090_progress as progress
import train_real_chart_v090_round_accel as round_accel
import train_real_chart_v100_start_micro as v100


def main() -> None:
    with LiveModernTrainerConsole.from_argv(sys.argv[1:]):
        round_accel.install_round_acceleration()
        progress.install_progress_instrumentation()
        chunk_gru.install_chunk_gru_acceleration()
        v100.main()


if __name__ == "__main__":
    main()
