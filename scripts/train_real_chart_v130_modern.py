from __future__ import annotations

"""Modern live frontend for the v1.3 aggregate-XAcc trainer."""

import sys

from dmdod.cli.bootstrap import BootstrapLiveModernTrainerConsole
import train_real_chart_v090_chunk_gru as chunk_gru
import train_real_chart_v090_progress as progress
import train_real_chart_v090_round_accel as round_accel
import train_real_chart_v100_bootstrap_progress as bootstrap_progress
import train_real_chart_v110_bootstrap_cuda as bootstrap_cuda
import train_real_chart_v110_nonempty_segments as nonempty_segments
import train_real_chart_v110_rejection_telemetry as rejection_telemetry
import train_real_chart_v120_bc_guard_overlap as bc_guard_overlap
import train_real_chart_v120_train_preserve as v120
import train_real_chart_v130_aggregate_xacc as v130


def main() -> None:
    with BootstrapLiveModernTrainerConsole.from_argv(sys.argv[1:]):
        nonempty_segments.install_nonempty_segment_relocation()
        round_accel.install_round_acceleration()
        bc_guard_overlap.install_bc_guard_overlap()
        v130.install_aggregate_xacc_line_search()
        # v1.2 selection wraps the v1.3 staged guard result.
        v120.install_train_preserve_gate()
        rejection_telemetry.install_rejection_telemetry()
        progress.install_progress_instrumentation()
        chunk_gru.install_chunk_gru_acceleration()
        bootstrap_cuda.install_bootstrap_cuda_acceleration()
        bootstrap_progress.install_bootstrap_progress()
        try:
            v130.main()
        finally:
            bc_guard_overlap.print_overlap_stats()
            rejection_telemetry.print_rejection_telemetry()


if __name__ == "__main__":
    main()
