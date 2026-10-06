from __future__ import annotations

"""Modern tqdm frontend for the v1.7.2 connectome trust-DAgger trainer."""

from dmdod.cli.dagger import run_with_dagger_modern_console
import train_real_chart_v172_n_key_connectome_reject_early_stop_dagger as v172


def main() -> None:
    run_with_dagger_modern_console(v172.main)


if __name__ == "__main__":
    main()
