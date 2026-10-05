from __future__ import annotations

"""Modern tqdm frontend for the v1.8.1 connectome SGD micro-step DAgger trainer."""

from dmdod.modern_cli_dagger import run_with_dagger_modern_console
import train_real_chart_v181_n_key_connectome_sgd_microstep_dagger as v181


def main() -> None:
    run_with_dagger_modern_console(v181.main)


if __name__ == "__main__":
    main()
