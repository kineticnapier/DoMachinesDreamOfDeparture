from __future__ import annotations

"""Modern tqdm frontend for the v1.8 connectome micro-step DAgger trainer."""

from dmdod.modern_cli_dagger import run_with_dagger_modern_console
import train_real_chart_v180_n_key_connectome_microstep_dagger as v180


def main() -> None:
    run_with_dagger_modern_console(v180.main)


if __name__ == "__main__":
    main()
