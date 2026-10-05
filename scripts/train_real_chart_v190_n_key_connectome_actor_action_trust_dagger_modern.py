from __future__ import annotations

"""Modern tqdm frontend for the v1.9 connectome actor action-trust DAgger trainer."""

from dmdod.modern_cli_dagger import run_with_dagger_modern_console
import train_real_chart_v190_n_key_connectome_actor_action_trust_dagger as v190


def main() -> None:
    run_with_dagger_modern_console(v190.main)


if __name__ == "__main__":
    main()
