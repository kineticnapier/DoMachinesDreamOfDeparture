from __future__ import annotations

"""Read-only coupled-contact KeyDown -> current ADOFAI scorer probe."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_contact_judgement_audit import REPORT,run_audit


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/training/prestige1.toml')
    args=parser.parse_args()
    report=run_audit(PrestigeConfig.load(args.config))
    print(f'Fixed contact/judgement fixtures: {len(report["cases"])}; report: {REPORT}')
    print('Existing scorer only; full ADOFAI Multipress/OverPress not implemented.')


if __name__=='__main__':
    main()
