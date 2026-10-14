from __future__ import annotations
"""Train20-only audit of temporal next-event vs physical next-unresolved semantics."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_alignment_audit import run_train_alignment_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/prestige1.toml")
    args = parser.parse_args()
    result = run_train_alignment_audit(PrestigeConfig.load(args.config))
    agg = result["aggregate_counts"]
    print("Prestige Train-only alignment audit complete: "
          f"mismatches={agg['mismatch_frames']}/{agg['frames']} "
          f"already-resolved={agg['legacy_points_to_resolved']} "
          f"near-disagreements={agg['near_flag_disagreement_frames']} "
          f"originals-unchanged={result['protected_inputs_unchanged']}", flush=True)


if __name__ == "__main__":
    main()
