from __future__ import annotations

"""Read-only near/far error audit of the frozen Stage T Timing checkpoints."""

import argparse

from dmdod.training.prestige1 import PrestigeConfig
from dmdod.training.prestige1_autoplay_diagnostics import run_distance_audit, BINS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/prestige1.toml")
    options = parser.parse_args()
    result = run_distance_audit(PrestigeConfig.load(options.config))
    print("=== Frozen Stage T distance diagnostic (Train20 only) ===")
    for split in ("train_fit", "train_heldout"):
        print(split)
        for model in ("constant", "mlp", "gru"):
            bins = result["scores"][split][model]["aggregate"]
            row = " ".join(
                f"{name}={bins[name]['mae_ms']:.2f}ms"
                f"(n={bins[name]['frames']})" if bins[name]["frames"] else f"{name}=N/A"
                for name in BINS
            )
            print(f"  {model}: {row}", flush=True)
    print("Read-only audit PASS. Original Stage A FAIL remains unchanged.", flush=True)


if __name__ == "__main__":
    main()
