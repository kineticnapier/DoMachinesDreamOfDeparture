from __future__ import annotations

"""Offline v3.0 teacher-fingering audit; Train20 only, no physics rerun."""

import argparse
from pathlib import Path

from dmdod.training.prestige1_fingering_audit import audit_files, write_once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=Path("diagnostics/prestige1"),
        help="Location of immutable Stage C v3.0 result JSONs",
    )
    args = parser.parse_args()
    root = args.directory
    report = root / "stage_c_v30_finger_phase_annealing.json"
    plans = root / "stage_c_v30_safe_teacher_plans.json"
    output = root / "stage_c_v30_fingering_audit.json"
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite existing audit: {output}")
    result = audit_files(report, plans)
    write_once(output, result)
    hits = result["fingering"]
    print(
        "Stage C v3.0 offline fingering: "
        f"Hit={hits['hit_count']} "
        f"center={hits['central_pair_hits']} "
        f"left={hits['hand_hit_counts']['left']} "
        f"right={hits['hand_hit_counts']['right']} "
        f"MAE={hits['mae_ms_from_events']:.6f}ms "
        f"adjacent-under20ms={hits['adjacent_hit_gaps']['under_20ms']} "
        f"same-target-time-pairs={hits['same_target_time_adjacent_pairs']}",
        flush=True,
    )
    print(f"Report: {output}", flush=True)


if __name__ == "__main__":
    main()
