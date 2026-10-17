from __future__ import annotations

"""Finalize a completed v0.1 run without retraining or changing raw results."""

import argparse
import copy
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_fingering_network_v01 as experiment


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def correct_viewer_associations(report: dict) -> dict:
    """Keep one nearest observed KeyDown per selected physical axis."""

    output = copy.deepcopy(report)
    for case in output["physical_evaluation"]["cases"]:
        for row in case["network_v0_1"]["decisions"]:
            nearby = row.get("keydowns", [])
            selected = []
            for axis in row["axes"]:
                matches = [value for value in nearby if value["physical_axis"] == axis]
                if matches:
                    selected.append(min(
                        matches, key=lambda value: abs(value["time_ms"] - row["due_ms"])
                    ))
            row["keydowns"] = selected
            row["too_early_keydowns"] = 0
            row["judgement"] = row["outcome"]
    return output


def analysis(raw: dict, *, raw_path: Path, checkpoint: Path, dataset: Path) -> dict:
    aggregate = raw["physical_evaluation"]["aggregate"]
    center = aggregate["center_first"]
    oracle = aggregate["teacher_v0_2"]
    learned = aggregate["network_v0_1"]
    criteria = {
        "hit_rate_not_below_center_first": learned["hit_rate"] >= center["hit_rate"],
        "misses_not_above_center_first": learned["misses"] <= center["misses"],
        "too_early_not_above_center_first": learned["too_early"] <= center["too_early"],
        "overload_not_above_center_first": learned["overloaded_charts"] <= center["overloaded_charts"],
        "timing_mae_below_center_first": learned["timing_mae_ms"] < center["timing_mae_ms"],
        "different_teacher_choice_can_hit": learned["different_from_teacher_but_hit"] > 0,
    }
    return {
        "format": "dmdod-fingering-network-v0.1-final-analysis",
        "status": "PASS" if all(criteria.values()) else "FAIL",
        "success_criteria": criteria,
        "comparison": aggregate,
        "teacher_timing_mae_gap_ms": learned["timing_mae_ms"] - oracle["timing_mae_ms"],
        "ranking_metrics": raw["training"]["split_metrics"],
        "artifacts": {
            "raw_report": str(raw_path), "raw_report_sha256": file_sha256(raw_path),
            "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
            "dataset": str(dataset), "dataset_sha256": file_sha256(dataset),
        },
        "interpretation": (
            "v0.1 improves physical timing over CenterFirst and succeeds with different "
            "teacher choices, but does not match Teacher v0.2 timing MAE."
        ),
        "retraining_performed_by_finalizer": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--viewer", type=Path, required=True)
    args = parser.parse_args()
    if args.analysis.exists() or args.viewer.exists():
        parser.error("refusing to overwrite final artifacts")
    raw = json.loads(args.raw_report.read_text(encoding="utf-8"))
    corrected = correct_viewer_associations(raw)
    args.analysis.parent.mkdir(parents=True, exist_ok=True)
    args.analysis.write_text(
        json.dumps(analysis(raw, raw_path=args.raw_report,
                            checkpoint=args.checkpoint, dataset=args.dataset),
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.viewer.write_text(experiment.render_viewer(corrected), encoding="utf-8")


if __name__ == "__main__":
    main()
