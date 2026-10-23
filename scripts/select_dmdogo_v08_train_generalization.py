from __future__ import annotations

"""Select distinct local Train charts for a FROZEN v0.8 generalization audit.

Selection uses compile-time Oracle timing *only to choose evaluation cases*.
It never executes Motor, fits a model, tunes parameters, selects from
Validation/FINAL, or overwrites any output. Selection is exploratory and
does not constitute a representative population sample.
"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import subprocess

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher

BOOL_SHA = "04981e92f74288b8aac7286e83420b933036d2caead9c157bc33b706779ad1b1"
FROZEN_HEAD = "45f30bcaf968281400e2c2913a96dd95515222d2"
FORBIDDEN = {"validation", "validation20", "final", "holdout"}
DEFAULT_CASES = 3
FROZEN_PLANNER_BLOB = "4ce02c2e4450dc7a5181ab74cbaa0c1accea94f5"
FROZEN_RUNNER_BLOB = "5fddfc84419722de7bbd0556419511069fd3200d"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def describe_events(events: tuple, *, name: str, sha256: str,
                    path: str) -> dict:
    """Statistical labels never use fingering or Motor/Hit outcomes."""
    times = [float(e.episode_time_s) for e in events]
    gaps = [b - a for a, b in zip(times, times[1:])
            if 1e-6 < b - a < .450]
    q10 = _quantile(gaps, .10)
    q90 = _quantile(gaps, .90)
    chord_events = sum(e.chord_size > 1 for e in events)
    return {
        "chart": name, "path": path, "sha256": sha256,
        "event_count": len(events),
        "duration_s": round(times[-1] - times[0], 6) if times else 0.,
        "chord_events": chord_events,
        "chord_fraction": round(chord_events / len(events), 8) if events else 0.,
        "min_inter_onset_ms": round(min(gaps) * 1000., 6) if gaps else None,
        "p10_gap_ms": round(q10 * 1000., 6) if gaps else None,
        "p90_to_p10_gap_ratio": round(q90 / max(q10, 1e-9), 6) if gaps else None,
        "median_gap_ms": round(median(gaps) * 1000., 6) if gaps else None,
    }


def select_cases(records: list[dict], count: int = DEFAULT_CASES) -> list[dict]:
    """Choose evidence-diverse cases without touching v0.8's reward/weights."""
    if count < 1:
        raise ValueError("count must be positive")
    pool = sorted(
        (r for r in records if r["event_count"] >= 12
         and r["sha256"] != BOOL_SHA),
        key=lambda r: (r["sha256"], r["path"]),
    )
    unique = {}
    for record in pool:
        unique.setdefault(record["sha256"], record)
    pool = list(unique.values())
    cases = []
    used = set()

    def add(label: str, predicate, key) -> None:
        if len(cases) >= count:
            return
        eligible = [r for r in pool
                    if r["sha256"] not in used and predicate(r)]
        if eligible:
            choice = min(eligible, key=key)
            cases.append({"selection_reason": label, **choice})
            used.add(choice["sha256"])

    # Explicit diversity axes; not a claim that 3 cases generalize to all U charts.
    add("simultaneous_contacts", lambda r: r["chord_events"] > 0,
        lambda r: (-r["chord_fraction"], -r["chord_events"], r["sha256"]))
    add("small_inter_onset_gaps",
        lambda r: r["p10_gap_ms"] is not None,
        lambda r: (r["p10_gap_ms"], r["sha256"]))
    add("uneven_timing",
        lambda r: r["p90_to_p10_gap_ratio"] is not None,
        lambda r: (-r["p90_to_p10_gap_ratio"], r["sha256"]))
    while len(cases) < count:
        before = len(cases)
        add("remaining_high_event_count", lambda r: True,
            lambda r: (-r["event_count"], r["sha256"]))
        if len(cases) == before:
            break
    return cases


def make_manifest(records: list[dict], skipped: list[dict],
                  *, train_root: str, baseline_excluded: int = 0) -> dict:
    return {
        "format": "dmdogo-v08-frozen-train-generalization-manifest-v1",
        "frozen_implementation_commit": FROZEN_HEAD,
        "frozen_file_blobs": {
            "planner": FROZEN_PLANNER_BLOB, "runner": FROZEN_RUNNER_BLOB,
        },
        "frozen_planner": "GestureChunkPlannerV08",
        "planning_mode": "gesture_v08",
        "train_root": train_root,
        "train_charts_screened": len(records) + len(skipped) + baseline_excluded,
        "baseline_charts_excluded": baseline_excluded,
        "train_charts_parsed": len(records),
        "baseline_bool_sha256_excluded": BOOL_SHA,
        "selection_pre_registered_axes": [
            "simultaneous_contacts", "small_inter_onset_gaps",
            "uneven_timing", "remaining_high_event_count",
        ],
        "cases": select_cases(records),
        "skipped": skipped,
        "execution_status": "selection_only; no physical evaluation performed",
        "limitations": [
            "Oracle labels only select charts, not a HUD-only policy.",
            "Three selected Train charts are exploratory, not representative.",
            "Do not tune gesture weights, reuse limits or Timing per chart.",
            "No Validation20 or FINAL charts are accessed by this command.",
            "Prior bool benchmark remains an independent frozen reference.",
        ],
    }


def verify_frozen_code(repo: Path) -> None:
    for source, expected in (
        ("src/dmdod/planning/gesture_chunk_planner_v08.py", FROZEN_PLANNER_BLOB),
        ("scripts/run_dmdogo_v06_pattern_first.py", FROZEN_RUNNER_BLOB),
    ):
        completed = subprocess.run(
            ["git", "hash-object", source], cwd=repo,
            capture_output=True, text=True, check=True,
        )
        if completed.stdout.strip() != expected:
            raise RuntimeError(
                f"frozen source changed: {source}; refuse to claim v0.8 baseline")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True,
                        help="Directory whose final component must be Train")
    parser.add_argument("--output", type=Path, required=True,
                        help="Create-once JSON manifest path")
    parser.add_argument("--max-charts", type=int, default=1000,
                        help="Fixed screening budget, sorted by path")
    args = parser.parse_args()
    try:
        verify_frozen_code(Path(__file__).resolve().parents[1])
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        parser.error(str(exc))
    root = args.train_root.resolve()
    if root.name.lower() != "train" or not root.is_dir():
        parser.error("--train-root must point to an existing directory named Train")
    if args.max_charts < 1:
        parser.error("--max-charts must be positive")
    if args.output.exists():
        parser.error("refusing to overwrite selection manifest")
    candidates = sorted(root.rglob("*.adofai"), key=lambda p: str(p).lower())
    if not candidates:
        parser.error("no .adofai charts inside this Train directory")
    records, skipped = [], []
    baseline_excluded = 0
    for path in candidates[:args.max_charts]:
        safe_path = path.resolve()
        if not safe_path.is_relative_to(root):
            skipped.append({"path": str(path), "reason": "symlink_outside_Train"})
            continue
        rel = path.relative_to(root)
        if any(part.lower() in FORBIDDEN for part in rel.parts):
            skipped.append({"path": str(path), "reason": "forbidden_subdirectory"})
            continue
        try:
            chart_sha = _sha256(safe_path)
            if chart_sha == BOOL_SHA:
                baseline_excluded += 1
                continue  # exclude the exact already evaluated chart
            compiled = compile_adofai(load_adofai(safe_path))
            segment = build_playable_segment(compiled)
            events = AutoplayTimingTeacher.from_segment(segment).events
            records.append(describe_events(
                tuple(events), name=path.name, sha256=chart_sha,
                path=str(path),
            ))
        except Exception as exc:
            skipped.append({
                "path": str(path), "reason": type(exc).__name__,
                "detail": str(exc)[:200],
            })
    manifest = make_manifest(records, skipped, train_root=str(root),
                             baseline_excluded=baseline_excluded)
    if not manifest["cases"]:
        parser.error("no eligible non-baseline Train charts; no output written")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(
        {"selected": manifest["cases"], "skipped": len(skipped),
         "manifest": str(args.output)},
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
