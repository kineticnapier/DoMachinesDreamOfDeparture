from __future__ import annotations

"""Read-only Free/Human-Proxy comparison on a dense Train chart.

The trained checkpoint, source chart, and existing artifacts are never written.
Output files are created once in a new directory. Timing remains oracle-based,
not a claim about the ability of a HUD-only Student to perform this chart.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_oracle_fingering_v02 as teacher
import build_fingering_replay_viewer_v01 as viewer
import build_fingering_replay_viewer_v03 as viewer_v03
import run_fingering_network_v011_comparison as comparison_v011
import run_full_chart_fingering_replay_v02 as runner

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher


CHART = Path("data/DMDOD-v140-tuf-p11-p20/Train/04042 - bool - vo[ICE].adofai")
CHART_SHA256 = "04981e92f74288b8aac7286e83420b933036d2caead9c157bc33b706779ad1b1"
CHECKPOINT = Path("checkpoints/fingering/fingering_network_v01-4fa2ae1.pt")
THRESHOLDS_MS = (35, 60, 100, 200, 500)


def _write_once(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as output:
        output.write(content)


def _stroke_log(result: dict) -> list[tuple[float, str]]:
    """Unique physical strokes; two logical contacts on one axis count once."""
    keydowns = sorted((float(row["time_ms"]), str(row["axis"]))
                      for row in result["keydowns"])
    strokes = []
    for at_ms, axis in keydowns:
        if strokes and strokes[-1][1] == axis and abs(strokes[-1][0] - at_ms) <= 0.001:
            continue
        strokes.append((at_ms, axis))
    return strokes


def reuse_statistics(result: dict) -> dict:
    strokes = _stroke_log(result)
    last_by_axis: dict[str, float] = {}
    repeated_gaps = []
    for at_ms, axis in strokes:
        if axis in last_by_axis:
            repeated_gaps.append(at_ms - last_by_axis[axis])
        last_by_axis[axis] = at_ms

    bands = (("0-35ms", 0, 35), ("35-60ms", 35, 60),
             ("60-100ms", 60, 100), ("100-200ms", 100, 200),
             ("200-500ms", 200, 500), ("500ms+", 500, float("inf")))
    histogram = {name: sum(lower <= value < upper for value in repeated_gaps)
                 for name, lower, upper in bands}

    adjacent = [((right_time - left_time), left_axis == right_axis)
                for (left_time, left_axis), (right_time, right_axis)
                in zip(strokes, strokes[1:])]
    adjacent_by_interval = {}
    for threshold in THRESHOLDS_MS:
        rows = [equal for gap, equal in adjacent if 0 <= gap <= threshold]
        adjacent_by_interval[str(threshold)] = {
            "pairs": len(rows),
            "same_axis": sum(rows),
            "different_axis": len(rows) - sum(rows),
            "same_axis_rate": round(sum(rows) / len(rows), 6) if rows else None,
        }

    # Candidate eligibility means not pressed/reserved, not guaranteed timely reuse.
    choices = sorted(result["decisions"], key=lambda row: (row["due_ms"], row["event_index"]))
    option_axes = [tuple(f.axis for f in option) for option in teacher.finger_options(1)]
    previous_eligible = {}
    for threshold in THRESHOLDS_MS:
        examined = available = reused = switched = 0
        for prev, row in zip(choices, choices[1:]):
            gap = float(row["due_ms"]) - float(prev["due_ms"])
            if not (0 <= gap <= threshold and len(prev["axes"]) == len(row["axes"]) == 1):
                continue
            examined += 1
            axis = prev["axes"][0]
            eligible = row.get("eligible", [])
            if any(len(option_axes[i]) == 1 and option_axes[i][0] == axis and ok
                   for i, ok in enumerate(eligible) if i < len(option_axes)):
                available += 1
                if row["axes"][0] == axis:
                    reused += 1
                else:
                    switched += 1
        previous_eligible[str(threshold)] = {
            "adjacent_single_note_pairs": examined,
            "previous_axis_candidate_eligible": available,
            "reused_previous_axis": reused,
            "chose_other_axis": switched,
        }

    return {
        "stroke_count": len(strokes),
        "axis_uses": dict(sorted(Counter(axis for _, axis in strokes).items())),
        "repeated_stroke_intervals": len(repeated_gaps),
        "same_axis_reuse_gap_histogram": histogram,
        "same_axis_reuse_within_ms": {
            str(threshold): sum(0 <= gap <= threshold for gap in repeated_gaps)
            for threshold in THRESHOLDS_MS
        },
        "adjacent_stroke_pairs_within_ms": adjacent_by_interval,
        "previous_axis_eligible_at_decision_within_ms": previous_eligible,
    }


def run(chart: Path, checkpoint: Path, output_dir: Path, *,
        expected_sha256: str, stop_after_s: float | None,
        wall_timeout_s: float) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    if viewer.file_sha256(chart) != expected_sha256:
        raise ValueError("source chart SHA256 mismatch; no experiment was started")
    if stop_after_s is not None and stop_after_s <= 0:
        raise ValueError("--stop-after-s must be positive")
    if wall_timeout_s <= 0:
        raise ValueError("--wall-timeout-s must be positive")

    compiled = compile_adofai(load_adofai(chart))
    segment = build_playable_segment(compiled)
    model = runner.load_model(checkpoint)
    started = time.perf_counter()
    policies = {}
    for name, alpha in (("free", 0.0), ("human_proxy", comparison_v011.FIXED_HUMAN_ALPHA)):
        policies[name] = runner.run_network(
            segment, model, stop_after_s=stop_after_s,
            wall_timeout_s=wall_timeout_s, style_alpha=alpha,
            policy_name=name,
        )
    elapsed = time.perf_counter() - started

    free, human = policies["free"], policies["human_proxy"]
    oracle = AutoplayTimingTeacher.from_segment(segment)
    event_times = {event.event_index: event.episode_time_s for event in oracle.events}
    difference = comparison_v011.compare_results(free, human, event_times)
    payload = comparison_v011.build_payload(chart, checkpoint, compiled, segment, free, human)
    summary = {
        "format": "dmdod-fingering-network-v011-short-reuse-audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "chart": chart.name,
        "chart_sha256": expected_sha256,
        "checkpoint_sha256": viewer.file_sha256(checkpoint),
        "chart_duration_s": segment.duration_s,
        "stop_after_s": stop_after_s,
        "runtime_s": round(elapsed, 3),
        "control_ms": 10,
        "physics_ms": 1,
        "timing_source": "exact AutoPlay Oracle (not HUD-only)",
        "human_proxy_alpha": comparison_v011.FIXED_HUMAN_ALPHA,
        "training_performed": False,
        "physical_counterfactual_teacher_used": False,
        "notes": [
            "Reuse means physical-axis KeyDown repetitions, not keyboard labels.",
            "Simultaneous logical contacts from one axis count as one stroke.",
            "Eligibility at decision does not guarantee a timely physical repeat.",
            "Prefix runs do not test the remainder of the chart.",
        ],
        "policy_results": {
            name: {
                "stats": result["stats"],
                "termination_reason": result["termination_reason"],
                "decisions": len(result["decisions"]),
                "reuse": reuse_statistics(result),
            }
            for name, result in policies.items()
        },
        "comparison": {
            "different_event_count": difference["different_event_count"],
            "human_proxy_worsened_event_count": difference["human_proxy_worsened_event_count"],
            "human_proxy_improved_timing_event_count": difference["human_proxy_improved_timing_event_count"],
            "special_fingering_avoided_event_count": difference["special_fingering_avoided_event_count"],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_once(output_dir / "short-reuse-summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    _write_once(output_dir / "replay-v011.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _write_once(output_dir / "replay-v011.html", viewer_v03.render_html(payload))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chart", type=Path, default=CHART)
    parser.add_argument("--expected-sha256", default=CHART_SHA256)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stop-after-s", type=float)
    parser.add_argument("--wall-timeout-s", type=float, default=300.0)
    args = parser.parse_args()
    summary = run(args.chart, args.checkpoint, args.output_dir,
                  expected_sha256=args.expected_sha256,
                  stop_after_s=args.stop_after_s,
                  wall_timeout_s=args.wall_timeout_s)
    for name in ("free", "human_proxy"):
        row = summary["policy_results"][name]
        strokes = row["reuse"]["stroke_count"]
        repeats = row["reuse"]["same_axis_reuse_within_ms"]["100"]
        eligible = row["reuse"]["previous_axis_eligible_at_decision_within_ms"]["100"]
        print(f"{name}: {row['stats']['hits']} hits, {row['stats']['misses']} misses; "
              f"{strokes} strokes, same-axis repeats <=100ms: {repeats}; "
              f"eligible previous-axis choices <=100ms: "
              f"{eligible['reused_previous_axis']}/{eligible['previous_axis_candidate_eligible']}")
    print(f"different event choices: {summary['comparison']['different_event_count']}")
    print(f"viewer: {args.output_dir / 'replay-v011.html'}")
    print(f"report: {args.output_dir / 'short-reuse-summary.json'}")


if __name__ == "__main__":
    main()
