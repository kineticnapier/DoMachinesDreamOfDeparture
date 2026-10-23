from __future__ import annotations

"""Create-once abstract Pattern-first v0.6 schedule for an ADOFAI chart."""

import argparse
import json
from pathlib import Path
import time

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.planning.pattern_scheduler_v06 import (
    PatternEvent, PatternFirstScheduler, schedule_metrics,
)
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chart", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keys", type=int, default=8)
    parser.add_argument("--minimum-reuse-ms", type=float, default=80.0)
    parser.add_argument("--minimum-cycle-keys", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    compiled = compile_adofai(load_adofai(args.chart))
    segment = build_playable_segment(compiled)
    oracle = AutoplayTimingTeacher.from_segment(segment)
    events = tuple(PatternEvent(event.event_index, event.episode_time_s,
                                event.chord_size) for event in oracle.events)
    scheduler = PatternFirstScheduler(
        tuple(f"key_{index}" for index in range(args.keys)),
        minimum_reuse_s=args.minimum_reuse_ms / 1000.0,
        minimum_cycle_keys=args.minimum_cycle_keys,
    )
    started = time.perf_counter()
    schedule = scheduler.schedule(events)
    runtime_ms = 1000.0 * (time.perf_counter() - started)
    result = {
        "format": "dmdogo-pattern-first-schedule-v0.6",
        "chart": args.chart.name,
        "event_count": len(events),
        "runtime_ms": runtime_ms,
        "configuration": schedule.model_contract,
        "feasible": schedule.feasible,
        "failure_reason": schedule.failure_reason,
        "objective": schedule.objective,
        "metrics": schedule_metrics(schedule),
        "pattern_change_indices": schedule.pattern_change_indices,
        "patterns": [
            {"start_event_index": row.start_event_index,
             "end_event_index": row.end_event_index, "width": row.width,
             "cycle": row.cycle, "event_count": row.event_count}
            for row in schedule.patterns
        ],
        "events": [
            {"event_index": event.event_index, "due_s": event.due_s,
             "chord_size": event.chord_size,
             "keys": schedule.assignments[index] if schedule.feasible else ()}
            for index, event in enumerate(events)
        ],
        "student_inputs": "Oracle event times, chord sizes, declared logical key set only",
        "motor_used": False, "network_used": False, "training_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
