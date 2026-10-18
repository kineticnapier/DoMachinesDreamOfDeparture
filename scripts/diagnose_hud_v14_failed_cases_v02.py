from __future__ import annotations

"""Targeted differential replay for the four failures in HUD-v1.4 r1.

Only four predeclared fixture/Train cases, with the same three frozen arms.
All chart truth is used AFTER each policy run, never as HUD input. No model,
Motor, Timing parameter, validation data, or original artifacts are changed.

Create-only per-arm JSON allows interrupted experiments to retain completed
results. Summary compares missed target ordinals and physical KeyDown traces;
nearest-time correlation is NOT proof of the selected finger's causation.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_full_chart_fingering_replay_v02 as replay
import run_hud_v14_fingering_motor_integration_v02 as integration
from audit_hud_timing_tracker_transition_v14_matrix import generated
from dmdod.adofai.chart import load_adofai
from dmdod.adofai.timing import compile_adofai
from dmdod.adofai.playable import build_playable_segment


FORMAT = "dmdod-hud-v14-targeted-failure-crosscheck-v0.1"
CASES = (
    "ordinary-480bpm",
    "special-speed_double",
    "special-pause_1",
    "train-bool-5s",
)
ARMS = tuple(name for name, _, _ in integration.ARMS)


def get_case(name: str, *, chart: Path):
    if name == "ordinary-480bpm":
        return integration.synthetic_segment(480.)
    if name == "special-speed_double":
        return generated((0., 90., 180., 270.), 496., "speed_double")
    if name == "special-pause_1":
        return generated((0., 90., 180., 270.), 496., "pause_1")
    if name == "train-bool-5s":
        return build_playable_segment(
            compile_adofai(load_adofai(chart)), end_s=5.
        )
    raise ValueError("case outside frozen targeted regression list")


def _nearest_physical_down(
    physical_trace: list[dict], due_s: float, *, radius_s: float = .10
) -> list[dict]:
    """Posthoc physical down events near a truth event (not finger attribution)."""
    candidates = [
        {"time_s": float(edge["time_s"]),
         "axis": edge["axis"],
         "error_ms": round((edge["time_s"] - due_s) * 1000, 3)}
        for edge in physical_trace
        if str(edge["edge"]).lower().endswith("down")
        and abs(float(edge["time_s"]) - due_s) <= radius_s
    ]
    return sorted(candidates, key=lambda x: (abs(x["error_ms"]), x["axis"]))[:5]


def _per_target(result: dict, ordinal: int) -> dict:
    truth = next((x for x in result.get("target_outcomes_posthoc", ())
                  if x["ordinal"] == ordinal), None)
    if truth is None:
        return {"available": False, "reason": "arm errored or missing target trace"}
    due = truth["due_s"]
    matches = [
        x for x in result["first_seen_predictions"]
        if abs(x["due_local_s"] - due) <= .020
    ]
    closest = min(matches,
                  key=lambda x: (abs(x["due_local_s"] - due), x["local_id"]),
                  default=None)
    decision = next((x for x in result["decisions"]
                     if closest is not None and x["local_id"] == closest["local_id"]),
                    None)
    downs = _nearest_physical_down(result["physical_switch_trace"], due)
    if closest is None:
        category = "no_forecast_within_20ms"
    elif decision is None:
        category = "forecast_no_decision"
    elif decision["reason"] != "assigned":
        category = decision["reason"]
    elif not downs:
        category = "assigned_but_no_physical_down_within_100ms"
    elif truth["miss"]:
        category = "assigned_missed_needs_physical_comparison"
    else:
        category = "hit"
    return {
        "available": True,
        "hit": truth["hit"],
        "miss": truth["miss"],
        "due_s": due,
        "category": category,
        "closest_forecast": (
            {"local_id": closest["local_id"],
             "due_error_ms": round((closest["due_local_s"] - due) * 1000, 3),
             "first_seen_lead_ms": round(
                 (closest["due_local_s"] - closest["first_seen_s"]) * 1000, 3,
             )}
            if closest is not None else None
        ),
        "decision": (
            {"step": decision["step"], "reason": decision["reason"],
             "choice": decision["choice"],
             "extra_lead_ms": round(decision["extra_lead_s"] * 1000, 3)}
            if decision is not None else None
        ),
        "physical_downs_within_100ms": downs,
    }


def compare_case(arms: dict) -> dict:
    """Per-event cross-arm counterfactual comparison without changing policy."""
    failed = {name: entry["error"] for name, entry in arms.items()
              if "error" in entry}
    missed_ordinals = sorted({
        target["ordinal"]
        for record in arms.values()
        for target in record.get("target_outcomes_posthoc", ())
        if target["miss"]
    })
    comparisons = []
    for ordinal in missed_ordinals:
        comparisons.append({
            "ordinal": ordinal,
            "arms": {
                name: _per_target(record, ordinal)
                for name, record in arms.items()
            },
        })
    return {
        "errors": failed,
        "missed_ordinals_union": missed_ordinals,
        "per_missed_ordinal": comparisons,
        "arm_totals": {
            name: {
                "hits": record["stats"]["hits"],
                "misses": record["stats"]["misses"],
                "too_early": record["stats"]["too_early"],
                "x_accuracy_percent": record["stats"]["x_accuracy_percent"],
                "forecast_count": record["forecast_new_token_count"],
                "decision_reason_counts": {
                    reason: sum(x["reason"] == reason
                                for x in record["decisions"])
                    for reason in sorted(set(x["reason"]
                                             for x in record["decisions"]))
                },
            }
            for name, record in arms.items() if "stats" in record
        },
        "caveat": (
            "True target ordinals and physical timing are consulted posthoc. "
            "The nearest physical switch event is not proof of causal "
            "finger-to-target attribution. Do not optimize on Validation20."
        ),
    }


def _write_once(path: Path, obj: object) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=integration.DEFAULT_CHECKPOINT)
    parser.add_argument("--train-chart", type=Path,
                        default=integration.DEFAULT_TRAIN_CHART)
    parser.add_argument("--wall-timeout-s", type=float, default=180.)
    args = parser.parse_args()
    if args.wall_timeout_s <= 0:
        parser.error("wall timeout must be positive")
    if args.output_dir.exists():
        parser.error("output already exists; use a new path")
    if not args.checkpoint.exists():
        parser.error("frozen Fingering checkpoint not found")
    if not args.train_chart.is_file() or "train" not in (
        part.lower() for part in args.train_chart.parts
    ):
        parser.error("explicit Train chart required; never load Validation")
    model = replay.load_model(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cases": {},
        "method": "12 targeted replays, frozen three-arm comparison; no training",
    }
    errors = 0
    started = time.monotonic()
    for name in CASES:
        segment = get_case(name, chart=args.train_chart)
        case_dir = args.output_dir / name
        case_dir.mkdir()
        arms = {}
        for arm in ARMS:
            print(f"=== {name} / {arm} ===", flush=True)
            try:
                record = integration.simulate(
                    segment, model, arm, wall_timeout_s=args.wall_timeout_s
                )
            except Exception as exc:
                errors += 1
                record = {"arm": arm, "error": f"{type(exc).__name__}: {exc}"}
            _write_once(case_dir / f"{arm}.json", record)
            arms[arm] = record
            if "stats" in record:
                stats = record["stats"]
                print(
                    f"  {stats['hits']}/{stats['targets']} H, "
                    f"{stats['misses']} Miss, {stats['too_early']} TE",
                    flush=True,
                )
            else:
                print("  ERROR", record["error"], flush=True)
        comparison = compare_case(arms)
        _write_once(case_dir / "comparison.json", comparison)
        summary["cases"][name] = comparison
        if comparison["missed_ordinals_union"]:
            print("  missed target ordinals:",
                  comparison["missed_ordinals_union"], flush=True)
            for item in comparison["per_missed_ordinal"]:
                print("  ordinal", item["ordinal"], flush=True)
                for arm, detail in item["arms"].items():
                    print("   ", arm, json.dumps(detail, ensure_ascii=False),
                          flush=True)
    summary["wall_s"] = round(time.monotonic() - started, 3)
    summary["execution_errors"] = errors
    _write_once(args.output_dir / "comparison-summary.json", summary)
    print("saved:", args.output_dir / "comparison-summary.json", flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
