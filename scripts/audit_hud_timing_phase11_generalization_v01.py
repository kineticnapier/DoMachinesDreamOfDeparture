from __future__ import annotations

"""Phase 11: precommitted Train-only generalization audit, never a training loop.

Run --prepare first to freeze 16 chart names and hashes before any inference.
Run --audit with that manifest; this script never silently replaces charts.
All target associations and chart-event summaries are posthoc evaluator data.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from math import pi
import json
from pathlib import Path

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_hud_chart_v02 import HudCoupledContactRealChartEnv
from dmdod.features.countdown_hud_v3 import CountdownVisualSensor, StockCountdownRenderer
from dmdod.features.real_chart import DEFAULT_REAL_CHART_FEATURE_CONFIG as FLOORS
from dmdod.motor.contact_16k import ContactAction
from dmdod.planning.hud_v3_composite_forecast_source_v30 import (
    HUDV3CompositeForecastSource,
    LIFECYCLE_V14,
    LIFECYCLE_V15,
)
if __package__:
    from scripts.audit_hud_timing_phase8_lifecycle_integration_v01 import (
        _serialize, _evaluable_rows, _lifecycle_counts, _compare_rows,
    )
    from scripts.audit_timing_generalization_phase7_v01 import _evaluate
else:
    from audit_hud_timing_phase8_lifecycle_integration_v01 import (
        _serialize, _evaluable_rows, _lifecycle_counts, _compare_rows,
    )
    from audit_timing_generalization_phase7_v01 import _evaluate


FORMAT = "dmdod-timing-phase11-unseen-train16-v0.1"
MANIFEST_FORMAT = "dmdod-timing-phase11-frozen-selection-v0.1"
SELECTION_SALT = "dmdod-phase11-unseen-train16-20261019-v1"
TRAIN_DIR = Path("data/DMDOD-v140-tuf-p11-p20/Train")
N_UNSEEN = 16
UNSEEN_END_S = 12.0
CONTROL_DT_S = 0.010
ARMS = (
    "A-v14-identity-off",
    "B-v15-identity-off",
    "C-v15-identity-on",
)
REFERENCE = {
    "02334 - Time Leaper - Jipper.adofai": ("f4f71701065c986da6fd999cbc0a659d8fe47434793b5898e099ec2df83a5f70", 12.0),
    "02521 - Triple Cross - Milkyway.adofai": ("5571481d5a0eee9dee9fe70c81702b3bdf6dfd0491a6904c80d19e22610d86d3", 12.0),
    "07322 - Runaway - smy258.adofai": ("2e67ce666b72d72e61f353be22467eef60a10a6cbc935adc73baf833d3cdb27e", 12.0),
    "07827 - 24eeev0-$ - hotduck _ COMMON.adofai": ("3e93ec4afae58c516b367a45e1e504cf31628afd205e9f87b0f7f8de465782b3", 12.0),
    "04042 - bool - vo[ICE].adofai": ("04981e92f74288b8aac7286e83420b933036d2caead9c157bc33b706779ad1b1", 5.0),
}


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def candidate_names(train_dir: Path) -> list[str]:
    """Selection uses filenames only, never metrics, contents or chart difficulty."""
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Train80 not mounted: {train_dir}")
    candidates = [p.name for p in train_dir.glob("*.adofai")
                  if p.is_file() and p.name not in REFERENCE]
    if len(set(candidates)) != len(candidates):
        raise ValueError("duplicate Train filenames")
    return sorted(candidates)


def selected_names(names: list[str], *, n: int = N_UNSEEN) -> list[str]:
    if len(set(names)) != len(names) or any(name in REFERENCE for name in names):
        raise ValueError("selection candidates must be unique and unseen")
    if len(names) < n or n <= 0:
        raise ValueError(f"need at least {n} eligible unseen Train charts; found {len(names)}")
    return sorted(
        names,
        key=lambda name: (sha256((SELECTION_SALT + "\0" + name).encode("utf-8")).hexdigest(), name),
    )[:n]


def _pool_digest(names: list[str]) -> str:
    return sha256(json.dumps(sorted(names), ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def freeze_selection(train_dir: Path) -> dict:
    names = candidate_names(train_dir)
    chosen = selected_names(names)
    return {
        "format": MANIFEST_FORMAT,
        "selection_salt": SELECTION_SALT,
        "selection_rule": "sort SHA256(salt + NUL + UTF-8 filename), then filename; choose first 16",
        "candidate_pool_count": len(names),
        "candidate_pool_sha256": _pool_digest(names),
        "excluded_reference_charts": sorted(REFERENCE),
        "scope_seconds": UNSEEN_END_S,
        "charts": [{"filename": name, "sha256": _hash(train_dir / name)} for name in chosen],
    }


def verify_selection(manifest: dict, train_dir: Path) -> list[Path]:
    if manifest.get("format") != MANIFEST_FORMAT or manifest.get("selection_salt") != SELECTION_SALT:
        raise ValueError("unknown frozen selection format or salt")
    if manifest.get("scope_seconds") != UNSEEN_END_S:
        raise ValueError("audit scope differs from frozen selection")
    current = candidate_names(train_dir)
    if manifest.get("candidate_pool_count") != len(current) or manifest.get("candidate_pool_sha256") != _pool_digest(current):
        raise ValueError("Train filename pool changed after selection was frozen")
    chosen = selected_names(current)
    entries = manifest.get("charts")
    if not isinstance(entries, list) or len(entries) != N_UNSEEN:
        raise ValueError("manifest must freeze exactly 16 charts")
    if [entry.get("filename") for entry in entries] != chosen:
        raise ValueError("manifest differs from deterministic predeclared selection")
    paths = []
    for entry in entries:
        path = train_dir / entry["filename"]
        if not path.is_file() or _hash(path) != entry.get("sha256"):
            raise ValueError(f"frozen chart missing or hash changed: {entry['filename']}")
        paths.append(path)
    for name, (expected_hash, _) in REFERENCE.items():
        path = train_dir / name
        if not path.is_file() or _hash(path) != expected_hash:
            raise ValueError(f"reference chart missing or hash changed: {name}")
    return paths


def _source(arm: str) -> HUDV3CompositeForecastSource:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    return HUDV3CompositeForecastSource(
        horizon_s=.080,
        lifecycle_version=LIFECYCLE_V14 if arm == ARMS[0] else LIFECYCLE_V15,
        reconcile_countdown_v14_identity=(arm == ARMS[2]),
    )


def capture(path: Path, end_s: float, arms: tuple[str, ...]) -> tuple[dict[str, list[dict]], object]:
    """Capture ALL policy streams before reading any evaluator target timestamps."""
    chart = compile_adofai(load_adofai(path))
    segment = build_playable_segment(chart, end_s=end_s)
    renderer = StockCountdownRenderer(
        countdown_ticks=chart.countdown_ticks,
        first_tick_s=0.0,
        tick_period_s=60.0 / chart.initial_bpm / segment.pitch_ratio,
        go_duration_s=60.0 / chart.initial_bpm,
    )
    sensor = CountdownVisualSensor()
    sources = {arm: _source(arm) for arm in arms}
    env = HudCoupledContactRealChartEnv(
        segment,
        control_dt_s=CONTROL_DT_S, physics_dt_s=.001,
        behind_floors=FLOORS.behind_floors,
        ahead_floors=FLOORS.ahead_floors, tail_s=.040,
    )
    observation = env.reset()
    idle = ContactAction((0.0,) * 12)
    rows = {arm: [] for arm in arms}
    for step in range(int(end_s / CONTROL_DT_S) + 1):
        now_s = step * CONTROL_DT_S
        cue = sensor.observe(
            step=step, observed_at_s=now_s, frame=renderer.render(now_s),
        )
        for arm, source in sources.items():
            public, raw = source.observe(step, observation, cue)
            rows[arm].append(_serialize(public, raw))
        transition = env.step(idle)
        observation = transition.observation
        if transition.done:
            break
    return rows, segment


def _category_counts(events: list[dict]) -> dict:
    counts = Counter(str(item["category"]) for item in events)
    return {str(i): counts.get(str(i), 0) for i in range(1, 6)}


def _constructs(segment, end_s: float) -> dict:
    floors = [f for f in segment.chart.floors if f.target_time_s <= end_s + 1e-9]
    return {
        "countdown_ticks": segment.chart.countdown_ticks,
        "setspeed": sum("SetSpeed" in f.event_markers for f in floors),
        "pause": sum("Pause" in f.event_markers for f in floors),
        "twirl": sum("Twirl" in f.event_markers for f in floors),
        "midspin": sum(f.midspin for f in floors),
        "full_turn_non_midspin": sum(
            not f.midspin and f.angle_moved_rad >= 2 * pi - 1e-5 for f in floors
        ),
        "multiplanet": sum(f.num_planets > 2 for f in floors),
    }


def _case(path: Path, end_s: float, arms: tuple[str, ...]) -> dict:
    rows, segment = capture(path, end_s, arms)
    # All three streams have now finished. This is the ONLY point that truth is read.
    truths = tuple(float(t.episode_time_s) for t in segment.targets
                   if t.episode_time_s <= end_s + 1e-9)
    analyzed = {
        arm: _evaluate(_evaluable_rows(rows[arm], end_s), truths)
        for arm in arms
    }
    before = analyzed[arms[0]]["events"]
    outcome = {
        "chart": path.name,
        "sha256": _hash(path),
        "end_s": end_s,
        "event_count": len(truths),
        "constructs_posthoc": _constructs(segment, end_s),
        "arms": analyzed,
        "lifecycle_counts": {arm: _lifecycle_counts(rows[arm]) for arm in arms},
        "accepted_adapter_frames": {arm: len(rows[arm]) for arm in arms},
        "abstention_status_frames": {
            arm: {k: v for k, v in Counter(row["status"] for row in rows[arm]).items()
                  if k not in ("eligible", "eligible_partial", "eligible_lead_guaranteed",
                               "eligible_insufficient_lead")}
            for arm in arms
        },
    }
    if len(arms) == 3:
        final = analyzed[arms[2]]["events"]
        outcome["successful_event_regressions_A_to_C"] = [
            a["event_index"] for a, c in zip(before, final, strict=True)
            if a["category"] == 1 and c["category"] != 1
        ]
        outcome["successful_event_regressions_B_to_C"] = [
            b["event_index"] for b, c in zip(analyzed[arms[1]]["events"], final, strict=True)
            if b["category"] == 1 and c["category"] != 1
        ]
        outcome["forecast_differences_A_to_B"] = _compare_rows(
            rows[arms[0]], rows[arms[1]],
        )
        outcome["forecast_differences_B_to_C"] = _compare_rows(
            rows[arms[1]], rows[arms[2]],
        )
        shared = [
            i for i in range(len(truths))
            if all(analyzed[arm]["events"][i]["category"] == 1 for arm in arms)
        ]
        outcome["shared_success"] = {
            "event_indices": shared,
            "mae_ms": {
                arm: (sum(abs(analyzed[arm]["events"][i]["first_forecast_error_ms"]) for i in shared)
                      / len(shared) if shared else None)
                for arm in arms
            },
        }
        outcome["failure_details_final"] = [
            {"event_index": e["event_index"], "category": e["category"],
             "classification": e["classification"],
             "pre_event_statuses": e["pre_event_statuses"],
             "note": "A justified abstention is contract-relative, not a proof of global unobservability."}
            for e in final if e["category"] != 1
        ]
    return outcome


def aggregate(cases: list[dict], arm: str) -> dict:
    events = [event for case in cases for event in case["arms"][arm]["events"]]
    counts = _category_counts(events)
    clean = counts["1"]
    errors = [abs(e["first_forecast_error_ms"]) for e in events if e["category"] == 1]
    leads = [e["first_forecast_lead_ms"] for e in events if e["category"] == 1]
    return {
        "charts": len(cases),
        "events": len(events),
        "clean_events": clean,
        "clean_coverage": clean / len(events) if events else None,
        "per_chart_mean_clean_coverage": (
            sum(case["arms"][arm]["metrics"]["coverage"] for case in cases) / len(cases)
            if cases else None
        ),
        "category_counts": counts,
        "false_positive_ids": sum(case["arms"][arm]["metrics"]["false_positive_ids"] for case in cases),
        "timing_mae_ms_conditional_on_clean": sum(errors) / len(errors) if errors else None,
        "mean_forecast_lead_ms_conditional_on_clean": sum(leads) / len(leads) if leads else None,
        "lifecycle_counts": {
            status: sum(case["lifecycle_counts"][arm][status] for case in cases)
            for status in ("new", "reused", "canceled", "elapsed")
        },
    }


def run(manifest: dict, train_dir: Path) -> dict:
    paths = verify_selection(manifest, train_dir)   # freeze + hash all 21 before evaluation
    cases = [_case(path, UNSEEN_END_S, ARMS) for path in paths]
    reference_cases = [
        _case(train_dir / name, seconds, (ARMS[2],))
        for name, (_, seconds) in REFERENCE.items()
    ]
    reference_summary = aggregate(reference_cases, ARMS[2])
    aggregated = {arm: aggregate(cases, arm) for arm in ARMS}
    priority = Counter(
        status
        for case in cases
        for event in case["failure_details_final"]
        for status in (event["pre_event_statuses"] or ["unclassified"])
    )
    fp_nonincrease = aggregated[ARMS[2]]["false_positive_ids"] <= aggregated[ARMS[0]]["false_positive_ids"]
    regressions = sum(len(case["successful_event_regressions_A_to_C"]) for case in cases)
    old_to_new = all(not any(c["forecast_differences_A_to_B"][key]
                             for key in ("due_differences", "uncertainty_differences",
                                         "countdown_interval_differences"))
                     for c in cases)
    reference_ok = (reference_summary["events"] == 200
                    and reference_summary["clean_events"] == 199
                    and reference_summary["false_positive_ids"] == 0)
    structural = {
        "frozen_16_complete": len(cases) == 16,
        "reference_5_199_of_200": reference_ok,
        "A_B_forecast_invariant": old_to_new,
        "false_positives_not_increased_A_to_C": fp_nonincrease,
        "existing_success_regressions_A_to_C_zero": regressions == 0,
        "adapter_frames_accepted": all(
            all(n > 0 for n in case["accepted_adapter_frames"].values())
            for case in cases + reference_cases
        ),
    }
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "AUDIT_COMPLETED" if all(structural.values()) else "AUDIT_COMPLETED_WITH_FINDINGS",
        "no_coverage_acceptance_threshold": True,
        "manifest": manifest,
        "structural_checks": structural,
        "unseen_aggregate": aggregated,
        "reference_aggregate_C": reference_summary,
        "priority_status_counts_on_unsuccessful_events_C": priority.most_common(),
        "cases": cases,
        "reference_cases": reference_cases,
        "causality": {
            "policy": "current/past HUD263 and visible countdown cues only",
            "truth": "evaluated only after all arm streams per chart are captured",
            "no_learning_checkpoint_update_or_motor_policy_run": True,
            "abstention_not_global_impossibility_proof": True,
        },
    }


def _write_new(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "audit"))
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.output is not None:
            parser.error("--output is for audit mode only")
        _write_new(args.manifest, freeze_selection(args.train_dir))
        print("frozen:", args.manifest, "16 charts; no inference performed")
        return
    if args.output is None:
        parser.error("--output is required for audit mode")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = run(manifest, args.train_dir)
    _write_new(args.output, result)
    print("verdict:", result["verdict"])
    print("unseen:", result["unseen_aggregate"])
    print("reference:", result["reference_aggregate_C"])


if __name__ == "__main__":
    main()
