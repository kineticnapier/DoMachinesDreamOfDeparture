from __future__ import annotations

"""Bounded physical counterfactuals at saved Fingering v0.1.1 decisions.

Replay each frozen policy's SAVED decisions (no Network inference) to recover
exact internal 12-axis Motor state, verify against saved KeyDown/Hit metrics,
then compare the chosen axis with the previous event's axis from identical
physical snapshots. Every branch is a local motor preview, not a new rollout.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_oracle_fingering_v02 as teacher
import build_fingering_replay_viewer_v01 as viewer

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_chart_16k import CoupledContactRealChartEnv
from dmdod.motor.keyboard import KeyEvent
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher


def _events(payload: dict, policy: str):
    items = []
    for event in payload["events"]:
        row = (event.get("decisions") or {}).get(policy)
        if row is not None:
            items.append(row)
    items.sort(key=lambda r: (float(r["decision_s"]), int(r["event_index"])))
    if not items:
        raise ValueError(f"no saved {policy} decisions")
    if len({r["event_index"] for r in items}) != len(items):
        raise ValueError("duplicate saved decision index")
    if any(len(r["axes"]) != 1 for r in items):
        raise ValueError("only single-axis chart supported")
    return items


def _band(gap: float):
    if gap <= 35:
        return "up_to_35"
    if gap <= 100:
        return "35_to_100"
    if gap <= 200:
        return "100_to_200"
    return None


def _sample_anchors(rows: list[dict], per_band: int) -> dict[int, dict]:
    chronological = sorted(rows, key=lambda row: (row["due_s"], row["event_index"]))
    by_band = defaultdict(list)
    for earlier, later in zip(chronological, chronological[1:]):
        if earlier["axes"] == later["axes"]:
            continue
        delta_ms = (float(later["due_s"]) - float(earlier["due_s"])) * 1000.0
        if delta_ms < 0:
            raise ValueError("nonchronological event times")
        label = _band(delta_ms)
        if label is None:
            continue
        by_band[label].append({
            "event_index": later["event_index"],
            "previous_event_index": earlier["event_index"],
            "previous_axis": earlier["axes"][0],
            "gap_ms": round(delta_ms, 3),
            "band": label,
        })
    chosen = {}
    for band in ("up_to_35", "35_to_100", "100_to_200"):
        pool = by_band[band]
        if not pool:
            continue
        count = min(per_band, len(pool))
        indices = [(2 * i + 1) * len(pool) // (2 * count) for i in range(count)]
        for i in sorted(set(indices)):
            row = pool[i]
            chosen[row["event_index"]] = row
    return chosen


def _local_event(branch: dict, event_index: int) -> dict:
    event = next((row for row in branch["per_event"] if row["event_index"] == event_index), None)
    if event is None:
        raise RuntimeError("target event missing in counterfactual preview")
    errors = event["errors_ms"]
    best_error = min(errors, key=abs) if errors else None
    return {
        "keydown_error_ms": best_error,
        "hit_within_30ms": best_error is not None and abs(best_error) <= 30,
        "missing": event["missing"],
        "branch_cost": branch["cost"], "branch_missing": branch["missing"],
        "branch_extra": branch["extra"],
    }


def _snapshot_comparison(env, committed: dict, event, saved_row: dict, anchor: dict):
    now = env.privileged_episode_time_s()
    occupied = {axis for item in committed.values() for axis in item.axes}
    prev = teacher.Finger(anchor["previous_axis"],
                          "tip" if anchor["previous_axis"].endswith("_5") else "single")
    eligible, reason = teacher.eligible(env.motor, occupied, (prev,))
    saved_axes = list(saved_row["axes"])
    original = viewer.parse_allocation(saved_row["chosen"])
    due_s = event.episode_time_s
    current = teacher.PlanItem(event.event_index, due_s, original)
    alternative = teacher.PlanItem(event.event_index, due_s, (prev,))
    max_until = due_s + teacher.PREVIEW_TAIL
    # These previews do NOT include policies' later decisions or future adaptations.
    plans = tuple(committed.values())
    baseline = teacher.simulated_sequence(env.motor, plans + (current,), until_s=max_until)
    branch = teacher.simulated_sequence(env.motor, plans + (alternative,), until_s=max_until)
    return {
        **anchor, "decision_s": round(now, 6),
        "original_axis": saved_axes[0],
        "previous_axis_is_eligible": eligible,
        "previous_axis_ineligibility": reason,
        "occupied": anchor["previous_axis"] in occupied,
        "previous_axis_pressed": env.motor.motor.observe().pressed(anchor["previous_axis"]),
        "baseline": _local_event(baseline, event.event_index),
        "forced_previous_axis": _local_event(branch, event.event_index),
        "baseline_cost": baseline["cost"], "forced_cost": branch["cost"],
        "forced_minus_baseline_cost": round(branch["cost"] - baseline["cost"], 3),
    }


def _verify_run(env, saved_policy: dict, actual_downs: list[tuple[float, str]], policy: str):
    expected = [(float(row["time_ms"]), str(row["contact"]))
                for row in saved_policy["keydowns"]]
    if len(expected) != len(actual_downs):
        raise RuntimeError(f"{policy} fixed-replay KeyDown count drift: "
                           f"{len(actual_downs)} != {len(expected)}")
    max_delta_ms = 0.0
    for index, ((actual_ms, actual_contact), (expected_ms, expected_contact)) in enumerate(
        zip(actual_downs, expected)
    ):
        diff = abs(actual_ms - expected_ms)
        if actual_contact != expected_contact or diff > .002:
            raise RuntimeError(f"{policy} fixed-replay KeyDown drift at {index}: "
                               f"{actual_ms} {actual_contact} != "
                               f"{expected_ms} {expected_contact}")
        max_delta_ms = max(max_delta_ms, diff)
    source = saved_policy["saved_metrics"]
    stats = env.stats
    actual = {
        "targets": stats.targets, "hits": stats.hits, "misses": stats.misses,
        "too_early": stats.too_early_presses, "overloaded": bool(stats.overloaded),
        "physical_keydowns": env.physical_keydowns,
    }
    for key, value in actual.items():
        if value != source[key]:
            raise RuntimeError(f"{policy} fixed-replay metric drift: {key} "
                               f"{value} != {source[key]}")
    for field, number in (("x_accuracy_percent", stats.x_accuracy_percent),
                          ("timing_mae_ms", stats.mean_abs_error_ms)):
        if source[field] is not None and abs(float(source[field]) - float(number)) > 1e-8:
            raise RuntimeError(f"{policy} fixed-replay accuracy drift: {field}")
    return {
        "replay_exact": True,
        "physical_keydowns": len(expected),
        "max_saved_actual_keydown_delta_ms": round(max_delta_ms, 6),
    }


def run_saved_policy(segment, saved_policy: dict, payload: dict, policy: str,
                     cases_per_band: int) -> dict:
    rows = _events(payload, policy)
    anchors = _sample_anchors(rows, cases_per_band)
    oracle = AutoplayTimingTeacher.from_segment(segment)
    event_by_index = {event.event_index: event for event in oracle.events}
    env = CoupledContactRealChartEnv(segment, control_dt_s=teacher.CTRL,
                                     physics_dt_s=teacher.PHYS)
    env.reset()
    committed = {}
    next_row = 0
    cases = []
    actual_downs = []
    maximum = int((segment.duration_s + 2.0) / teacher.CTRL) + 200
    for _ in range(maximum):
        now = env.privileged_episode_time_s()
        committed = {
            index: plan for index, plan in committed.items()
            if plan.due_s + teacher.PREVIEW_TAIL >= now
            and not all(env._used[o] for o in event_by_index[index].target_ordinals)
        }
        while next_row < len(rows) and float(rows[next_row]["decision_s"]) <= now + 1e-7:
            row = rows[next_row]
            next_row += 1
            idx = row["event_index"]
            event = event_by_index[idx]
            if idx in anchors:
                cases.append(_snapshot_comparison(env, committed, event, row, anchors[idx]))
            allocation = viewer.parse_allocation(row["chosen"])
            if allocation:
                committed[idx] = teacher.PlanItem(idx, event.episode_time_s, allocation)
        action = teacher.action_for(env.motor, tuple(committed.values()), now)
        transition = env.step(action)
        for keydown in env.last_events:
            if keydown.event is KeyEvent.DOWN:
                actual_downs.append((keydown.time_s * 1000.0, keydown.key))
        if transition.done:
            break
    else:
        raise RuntimeError(f"{policy} replay exceeded frame budget")
    if next_row != len(rows):
        raise RuntimeError(f"{policy} terminated before recorded decisions: "
                           f"{next_row}/{len(rows)}")
    verified = _verify_run(env, saved_policy, actual_downs, policy)
    by_band = {}
    for label in ("up_to_35", "35_to_100", "100_to_200"):
        subset = [c for c in cases if c["band"] == label]
        by_band[label] = {
            "n": len(subset),
            "eligible_previous_axis": sum(c["previous_axis_is_eligible"] for c in subset),
            "reason_counts": dict(sorted(defaultdict(int, {
                reason: sum(c["previous_axis_ineligibility"] == reason for c in subset)
                for reason in ("reserved", "pressed", "topology_switch_unsafe", "eligible")
            }).items())),
            "forced_on_time": sum(c["forced_previous_axis"]["hit_within_30ms"] for c in subset),
            "baseline_on_time": sum(c["baseline"]["hit_within_30ms"] for c in subset),
            "forced_better_cost": sum(c["forced_minus_baseline_cost"] < -1e-6 for c in subset),
        }
    return {
        "verification": verified,
        "total_sampled_cases": len(cases),
        "by_band": by_band,
        "cases": cases,
    }


def analyze(replay: dict, chart_path: Path, cases_per_band: int) -> dict:
    if cases_per_band < 1 or cases_per_band > 20:
        raise ValueError("cases_per_band must be 1..20")
    if viewer.file_sha256(chart_path) != replay.get("chart_sha256"):
        raise ValueError("chart hash mismatch")
    if chart_path.name != replay.get("chart"):
        raise ValueError("chart filename mismatch")
    compiled = compile_adofai(load_adofai(chart_path))
    segment = build_playable_segment(compiled)
    if not all(name in replay.get("policies", {}) for name in ("free", "human_proxy")):
        raise ValueError("requires saved v0.1.1 comparison")
    result = {
        "format": "dmdod-fingering-v011-physical-previous-axis-counterfactual",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "chart": chart_path.name, "chart_sha256": viewer.file_sha256(chart_path),
        "cases_per_band": cases_per_band,
        "retraining": False, "inference_rerun": False,
        "policy": {},
    }
    for policy in ("free", "human_proxy"):
        result["policy"][policy] = run_saved_policy(
            segment, replay["policies"][policy], replay, policy, cases_per_band
        )
    result["limitations"] = [
        "All full-length base replays are VERIFIED against saved physical KeyDown timestamps and metrics.",
        "Counterfactual branches start from identical deep-copied physical Motor state.",
        "Branches keep previously committed plans and replace only the current allocation.",
        "Branches are short local previews with no Network rerun or later policy adaptation.",
        "The preview cost is an oracle-based diagnostic, NOT evidence of end-to-end chart improvement.",
        "Forced reuse may violate eligibility and must not be described as an available policy choice.",
        "A preview hit-within-30ms is only a local timing proxy, not the full ADOFAI judgement.",
    ]
    return result


def render_html(report: dict) -> str:
    rows = []
    for policy, item in report["policy"].items():
        for band, value in item["by_band"].items():
            rows.append("<tr>" + "".join(f"<td>{escape(str(x))}</td>" for x in (
                policy, band, value["n"], value["eligible_previous_axis"],
                value["reason_counts"].get("reserved", 0),
                value["reason_counts"].get("pressed", 0),
                value["baseline_on_time"], value["forced_on_time"],
                value["forced_better_cost"],
            )) + "</tr>")
    return ("<!doctype html><html lang='ja'><meta charset='utf-8'>"
            "<title>Fingering Motor counterfactual</title><style>"
            "body{background:#121826;color:#e2e8f0;font:15px/1.65 system-ui;"
            "max-width:1150px;margin:25px auto;padding:0 15px}"
            "table{width:100%;border-collapse:collapse}th,td{padding:9px;text-align:left;"
            "border-bottom:1px solid #42506a}</style><h1>同じMotor状態からの指再利用比較</h1>"
            f"<p>{escape(str(report['chart']))}／学習なし、保存済み運指の固定再生</p>"
            "<table><tr><th>方式</th><th>打鍵間隔</th><th>件数</th><th>前の指が適格</th>"
            "<th>予約</th><th>押下中</th><th>元候補が±30ms内</th>"
            "<th>前の指が±30ms内</th><th>前の指の局所コスト改善</th></tr>"
            + "".join(rows) + "</table><p>不適格な強制再利用は"
            "提案ではなく反事実的診断。全譜面の性能を保証しません。</p></html>\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chart", type=Path, required=True)
    p.add_argument("--replay-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cases-per-band", type=int, default=4)
    args = p.parse_args()
    if args.output_dir.exists():
        p.error(f"refusing to overwrite: {args.output_dir}")
    replay = json.loads(args.replay_json.read_text(encoding="utf-8"))
    report = analyze(replay, args.chart, args.cases_per_band)
    html = render_html(report)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "physical-counterfactual.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with (args.output_dir / "physical-counterfactual.html").open("x", encoding="utf-8") as f:
        f.write(html)
    for policy, item in report["policy"].items():
        print(f"{policy}: exact fixed replay={item['verification']['replay_exact']}; "
              f"{item['total_sampled_cases']} counterfactuals")
        for band, value in item["by_band"].items():
            print(f"  {band}: {value}")
    print("viewer:", args.output_dir / "physical-counterfactual.html")


if __name__ == "__main__":
    main()
