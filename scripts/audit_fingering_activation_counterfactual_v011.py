from __future__ import annotations

"""Activation-zero physical counterfactual on frozen Fingering v0.1.1 events.

Uses the existing, VERIFIED Motor-onset audit to select uniformly spaced
anchors. Replays saved allocations exactly. At each anchor, deep-copies the
whole chart evaluator (including Motor), sets ONLY the selected finger's
activation to zero in one copy, and runs both local branches with the same
saved future allocations and timing. The original branch MUST reproduce the
saved physical KeyDown timestamp; otherwise abort without writing output.

This is a *mechanistic* intervention, not a physically executable action,
and it never edits the Motor model, teacher, Network, chart or checkpoint.
"""

import argparse
import copy
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from statistics import mean, median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_oracle_fingering_v02 as teacher
import audit_fingering_physical_reuse_counterfactual_v011 as fixed
import build_fingering_replay_viewer_v01 as viewer

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_chart_16k import CoupledContactRealChartEnv
from dmdod.motor.keyboard import KeyEvent
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher


BANDS = ("first_use", "reuse_100_200ms", "reuse_200_500ms", "reuse_ge_500ms")
BASELINE_TOLERANCE_MS = .010


def choose_events(onset_report: dict, policy: str, per_band: int) -> dict[int, dict]:
    """Uniformly spaced event selection within each physical reuse band."""
    available = defaultdict(list)
    for row in onset_report["policies"][policy]["events"]:
        band = row["reuse_band"]
        if band not in BANDS or float(row["activation_at_command"]) >= -1e-7:
            continue
        available[band].append(row)
    picks = {}
    for band in BANDS:
        group = sorted(available[band], key=lambda r: int(r["event_index"]))
        n = min(per_band, len(group))
        for i in range(n):
            index = (2 * i + 1) * len(group) // (2 * n)
            row = group[index]
            picks[int(row["event_index"])] = row
    return picks


def _scheduled_commit(env, committed: dict, rows: list[dict],
                      event_by_index: dict, cursor: int, now: float) -> int:
    """The frozen allocation replay uses the production controller unchanged."""
    committed.update({})  # Make it explicit that callers own the mutable mapping.
    expired = [
        index for index, item in committed.items()
        if item.due_s + teacher.PREVIEW_TAIL < now
        or all(env._used[o] for o in event_by_index[index].target_ordinals)
    ]
    for index in expired:
        del committed[index]
    while cursor < len(rows) and float(rows[cursor]["decision_s"]) <= now + 1e-7:
        row = rows[cursor]
        cursor += 1
        allocation = viewer.parse_allocation(row["chosen"])
        if allocation:
            event = event_by_index[row["event_index"]]
            committed[row["event_index"]] = teacher.PlanItem(
                event.event_index, event.episode_time_s, allocation
            )
    return cursor


def _branch(env, committed: dict, rows: list[dict], cursor: int,
            event_by_index: dict, event_index: int,
            axis: str, due_s: float, *, zero_activation: bool) -> dict:
    """One snapshot branch to due+35ms, with future saved decisions."""
    env = copy.deepcopy(env)
    committed = dict(committed)
    motor = env.motor.motor
    finger_state = motor.body.states[motor.key_names.index(axis)]
    original_activation = float(finger_state.activation)
    if zero_activation:
        finger_state.activation = 0.0

    raw_keydowns = []
    logical_keydowns = 0
    max_end = due_s + teacher.PREVIEW_TAIL
    # Event is normally selected roughly 39ms before its due time.
    for _ in range(30):
        now = env.privileged_episode_time_s()
        if now >= max_end - 1e-9 or env._done:
            break
        cursor = _scheduled_commit(
            env, committed, rows, event_by_index, cursor, now
        )
        action = teacher.action_for(env.motor, tuple(committed.values()), now)
        env.step(action)
        for event in env.motor.motor.last_transition.evaluator_events:
            if event.key == axis and event.event is KeyEvent.DOWN:
                raw_keydowns.append(round(event.time_s * 1000, 6))
        for event in env.last_events:
            if event.event is KeyEvent.DOWN:
                logical_keydowns += 1
    else:
        raise RuntimeError("bounded local branch exceeded 30 control steps")

    due_ms = due_s * 1000.0
    best = min(raw_keydowns, key=lambda t: (abs(t - due_ms), t)) if raw_keydowns else None
    ordinals = event_by_index[event_index].target_ordinals
    return {
        "activation_initial": round(original_activation, 9),
        "activation_intervention": 0.0 if zero_activation else round(original_activation, 9),
        "keydown_ms": best,
        "signed_error_ms": round(best - due_ms, 3) if best is not None else None,
        "absolute_error_ms": round(abs(best - due_ms), 3) if best is not None else None,
        "raw_axis_keydowns_in_window": len(raw_keydowns),
        "logical_keydowns_in_window": logical_keydowns,
        "event_judged_hit": all(env._used[i] for i in ordinals),
        "event_judged_miss": any(env._missed[i] for i in ordinals),
        "end_time_ms": round(env.privileged_episode_time_s() * 1000, 3),
        "end_overloaded": env.stats.overloaded,
        "end_too_early": env.stats.too_early_presses,
    }


def _replay_policy(segment, replay: dict, onset_report: dict,
                   policy: str, per_band: int) -> dict:
    saved_policy = replay["policies"][policy]
    rows = fixed._events(replay, policy)
    oracle = AutoplayTimingTeacher.from_segment(segment)
    event_by_index = {e.event_index: e for e in oracle.events}
    chosen = choose_events(onset_report, policy, per_band)

    env = CoupledContactRealChartEnv(
        segment, control_dt_s=teacher.CTRL, physics_dt_s=teacher.PHYS
    )
    env.reset()
    committed = {}
    cursor = 0
    actual_downs = []
    captured = []
    already_captured = set()
    max_cycles = int((segment.duration_s + 2.0) / teacher.CTRL) + 200

    for _ in range(max_cycles):
        now = env.privileged_episode_time_s()
        cursor = _scheduled_commit(
            env, committed, rows, event_by_index, cursor, now
        )
        action = teacher.action_for(env.motor, tuple(committed.values()), now)

        for ix, anchor in chosen.items():
            if ix in already_captured or ix not in committed:
                continue
            axis = anchor["axis"]
            if action.commands[teacher.AXES.index(axis)] <= 0:
                continue
            original = env.motor.motor.body.states[
                env.motor.motor.key_names.index(axis)
            ].activation
            if abs(float(original) - float(anchor["activation_at_command"])) > 1e-5:
                raise RuntimeError(
                    f"{policy} #{ix}: replay activation drift vs onset audit: "
                    f"{original} != {anchor['activation_at_command']}"
                )
            if abs(now * 1000 - float(anchor["first_positive_command_ms"])) > .011:
                raise RuntimeError(f"{policy} #{ix}: onset time drift")
            due_s = event_by_index[ix].episode_time_s
            control = _branch(
                env, committed, rows, cursor, event_by_index, ix, axis,
                due_s, zero_activation=False
            )
            intervention = _branch(
                env, committed, rows, cursor, event_by_index, ix, axis,
                due_s, zero_activation=True
            )
            expected_keydown = float(anchor["actual_keydown_ms"])
            if control["keydown_ms"] is None or abs(
                control["keydown_ms"] - expected_keydown
            ) > BASELINE_TOLERANCE_MS:
                raise RuntimeError(
                    f"{policy} #{ix}: baseline branch does not reproduce "
                    f"recorded KeyDown {control['keydown_ms']} vs {expected_keydown}"
                )
            if not control["event_judged_hit"]:
                raise RuntimeError(
                    f"{policy} #{ix}: baseline branch failed to reproduce Hit"
                )
            actual = intervention["keydown_ms"]
            captured.append({
                "event_index": ix,
                "reuse_band": anchor["reuse_band"],
                "axis": axis,
                "due_ms": anchor["due_ms"],
                "start_activation": anchor["activation_at_command"],
                "start_position_mm": anchor["position_at_command_mm"],
                "start_velocity_mm_s": anchor["velocity_at_command_mm_s"],
                "start_finger_fatigue": anchor["finger_fatigue_at_command"],
                "baseline": control,
                "activation_zero": intervention,
                "keydown_shift_ms": (
                    round(actual - control["keydown_ms"], 3)
                    if actual is not None else None
                ),
                "absolute_error_improvement_ms": (
                    round(control["absolute_error_ms"] - intervention["absolute_error_ms"], 3)
                    if actual is not None else None
                ),
            })
            already_captured.add(ix)

        env.step(action)
        for event in env.last_events:
            if event.event is KeyEvent.DOWN:
                actual_downs.append((float(event.time_s) * 1000, event.key))
        if env._done:
            break
    else:
        raise RuntimeError(f"{policy}: fixed replay overran max cycles")

    if cursor != len(rows):
        raise RuntimeError(f"{policy}: fixed replay did not reach all saved decisions")
    checked = fixed._verify_run(env, saved_policy, actual_downs, policy)
    if len(captured) != len(chosen):
        raise RuntimeError(f"{policy}: captured {len(captured)}/{len(chosen)} anchors")

    by_band = {}
    for band in BANDS:
        selected = [item for item in captured if item["reuse_band"] == band]
        shifts = [item["keydown_shift_ms"] for item in selected
                  if item["keydown_shift_ms"] is not None]
        improvements = [item["absolute_error_improvement_ms"] for item in selected
                        if item["absolute_error_improvement_ms"] is not None]
        by_band[band] = {
            "n": len(selected),
            "activation0_keydown_recorded": len(shifts),
            "activation0_hit": sum(item["activation_zero"]["event_judged_hit"] for item in selected),
            "activation0_miss": sum(item["activation_zero"]["event_judged_miss"] for item in selected),
            "mean_keydown_shift_ms": round(mean(shifts), 3) if shifts else None,
            "median_keydown_shift_ms": round(median(shifts), 3) if shifts else None,
            "mean_absolute_error_improvement_ms": round(mean(improvements), 3)
            if improvements else None,
            "improved_abs_error": sum(value > .0001 for value in improvements),
            "worsened_abs_error": sum(value < -.0001 for value in improvements),
            "unchanged_abs_error": sum(abs(value) <= .0001 for value in improvements),
            "mean_initial_activation": round(mean(
                float(item["start_activation"]) for item in selected
            ), 5) if selected else None,
        }

    return {
        "verified_full_replay": checked,
        "sampled_events": len(captured),
        "by_reuse_band": by_band,
        "events": sorted(captured, key=lambda d: d["event_index"]),
    }


def analyze(chart: Path, replay: dict, onset_report: dict,
            cases_per_band: int) -> dict:
    if not 1 <= cases_per_band <= 40:
        raise ValueError("--cases-per-band must be 1..40")
    if viewer.file_sha256(chart) != replay.get("chart_sha256"):
        raise ValueError("chart/replay SHA256 mismatch")
    if chart.name != replay.get("chart"):
        raise ValueError("chart/replay name mismatch")
    if onset_report.get("chart_sha256") != replay.get("chart_sha256"):
        raise ValueError("onset/replay chart SHA256 mismatch")
    if onset_report.get("checkpoint_sha256") != replay.get("checkpoint_sha256"):
        raise ValueError("onset/replay checkpoint SHA256 mismatch")
    for policy in ("free", "human_proxy"):
        valid = onset_report["policies"][policy]["fixed_replay_verification"]
        if valid.get("replay_exact") is not True:
            raise ValueError(f"{policy}: onset audit not verified")
    compiled = compile_adofai(load_adofai(chart))
    segment = build_playable_segment(compiled)
    return {
        "format": "dmdod-fingering-v011-activation-only-counterfactual",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "chart": chart.name,
        "chart_sha256": replay["chart_sha256"],
        "checkpoint_sha256": replay["checkpoint_sha256"],
        "intervention": "selected_axis_activation=0 at first positive command",
        "cases_per_band": cases_per_band,
        "policies": {
            policy: _replay_policy(segment, replay, onset_report, policy, cases_per_band)
            for policy in ("free", "human_proxy")
        },
        "limitations": [
            "A zeroed activation is a nonphysical diagnostic intervention, not a policy action.",
            "Finger position, velocity, fatigue, other finger states and full future fixed decisions are unchanged at intervention time.",
            "After intervention, all Motor state evolves according to the original equations.",
            "Both arms use same original action_for and saved allocations; no Network inference.",
            "Original local branch must match saved physical KeyDown within 0.010ms and Hit.",
            "Hit/miss is checked only within due+35ms; end-to-end changed-policy replay is not claimed.",
            "Event band sampling is evenly spaced within the chosen observational group, not random.",
            "The motor profile is experimental and not calibrated to actual human physiology.",
        ],
    }


def render_html(report: dict) -> str:
    rows = []
    for policy in ("free", "human_proxy"):
        for band in BANDS:
            item = report["policies"][policy]["by_reuse_band"][band]
            rows.append("<tr>" + "".join(
                f"<td>{escape(str(v) if v is not None else '—')}</td>"
                for v in (
                    policy, band, item["n"], item["activation0_keydown_recorded"],
                    item["activation0_hit"], item["mean_initial_activation"],
                    item["mean_keydown_shift_ms"],
                    item["mean_absolute_error_improvement_ms"],
                    item["improved_abs_error"], item["worsened_abs_error"]
                )
            ) + "</tr>")
    return (
        "<!doctype html><html lang='ja'><meta charset='utf-8'>"
        "<title>Activation only Motor counterfactual</title>"
        "<style>body{max-width:1200px;margin:24px auto;padding:0 16px;"
        "background:#101822;color:#f2f5fc;font:15px/1.6 system-ui}"
        "table{width:100%;border-collapse:collapse}"
        "th,td{text-align:left;padding:8px;border-bottom:1px solid #425069}"
        "th{background:#24324a}</style>"
        "<h1>指の残留活性化だけを0にする反事実的診断</h1>"
        "<p>オリジナルの物理KeyDown・Hitを厳密照合した局所分岐のみ。</p>"
        "<table><thead><tr><th>方式</th><th>同一指再利用間隔</th><th>ケース数</th>"
        "<th>介入側KeyDown</th><th>介入側Hit</th><th>元の活性化</th>"
        "<th>KeyDown変化(ms)</th><th>絶対Timing誤差改善(ms)</th>"
        "<th>改善件数</th><th>悪化件数</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
        "<p>KeyDown変化が負なら早くなる。改善値が正なら予定時刻に近づく。"
        "瞬間的な活性化変更は実際には使用できない。</p></html>\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chart", type=Path, required=True)
    p.add_argument("--replay-json", type=Path, required=True)
    p.add_argument("--onset-json", type=Path, required=True)
    p.add_argument("--cases-per-band", type=int, default=12)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        p.error(f"refusing to overwrite existing directory: {args.output_dir}")
    replay = json.loads(args.replay_json.read_text(encoding="utf-8"))
    onset_report = json.loads(args.onset_json.read_text(encoding="utf-8"))
    result = analyze(args.chart, replay, onset_report, args.cases_per_band)
    html = render_html(result)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "activation-counterfactual.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with (args.output_dir / "activation-counterfactual.html").open("x", encoding="utf-8") as f:
        f.write(html)
    for policy, result in result["policies"].items():
        print(f"{policy}: exact fixed replay={result['verified_full_replay']['replay_exact']}; "
              f"local cases={result['sampled_events']}")
        for band in BANDS:
            row = result["by_reuse_band"][band]
            print(f"  {band}: n={row['n']}; "
                  f"keydown shift={row['mean_keydown_shift_ms']}ms; "
                  f"abs error improvement={row['mean_absolute_error_improvement_ms']}ms; "
                  f"intervention hit={row['activation0_hit']}/{row['n']}")
    print("viewer:", args.output_dir / "activation-counterfactual.html")


if __name__ == "__main__":
    main()
