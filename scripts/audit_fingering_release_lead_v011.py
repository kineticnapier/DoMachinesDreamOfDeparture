from __future__ import annotations

"""Frozen-policy diagnostic: release control, anticipatory lead, and both.

This script does NOT change the production controller, model or Motor profile.
Four full-chart arms are run for each saved Free/Human-Proxy fingering sequence:

  baseline  : exact existing action_for and decision times (must fully match).
  release   : coast (command=0) near the reset threshold, if moving outward.
  lead      : proactively press an axis based on the stored *future* allocation.
  combined  : apply both changes.

The anticipatory arm is NONCAUSAL: saved future Fingering Network choices are
revealed before the original ~39ms decision time. It is a diagnostic upper bound,
NOT a deployable Student or a validated online Network controller.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
from math import log1p
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_fingering_physical_reuse_counterfactual_v011 as frozen
import audit_oracle_fingering_v02 as teacher
import build_fingering_replay_viewer_v01 as viewer

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_chart_16k import CoupledContactRealChartEnv
from dmdod.motor.contact_16k import ContactAction
from dmdod.motor.keyboard import KeyEvent
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher


ARMS = ("baseline", "release", "lead", "combined")
RELEASE_COAST_MARGIN_M = .00030
MAX_EXTRA_LEAD_S = .025
ACTIVATION_GATE = -.010
EARLY_AXIS_DEPTH_MAX = 4


def release_coast(motor, base: ContactAction) -> tuple[ContactAction, int]:
    """Turn off outward muscle drive close to physical KeyUp (still in motion).

    This is *not* a direct state change. It may delay KeyUp or cause missed hits.
    It applies only when the original controller commands negative release.
    """
    obs = motor.motor.observe()
    commands = list(base.commands)
    reset = motor.motor.keyboard.config.reset_m
    changed = 0
    for i, axis in enumerate(teacher.AXES):
        if (commands[i] < -.999 and obs.pressed(axis)
                and obs.positions_m[i] <= reset + RELEASE_COAST_MARGIN_M
                and obs.velocities_m_s[i] < 0):
            commands[i] = 0.0
            changed += 1
    if not changed:
        return base, 0
    return ContactAction(tuple(commands), base.thumb_modes, base.edge_enabled), changed


def physical_extra_lead_s(activation: float, tau_s: float) -> float:
    """Bounded theoretical turnaround allowance, not learned/tuned."""
    if activation >= ACTIVATION_GATE:
        return 0.0
    if tau_s <= 0:
        raise ValueError("invalid Motor activation time constant")
    return min(MAX_EXTRA_LEAD_S, tau_s * log1p(-activation))


def apply_early_lead(motor, base: ContactAction, future_rows: list,
                     now_s: float, occupied: set[str],
                     armed: dict[int, str]) -> tuple[ContactAction, dict, int]:
    """Preview allocations not yet selected by Network; noncausal bound.

    No user-selected axis changes. Only ordinary contacts and only if idle,
    not reserved, currently eligible, and negative activation. Once armed,
    keep commanding until normal decision time or until physically pressed.
    """
    commands = list(base.commands)
    obs = motor.motor.observe()
    newly_armed = 0
    for row in future_rows:
        event_index = int(row["event_index"])
        decision_s = float(row["decision_s"])
        due_s = float(row["due_s"])
        if decision_s <= now_s + 1e-7:
            continue
        if due_s > now_s + teacher.LEAD + MAX_EXTRA_LEAD_S + 1e-9:
            # rows are ordered by decision time, due times are monotonic here.
            continue
        allocation = viewer.parse_allocation(row["chosen"])
        if len(allocation) != 1:
            continue
        finger = allocation[0]
        axis = finger.axis
        if int(axis.rsplit("_", 1)[1]) > EARLY_AXIS_DEPTH_MAX:
            continue
        if axis in occupied or obs.pressed(axis):
            continue
        i = teacher.AXES.index(axis)
        if event_index not in armed:
            if not teacher.eligible(motor, occupied, allocation)[0]:
                continue
            state = motor.motor.body.states[i]
            tau_s = motor.motor.body.configs[i].activation_tau_s
            extra = physical_extra_lead_s(state.activation, tau_s)
            if extra <= 0 or now_s + teacher.LEAD + extra + 1e-9 < due_s:
                continue
            armed[event_index] = axis
            newly_armed += 1
        if armed[event_index] == axis:
            commands[i] = 1.0
    if commands == list(base.commands):
        return base, armed, newly_armed
    return ContactAction(tuple(commands), base.thumb_modes, base.edge_enabled), armed, newly_armed


def _metrics(env: CoupledContactRealChartEnv, decisions: list, ordinals: dict,
             decision_count: int, intervention_stats: Counter, early_armed: dict) -> dict:
    stats = env.stats
    # The physical scorer, rather than proximity of KeyDown to an event,
    # determines which specific targets were really hit.
    per_event = {}
    for row in decisions:
        ix = int(row["event_index"])
        used = [env._used[i] for i in ordinals[ix]]
        missed = [env._missed[i] for i in ordinals[ix]]
        per_event[ix] = ("Hit" if all(used) else "Miss" if any(missed)
                         else "Unresolved")
    return {
        "targets": stats.targets, "hits": stats.hits, "misses": stats.misses,
        "too_early": stats.too_early_presses,
        "overloaded": bool(stats.overloaded),
        "x_accuracy_percent": stats.x_accuracy_percent,
        "timing_mae_ms": stats.mean_abs_error_ms,
        "physical_keydowns": env.physical_keydowns,
        "decisions_committed": decision_count,
        "expected_decisions": len(decisions),
        "episode_end_s": round(env.privileged_episode_time_s(), 6),
        "event_outcomes": per_event,
        "release_coast_axes_steps": intervention_stats["release_coast_axes_steps"],
        "anticipatory_events_armed": len(early_armed),
        "anticipatory_positive_axis_steps": intervention_stats["anticipatory_positive_axis_steps"],
        "future_allocation_leak": bool(early_armed),
    }


def replay_policy(segment, replay: dict, policy: str, arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unknown experiment arm: {arm}")
    rows = frozen._events(replay, policy)
    oracle = AutoplayTimingTeacher.from_segment(segment)
    by_event = {e.event_index: e for e in oracle.events}
    ordinals = {e.event_index: e.target_ordinals for e in oracle.events}
    env = CoupledContactRealChartEnv(
        segment, control_dt_s=teacher.CTRL, physics_dt_s=teacher.PHYS
    )
    env.reset()
    committed = {}
    cursor = 0
    keydowns = []
    interventions = Counter()
    armed: dict[int, str] = {}
    max_steps = int((segment.duration_s + 2.0) / teacher.CTRL) + 200

    for _ in range(max_steps):
        now = env.privileged_episode_time_s()
        committed = {
            ix: item for ix, item in committed.items()
            if item.due_s + teacher.PREVIEW_TAIL >= now
            and not all(env._used[o] for o in ordinals[ix])
        }
        while cursor < len(rows) and float(rows[cursor]["decision_s"]) <= now + 1e-7:
            row = rows[cursor]
            cursor += 1
            allocation = viewer.parse_allocation(row["chosen"])
            if allocation:
                e = by_event[row["event_index"]]
                committed[e.event_index] = teacher.PlanItem(
                    e.event_index, e.episode_time_s, allocation
                )
            # The pre-announced choice is now available causally.
            armed.pop(int(row["event_index"]), None)
        action = teacher.action_for(env.motor, tuple(committed.values()), now)
        if arm in ("release", "combined"):
            action, changed = release_coast(env.motor, action)
            interventions["release_coast_axes_steps"] += changed
        if arm in ("lead", "combined"):
            occupied = {axis for item in committed.values() for axis in item.axes}
            # The future saved fingering is privileged information for this
            # diagnostic. It must not be represented as causal Network output.
            action, armed, new_count = apply_early_lead(
                env.motor, action, rows[cursor:cursor + 16],
                now, occupied, armed
            )
            interventions["anticipatory_events_new"] += new_count
            interventions["anticipatory_positive_axis_steps"] += sum(
                action.commands[teacher.AXES.index(axis)] > 0
                for ix, axis in armed.items()
            )
        result = env.step(action)
        for event in env.last_events:
            if event.event is KeyEvent.DOWN:
                keydowns.append((float(event.time_s) * 1000.0, event.key))
        if result.done:
            break
    else:
        raise RuntimeError(f"{policy}/{arm}: fixed replay exceeded frame budget")

    if arm == "baseline":
        if cursor != len(rows):
            raise RuntimeError(f"{policy}: baseline failed to commit all recorded decisions")
        verified = frozen._verify_run(
            env, replay["policies"][policy], keydowns, policy
        )
    else:
        verified = {"baseline_reference_verified_separately": True}

    outcomes = _metrics(env, rows, ordinals, cursor, interventions, armed)
    if arm != "baseline":
        outcomes["future_allocation_leak"] = arm in ("lead", "combined")
    outcomes.pop("event_outcomes", None)
    baseline_stats = replay["policies"][policy]["saved_metrics"]
    outcomes["saved_original_x_accuracy_percent"] = baseline_stats["x_accuracy_percent"]
    return {
        "mode": arm, "policy": policy,
        "verified_replay": verified,
        "metrics": outcomes,
    }


def analyze(replay: dict, chart: Path) -> dict:
    if chart.name != replay.get("chart") or viewer.file_sha256(chart) != replay.get("chart_sha256"):
        raise ValueError("chart name/hash mismatch")
    if not all(name in replay.get("policies", {}) for name in ("free", "human_proxy")):
        raise ValueError("Free + Human-Proxy replay required")
    segment = build_playable_segment(compile_adofai(load_adofai(chart)))
    result = {
        "format": "dmdod-fingering-v011-release-lead-cf",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "chart": chart.name,
        "chart_sha256": replay["chart_sha256"],
        "checkpoint_sha256": replay["checkpoint_sha256"],
        "parameters": {
            "release_coast_margin_mm": RELEASE_COAST_MARGIN_M * 1000,
            "max_extra_anticipatory_lead_ms": MAX_EXTRA_LEAD_S * 1000,
            "activation_gate": ACTIVATION_GATE,
            "control_dt_ms": teacher.CTRL * 1000,
            "physics_dt_ms": teacher.PHYS * 1000,
        },
        "results": {},
    }
    for policy in ("free", "human_proxy"):
        baseline = replay_policy(segment, replay, policy, "baseline")
        if baseline["verified_replay"].get("replay_exact") is not True:
            raise RuntimeError("baseline not verified")
        result["results"][policy] = {"baseline": baseline}
        for arm in ("release", "lead", "combined"):
            result["results"][policy][arm] = replay_policy(
                segment, replay, policy, arm
            )
    result["limitations"] = [
        "Baseline is verified against saved full-chart physical KeyDowns and scorer metrics.",
        "Release uses only current Motor pressed, position and velocity, hence is actionable.",
        "Lead/combined peek at FUTURE SAVED FINGERING ALLOCATIONS from replay.",
        "The anticipation cases are noncausal feasibility/upper-bound experiments, not an online Network evaluation.",
        "The original allocation choices are held fixed, even when intervention changes the Motor state.",
        "Only one specified release strategy and one activation-based anticipation rule are compared; no tuning sweep.",
        "Misses, extra physical KeyDowns, TooEarly, overload and X Accuracy must be considered together.",
        "An experimental Motor profile is not a calibrated human hand.",
    ]
    return result


def render_html(report: dict) -> str:
    rows = []
    for policy, arms in report["results"].items():
        for arm in ARMS:
            m = arms[arm]["metrics"]
            rows.append("<tr>" + "".join(
                f"<td>{escape(str(v) if v is not None else '—')}</td>"
                for v in (
                    policy, arm, m["hits"], m["misses"], m["physical_keydowns"],
                    m["too_early"], m["overloaded"], m["x_accuracy_percent"],
                    m["timing_mae_ms"], m["release_coast_axes_steps"],
                    m["anticipatory_events_armed"], m["future_allocation_leak"],
                )
            ) + "</tr>")
    return (
        "<!doctype html><html lang='ja'><meta charset='utf-8'>"
        "<title>Fingering release and lead controller comparison</title>"
        "<style>body{max-width:1350px;margin:24px auto;padding:0 16px;"
        "font:14px/1.55 system-ui;background:#0f1725;color:#f1f5fb}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{text-align:left;padding:8px;border-bottom:1px solid #33465f}"
        "th{background:#1c2b3d;position:sticky;top:0}</style>"
        f"<h1>解放と押下先行の4条件比較</h1><p>{escape(report['chart'])}</p>"
        "<table><thead><tr><th>運指</th><th>条件</th><th>Hits</th><th>Misses</th>"
        "<th>KeyDown数</th><th>TooEarly</th><th>Overload</th>"
        "<th>X Accuracy</th><th>Timing MAE</th><th>解放介入数</th>"
        "<th>先行選択イベント</th><th>未来運指の参照</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
        "<p>LeadとCombinedは未来の保存済み指選択を参照する非因果的な診断。"
        "実行可能なNetwork改良と混同しない。</p></html>\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chart", type=Path, required=True)
    p.add_argument("--replay-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        p.error(f"refusing to overwrite: {args.output_dir}")
    replay = json.loads(args.replay_json.read_text(encoding="utf-8"))
    report = analyze(replay, args.chart)
    html = render_html(report)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "release-lead-compare.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with (args.output_dir / "release-lead-compare.html").open("x", encoding="utf-8") as f:
        f.write(html)
    for policy, arms in report["results"].items():
        print(f"\n{policy}: baseline exact={arms['baseline']['verified_replay'].get('replay_exact')}")
        for name in ARMS:
            m = arms[name]["metrics"]
            print(f"  {name}: hits={m['hits']}, misses={m['misses']}, "
                  f"X={m['x_accuracy_percent']}, MAE={m['timing_mae_ms']}ms, "
                  f"TE={m['too_early']}, overload={m['overloaded']}, "
                  f"physical_downs={m['physical_keydowns']}, "
                  f"release_steps={m['release_coast_axes_steps']}, "
                  f"anticipatory_events={m['anticipatory_events_armed']}")
    print("viewer:", args.output_dir / "release-lead-compare.html")


if __name__ == "__main__":
    main()
