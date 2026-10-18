from __future__ import annotations

"""Causal early Fingering Network audit on the full bool chart.

Unlike the noncausal saved-allocation lead experiment, EVERY policy choice is
made by the frozen Network at the simulated current time, from current Motor
state and oracle-relative upcoming event times. No saved future fingering is
ever read for an experimental arm. Oracle timing remains privileged and this
is NOT a HUD-only Student test.

baseline: original decision timing and original Motor controller; must match
          every physical KeyDown and scorer metric from saved v0.1.1 replay.
early_decision: decide 25ms earlier, retain original 44ms action window.
causal_lead: decide earlier and extend the chosen ordinary axis' press window
             by up to 25ms from its *decision-time* negative activation.
causal_combined: causal_lead plus the current-state release-coast diagnostic.

Motor/Network source and checkpoint are never modified or retrained.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
from math import log1p
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_oracle_fingering_v02 as teacher
import audit_fingering_physical_reuse_counterfactual_v011 as fixed
import audit_fingering_release_lead_v011 as release_audit
import build_fingering_replay_viewer_v01 as viewer
import run_full_chart_fingering_replay_v02 as runner

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_chart_16k import CoupledContactRealChartEnv
from dmdod.motor.contact_16k import ContactAction
from dmdod.motor.keyboard import KeyEvent
from dmdod.training import fingering_network_v01 as network
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher


CHART = Path("data/DMDOD-v140-tuf-p11-p20/Train/04042 - bool - vo[ICE].adofai")
CHECKPOINT = Path("checkpoints/fingering/fingering_network_v01-4fa2ae1.pt")
SAVED_REPLAY = Path("artifacts/fingering_network_v011/bool-reuse-p18-full-r1/replay-v011.json")
ARMS = ("baseline", "early_decision", "causal_lead", "causal_combined")
POLICIES = (("free", 0.0), ("human_proxy", .5))
EXTRA_LEAD_S = .025
NEGATIVE_ACTIVATION_GATE = -.010


def activation_extra_s(activation: float, tau_s: float) -> float:
    """Fixed, bounded lead rule, derived from frozen activation dynamics."""
    if activation >= NEGATIVE_ACTIVATION_GATE:
        return 0.0
    if tau_s <= 0.0:
        raise ValueError("nonpositive activation time constant")
    return min(EXTRA_LEAD_S, tau_s * log1p(-activation))


def decision_extra_s(motor, allocation: tuple) -> float:
    """Only physical ordinary axes (1..4); no thumb/edge topology changes."""
    if len(allocation) != 1:
        return 0.0
    axis = allocation[0].axis
    if int(axis.rsplit("_", 1)[1]) > 4:
        return 0.0
    idx = teacher.AXES.index(axis)
    physical = motor.motor
    if physical.observe().pressed(axis):
        return 0.0
    state = physical.body.states[idx]
    return activation_extra_s(state.activation, physical.body.configs[idx].activation_tau_s)


def adaptive_action(motor, plans: tuple[teacher.PlanItem, ...],
                    now_s: float, early_by_event: dict[int, float],
                    *, enable_release: bool) -> tuple[ContactAction, int, int]:
    """Normal controller plus *causal* extra lead for already-selected axes.

    No future allocation or future Motor state is read. Positive early
    commands are only sent to ordinary axes that are currently not pressed.
    """
    base = teacher.action_for(motor, plans, now_s)
    commands = list(base.commands)
    obs = motor.motor.observe()
    extra_axes = set()
    for item in plans:
        extra = early_by_event.get(item.event_index, 0.0)
        if extra <= 0.0:
            continue
        if not (now_s + teacher.LEAD + 1e-9 < item.due_s
                <= now_s + teacher.LEAD + extra + 1e-9):
            continue
        if now_s > item.due_s + teacher.PREVIEW_TAIL:
            continue
        for finger in item.fingers:
            axis = finger.axis
            if int(axis.rsplit("_", 1)[1]) > 4 or obs.pressed(axis):
                continue
            idx = teacher.AXES.index(axis)
            if commands[idx] != 1.0:
                commands[idx] = 1.0
                extra_axes.add(axis)
    action = ContactAction(tuple(commands), base.thumb_modes, base.edge_enabled)
    released = 0
    if enable_release:
        action, released = release_audit.release_coast(motor, action)
    return action, len(extra_axes), released


def pending_decision_events(events, selected: set[int], cursor: int,
                            now: float, horizon: float):
    """Equivalent pending/due sets without scanning every event on idle frames.

    Network decisions always process a sorted prefix in event order.  Advancing
    cursor past selected or expired entries is therefore safe.  Construct the
    full upcoming lookahead only when the next event is within the decision
    horizon (preserves the original encode_context inputs and order).
    """
    while cursor < len(events) and (
        events[cursor].event_index in selected
        or events[cursor].episode_time_s + teacher.PREVIEW_TAIL < now
    ):
        cursor += 1
    if (cursor == len(events)
            or events[cursor].episode_time_s > now + horizon + 1e-9):
        return (), (), cursor
    upcoming = tuple(
        event for event in events[cursor:]
        if event.event_index not in selected
        and event.episode_time_s + teacher.PREVIEW_TAIL >= now
    )
    due = tuple(event for event in upcoming
                if event.episode_time_s <= now + horizon + 1e-9)
    return upcoming, due, cursor


def execute(segment, model, policy: str, alpha: float, arm: str,
            *, wall_timeout_s: float) -> dict:
    """Run a fresh policy from actual current observations at each decision."""
    if arm not in ARMS:
        raise ValueError(f"unexpected arm {arm}")
    env = CoupledContactRealChartEnv(
        segment, control_dt_s=teacher.CTRL, physics_dt_s=teacher.PHYS
    )
    env.reset()
    events = AutoplayTimingTeacher.from_segment(segment).events
    committed = {}
    selected = set()
    next_event_cursor = 0
    decisions = []
    early_by_event: dict[int, float] = {}
    actual_downs = []
    keydowns = []
    interventions = Counter()
    begun = time.monotonic()
    max_cycles = int((segment.duration_s + 2.0) / teacher.CTRL) + 200
    horizon = (teacher.LEAD if arm == "baseline"
               else teacher.LEAD + EXTRA_LEAD_S)

    for _ in range(max_cycles):
        if time.monotonic() - begun > wall_timeout_s:
            raise TimeoutError(f"{policy}/{arm} exceeded {wall_timeout_s}s")
        now = env.privileged_episode_time_s()
        committed = {
            ix: item for ix, item in committed.items()
            if item.due_s + teacher.PREVIEW_TAIL >= now
            and not all(env._used[o] for o in events[ix].target_ordinals)
        }
        upcoming, due, next_event_cursor = pending_decision_events(
            events, selected, next_event_cursor, now, horizon
        )

        for event in due:
            if event.event_index in selected:
                continue
            occupied = {axis for item in committed.values() for axis in item.axes}
            look = tuple(e for e in upcoming if e.event_index >= event.event_index)
            allocations = teacher.finger_options(event.chord_size)
            eligible = tuple(
                teacher.eligible(env.motor, occupied, value)[0]
                for value in allocations
            )
            context = network.encode_context(env.motor, look, now, occupied)
            selected_index, _, _ = network.choose_candidate_styled(
                model, context, allocations, eligible, alpha=alpha
            )
            allocation = allocations[selected_index] if selected_index is not None else ()
            item = teacher.PlanItem(event.event_index, event.episode_time_s, allocation)
            selected.add(event.event_index)
            if allocation:
                committed[event.event_index] = item
            extra = (decision_extra_s(env.motor, allocation)
                     if arm in ("causal_lead", "causal_combined") else 0.0)
            early_by_event[event.event_index] = extra
            decisions.append({
                "event_index": event.event_index,
                "due_ms": round(event.episode_time_s * 1000.0, 3),
                "decision_ms": round(now * 1000.0, 3),
                "decision_lead_ms": round((event.episode_time_s - now) * 1000.0, 3),
                "chosen": item.label if allocation else None,
                "axes": [finger.axis for finger in allocation],
                "eligible_options": sum(eligible),
                "extra_lead_ms": round(extra * 1000.0, 3),
                "ordinals": list(event.target_ordinals),
            })
            if selected_index is None:
                interventions["unassigned_events"] += 1
            if extra > 0:
                interventions["extra_lead_allocations"] += 1

        if arm in ("causal_lead", "causal_combined"):
            action, early_steps, release_steps = adaptive_action(
                env.motor, tuple(committed.values()), now, early_by_event,
                enable_release=(arm == "causal_combined")
            )
            interventions["actual_early_axis_steps"] += early_steps
            interventions["release_coast_axis_steps"] += release_steps
        else:
            action = teacher.action_for(env.motor, tuple(committed.values()), now)
        step = env.step(action)
        for event in env.last_events:
            if event.event is KeyEvent.DOWN:
                actual_downs.append((event.time_s * 1000.0, event.key))
                keydowns.append({
                    "axis": viewer._physical_axis(event.key),
                    "contact": event.key,
                    "time_ms": round(event.time_s * 1000.0, 3),
                })
        if step.done:
            break
    else:
        raise RuntimeError(f"{policy}/{arm} exceeded physical step budget")

    stats = runner._summarize(env)
    outcomes = {}
    for row in decisions:
        used = [env._used[x] for x in row["ordinals"]]
        missed = [env._missed[x] for x in row["ordinals"]]
        outcomes[row["event_index"]] = ("Hit" if all(used)
                else "Miss" if any(missed) else "Unresolved")
        row["outcome"] = outcomes[row["event_index"]]
        row.pop("ordinals")
    return {
        "stats": stats,
        "decisions": decisions,
        "keydowns": keydowns,
        "actual_downs": actual_downs,
        "interventions": dict(interventions),
        "termination_reason": ("overload" if stats["overloaded"]
                               else "miss" if stats["misses"]
                               else "chart_complete" if stats["hits"] == stats["targets"]
                               else "incomplete"),
        "replay_end_s": round(env.privileged_episode_time_s(), 6),
        "elapsed_wall_s": round(time.monotonic() - begun, 3),
        "env": env,
    }


def verify_baseline_choices(result: dict, saved_replay: dict, policy: str) -> None:
    """Protect causal experiments against inference/selection drift as well as physics drift."""
    saved = [ev["decisions"][policy] for ev in saved_replay["events"]
             if ev.get("decisions", {}).get(policy) is not None]
    actual = result["decisions"]
    if len(actual) != len(saved):
        raise RuntimeError(f"{policy}: baseline decision count drift: "
                           f"{len(actual)} != {len(saved)}")
    for old, new in zip(saved, actual):
        if (int(old["event_index"]) != int(new["event_index"])
                or old["chosen"] != new["chosen"]
                or abs(float(old["decision_s"]) * 1000.0 - new["decision_ms"]) > .002):
            raise RuntimeError(f"{policy}: baseline Network decision drift at "
                               f"event {new['event_index']}")


def summarize_arm(result: dict, baseline: dict, arm: str) -> dict:
    baseline_choices = {r["event_index"]: r["chosen"] for r in baseline["decisions"]}
    choices = result["decisions"]
    comparable = [r for r in choices if r["event_index"] in baseline_choices]
    same = sum(r["chosen"] == baseline_choices[r["event_index"]] for r in comparable)
    lead_values = [r["decision_lead_ms"] for r in choices]
    extras = [r["extra_lead_ms"] for r in choices if r["extra_lead_ms"] > 0]
    return {
        "arm": arm,
        "stats": result["stats"],
        "termination_reason": result["termination_reason"],
        "elapsed_wall_s": result["elapsed_wall_s"],
        "decisions": len(choices),
        "unassigned": sum(r["chosen"] is None for r in choices),
        "common_decisions_vs_baseline": len(comparable),
        "changed_allocations_vs_baseline": len(comparable) - same,
        "mean_decision_lead_ms": round(sum(lead_values) / len(lead_values), 3) if lead_values else None,
        "mean_extra_lead_ms_when_active": round(sum(extras) / len(extras), 3) if extras else None,
        "extra_lead_events": len(extras),
        "interventions": result["interventions"],
        "keydown_count": len(result["keydowns"]),
        "future_saved_fingering_consulted": False,
        "teacher_counterfactual_evaluation": False,
        "event_decisions": choices,
    }


def analyze(chart: Path, checkpoint: Path, saved: Path,
            output_dir: Path, *, wall_timeout_s: float) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    replay = json.loads(saved.read_text(encoding="utf-8"))
    if replay.get("format") != "dmdod-fingering-network-v0.1.1-human-proxy-comparison":
        raise ValueError("expected v0.1.1 Free/Human-Proxy replay JSON")
    if viewer.file_sha256(chart) != replay["chart_sha256"]:
        raise ValueError("chart SHA256 mismatch")
    if viewer.file_sha256(checkpoint) != replay["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA256 mismatch")
    compiled = compile_adofai(load_adofai(chart))
    segment = build_playable_segment(compiled)
    model = runner.load_model(checkpoint)
    report = {
        "format": "dmdod-fingering-v0.2-causal-early-audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "chart": chart.name,
        "chart_sha256": replay["chart_sha256"],
        "checkpoint_sha256": replay["checkpoint_sha256"],
        "decision_horizon_ms": (teacher.LEAD + EXTRA_LEAD_S) * 1000.0,
        "original_decision_horizon_ms": teacher.LEAD * 1000.0,
        "physics_dt_ms": teacher.PHYS * 1000.0,
        "control_dt_ms": teacher.CTRL * 1000.0,
        "training_performed": False,
        "physical_parameters_changed": False,
        "oracle_timestamps_used": True,
        "saved_future_allocations_consulted": False,
        "policies": {},
    }
    for policy, alpha in POLICIES:
        baseline = execute(segment, model, policy, alpha, "baseline",
                           wall_timeout_s=wall_timeout_s)
        fixed._verify_run(
            baseline["env"], replay["policies"][policy],
            baseline["actual_downs"], policy
        )
        verify_baseline_choices(baseline, replay, policy)
        rows = {"baseline": summarize_arm(baseline, baseline, "baseline")}
        for arm in ARMS[1:]:
            attempt = execute(segment, model, policy, alpha, arm,
                              wall_timeout_s=wall_timeout_s)
            rows[arm] = summarize_arm(attempt, baseline, arm)
        report["policies"][policy] = rows
    report["limitations"] = [
        "AutoPlay target timestamps are privileged; this does not validate a HUD-only Timing System.",
        "All experimental finger choices are made with the frozen Network at the current Motor state.",
        "No future saved allocation or physically ranked teacher candidates are read by experimental policies.",
        "Earlier Network inference changes its input distribution and may change hand/finger choices.",
        "Eligibility and reservation constraints are unchanged, and may reduce available early choices.",
        "Early command extension is limited to ordinary physical fingers; thumb/side remain unchanged.",
        "Fixed activation-derived extra lead is calculated once at decision time, not optimized.",
        "Release-coast uses only current Motor state; the physical model is experimental.",
        "Any early or extra KeyDown can cause TooEarly, Miss or Overload; compare all safety metrics.",
        "Full-chart results use a single Train chart; do not generalize without independent evaluation.",
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "causal-early-summary.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with (output_dir / "causal-early-summary.html").open("x", encoding="utf-8") as f:
        f.write(render_html(report))
    return report


def render_html(report: dict) -> str:
    body = []
    for policy, arms in report["policies"].items():
        for arm in ARMS:
            row = arms[arm]
            m = row["stats"]
            body.append("<tr>" + "".join(
                f"<td>{escape(str(v) if v is not None else '—')}</td>"
                for v in (
                    policy, arm, m["hits"], m["misses"],
                    m["x_accuracy_percent"], m["timing_mae_ms"],
                    m["too_early"], m["overloaded"],
                    m["physical_keydowns"], row["unassigned"],
                    row["changed_allocations_vs_baseline"],
                    row["mean_decision_lead_ms"], row["extra_lead_events"],
                    row["interventions"].get("actual_early_axis_steps", 0),
                    row["interventions"].get("release_coast_axis_steps", 0),
                    row["termination_reason"],
                )
            ) + "</tr>")
    return (
        "<!doctype html><html lang='ja'><meta charset='utf-8'>"
        "<title>Causal Early Fingering v0.2</title>"
        "<style>body{max-width:1600px;margin:25px auto;padding:0 16px;"
        "font:14px/1.6 system-ui;background:#101723;color:#e9f0fa}"
        "table{border-collapse:collapse;width:100%}"
        "td,th{padding:9px;text-align:left;border-bottom:1px solid #43526c}"
        "th{position:sticky;top:0;background:#21314a}"
        "p{max-width:950px}</style>"
        "<h1>Causal Early Fingering v0.2 — 4条件比較</h1>"
        f"<p>{escape(report['chart'])}｜保存済みの未来運指は不使用。"
        "各条件で現在のMotor状態を入力し、Networkが運指を新規選択。"
        "ただしTimingはAutoPlay Oracleを使用。</p>"
        "<table><thead><tr><th>方式</th><th>条件</th><th>Hit</th>"
        "<th>Miss</th><th>X Accuracy</th><th>Timing MAE(ms)</th>"
        "<th>TooEarly</th><th>Overload</th><th>KeyDown</th>"
        "<th>運指未割当</th><th>旧運指との差</th><th>決定先行(ms)</th>"
        "<th>追加Lead選択</th><th>追加押下Step</th>"
        "<th>解放調整Step</th><th>終了理由</th></tr></thead><tbody>"
        + "".join(body) + "</tbody></table>"
        "<p>Baselineのみ元のKeyDown・判定・Network運指と厳密照合。"
        "その他の条件でHitやKeyDownが一致する保証はない。"
        "単一Train譜面の研究用診断であり、HUD-only Studentの評価ではない。</p>"
        "</html>\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chart", type=Path, default=CHART)
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    p.add_argument("--saved-replay", type=Path, default=SAVED_REPLAY)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--wall-timeout-s", type=float, default=600.0)
    args = p.parse_args()
    if args.wall_timeout_s <= 0:
        p.error("--wall-timeout-s must be positive")
    result = analyze(args.chart, args.checkpoint, args.saved_replay,
                     args.output_dir, wall_timeout_s=args.wall_timeout_s)
    for policy, arms in result["policies"].items():
        print(f"\n{policy}: original baseline physically verified")
        for arm in ARMS:
            row = arms[arm]
            m = row["stats"]
            xacc = ("n/a" if m["x_accuracy_percent"] is None
                    else f"{m['x_accuracy_percent']:.4f}%")
            mae = ("n/a" if m["timing_mae_ms"] is None
                   else f"{m['timing_mae_ms']:.3f}ms")
            print(f"  {arm}: H={m['hits']}/{m['targets']}, miss={m['misses']}, "
                  f"X={xacc}, MAE={mae}, "
                  f"TE={m['too_early']}, overload={m['overloaded']}, "
                  f"keydowns={m['physical_keydowns']}, "
                  f"choice_changed={row['changed_allocations_vs_baseline']}, "
                  f"unassigned={row['unassigned']}, "
                  f"extra_press_steps={row['interventions'].get('actual_early_axis_steps', 0)}")
    print("report:", args.output_dir / "causal-early-summary.json")
    print("viewer:", args.output_dir / "causal-early-summary.html")


if __name__ == "__main__":
    main()
