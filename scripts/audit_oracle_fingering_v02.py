from __future__ import annotations

"""Oracle-assisted 12-axis/16-contact fingering teacher, with *physical* 3-event lookahead.

All exact target timestamps are evaluator/teacher-only. No Student model is trained.
Each candidate sequence is executed on a deep-copied Coupled16ContactMotor with
10ms commands and 1ms physics. The selected first event alone is committed.
"""

import argparse
import copy
import csv
from dataclasses import dataclass
from html import escape
import hashlib
import itertools
import json
from math import ceil
from pathlib import Path

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_chart_16k import CoupledContactRealChartEnv
from dmdod.motor.contact_16k import ContactAction, Coupled16ContactMotor, PHYSICAL_AXES, ThumbMode
from dmdod.motor.keyboard import KeyEvent
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher

CTRL = 0.010
PHYS = 0.001
LEAD = 0.044
LOOKAHEAD = 0.155
PREVIEW_TAIL = 0.035
MAX_FOLLOWUPS = 3
AXES = PHYSICAL_AXES


@dataclass(frozen=True)
class Finger:
    axis: str
    mode: str = "single"  # single / tip / flat / edge

    @property
    def contact_count(self) -> int:
        return 2 if self.mode in ("flat", "edge") else 1

    @property
    def logical_keys(self) -> tuple[str, ...]:
        hand, depth_str = self.axis.split("_")
        depth = int(depth_str)
        if self.mode == "flat":
            return f"{hand}_5", f"{hand}_6"
        if self.mode == "edge":
            return f"{hand}_7", f"{hand}_8"
        return (f"{hand}_{5 if self.mode == 'tip' else depth}",)

    @property
    def label(self) -> str:
        return self.axis + ("/" + self.mode if self.mode != "single" else "")


@dataclass(frozen=True)
class PlanItem:
    event_index: int
    due_s: float
    fingers: tuple[Finger, ...]

    @property
    def label(self) -> str:
        return " + ".join(f.label for f in self.fingers)

    @property
    def axes(self) -> frozenset[str]:
        return frozenset(f.axis for f in self.fingers)


def finger_options(chord_size: int) -> tuple[tuple[Finger, ...], ...]:
    singles = tuple(Finger(axis, "tip" if axis.endswith("_5") else "single")
                    for axis in AXES if not axis.endswith("_6"))
    doubles = tuple(Finger(axis, "flat" if axis.endswith("_5") else "edge")
                    for axis in AXES if axis.endswith(("_5", "_6")))
    if chord_size == 1:
        return tuple((f,) for f in singles)
    if chord_size == 2:
        return tuple((f,) for f in doubles) + tuple(itertools.combinations(singles, 2))
    if 3 <= chord_size <= 8:
        return tuple(itertools.combinations(singles, chord_size))[:60]
    return ()


def _safe_topology(motor: Coupled16ContactMotor, finger: Finger) -> bool:
    hand, depth = finger.axis.split("_")
    depth = int(depth)
    if depth < 5:
        return True
    i = 0 if hand == "left" else 1
    if depth == 5:
        wanted = ThumbMode.FLAT if finger.mode == "flat" else ThumbMode.TIP
        current = motor._modes[i]
    else:
        wanted = True
        current = motor._edge[i]
    if wanted == current:
        return True
    state = motor.motor.observe()
    idx = state.index(finger.axis)
    pos = state.positions_m[idx]
    vel = state.velocities_m_s[idx]
    return (not state.pressed_flags[idx]
            and pos <= motor.motor.keyboard.config.reset_m
            and not (pos > 0 and vel > .0001))


def eligible(motor: Coupled16ContactMotor, occupied: set[str],
             allocation: tuple[Finger, ...]) -> tuple[bool, str]:
    if len(set(f.axis for f in allocation)) != len(allocation):
        return False, "duplicate_axis"
    state = motor.motor.observe()
    for finger in allocation:
        if finger.axis in occupied:
            return False, "reserved"
        if state.pressed(finger.axis):
            return False, "pressed"
        if not _safe_topology(motor, finger):
            return False, "topology_switch_unsafe"
    # A thumb or side may have multiple logical contacts. Count is explicit.
    contacts = [c for f in allocation for c in f.logical_keys]
    if len(contacts) != len(set(contacts)):
        return False, "duplicate_contact"
    return True, "eligible"


def followup_options(motor: Coupled16ContactMotor, occupied: set[str],
                     chord_size: int, previous: tuple[Finger, ...], *, limit=MAX_FOLLOWUPS):
    # Diverse deterministic continuations, not a hidden fingering label.
    choices = []
    last_hands = {f.axis.split("_")[0] for f in previous}
    for option in finger_options(chord_size):
        ok, _ = eligible(motor, occupied, option)
        if ok:
            hands = {f.axis.split("_")[0] for f in option}
            score = (bool(hands & last_hands),
                     sum(int(f.axis.split("_")[1]) for f in option),
                     tuple(f.label for f in option))
            choices.append((score, option))
    choices.sort(key=lambda x: x[0])
    picked = []
    for _, o in choices:
        if len(picked) >= limit:
            break
        if not any(o == other for other in picked):
            picked.append(o)
    return tuple(picked)


def action_for(motor: Coupled16ContactMotor, plans: tuple[PlanItem, ...], now_s: float,
               *, fired_axes: frozenset[str] = frozenset()):
    obs = motor.motor.observe()
    commands = {axis: -1.0 if obs.pressed(axis) else 0.0 for axis in AXES}
    modes = list(motor._modes)
    edge = list(motor._edge)
    for item in plans:
        if now_s + LEAD + 1e-9 < item.due_s or now_s > item.due_s + PREVIEW_TAIL:
            continue
        for f in item.fingers:
            if obs.pressed(f.axis) or f.axis in fired_axes:
                continue
            hand, depth = f.axis.split("_")
            i = 0 if hand == "left" else 1
            if int(depth) == 5:
                wanted = ThumbMode.FLAT if f.mode == "flat" else ThumbMode.TIP
                if wanted != modes[i] and not _safe_topology(motor, f):
                    continue
                modes[i] = wanted
            if int(depth) == 6:
                if not edge[i] and not _safe_topology(motor, f):
                    continue
                edge[i] = True
            commands[f.axis] = 1.0
    return ContactAction(tuple(commands[a] for a in AXES), tuple(modes), tuple(edge))


def simulated_sequence(motor: Coupled16ContactMotor, plans: tuple[PlanItem, ...],
                        *, until_s: float):
    """Evaluate physical KeyDown and preservation of upcoming plan from identical state."""
    branch = copy.deepcopy(motor)
    logs = []
    fired = set()
    starts = branch.motor.time_s
    cycles = max(0, ceil((until_s - starts) / CTRL - 1e-9))
    for _ in range(min(cycles, 40)):
        now = branch.motor.time_s
        action = action_for(branch, plans, now, fired_axes=frozenset(fired))
        result = branch.step(action)
        for event in result.logical_events:
            if event.event is KeyEvent.DOWN:
                logs.append((event.time_s, event.key))
                hand, logical_depth = event.key.split("_")
                depth = int(logical_depth)
                fired.add(f"{hand}_{6 if depth in (7, 8) else 5 if depth in (5, 6) else depth}")

    # Assign a logical down to a planned event only if that event requested that contact.
    consumed = set()
    deviations = []
    missing = 0
    for item in plans:
        errors = []
        contacts = [contact for f in item.fingers for contact in f.logical_keys]
        for contact in contacts:
            candidates = [(abs(t - item.due_s), i, t) for i, (t, k) in enumerate(logs)
                          if k == contact and i not in consumed
                          and t >= starts - 1e-8]
            if candidates:
                delta, ix, tm = min(candidates)
                consumed.add(ix)
                errors.append((tm - item.due_s) * 1000)
                if delta > .030:
                    missing += 1
            else:
                missing += 1
        deviations.append({"event_index": item.event_index, "fingering": item.label,
                           "due_ms": round(item.due_s * 1000, 3),
                           "errors_ms": [round(e, 3) for e in errors],
                           "contacts": contacts,
                           "missing": max(0, len(contacts) - len(errors))})
    extra = len(logs) - len(consumed)
    errors = [abs(v) for record in deviations for v in record["errors_ms"]]
    # Hard penalties keep a missed note worse than a few ms of timing improvement.
    cost = 150.0 * missing + 120.0 * extra + sum(errors)
    return {"cost": round(cost, 3), "missing": missing, "extra": extra,
            "total_error_ms": round(sum(errors), 3),
            "per_event": deviations,
            "physical_downs": [{"time_ms": round(t * 1000, 3), "contact": k} for t, k in logs]}


def rank_first_choices(motor: Coupled16ContactMotor, committed: tuple[PlanItem, ...],
                       upcoming: tuple[object, ...], *, max_depth=3):
    if not upcoming:
        return []
    first = upcoming[0]
    look = [first]
    for event in upcoming[1:max_depth]:
        if event.episode_time_s - first.episode_time_s <= LOOKAHEAD + 1e-9:
            look.append(event)
        else:
            break
    occupied = {a for item in committed for a in item.axes}
    results = []
    for option in finger_options(first.chord_size):
        label = " + ".join(f.label for f in option)
        ok, reason = eligible(motor, occupied, option)
        if not ok:
            results.append({"finger": label, "eligible": False, "reason": reason,
                            "score": None, "best_continuation": []})
            continue
        item = PlanItem(first.event_index, first.episode_time_s, option)
        partial = (item,)
        candidates = [(partial,)]
        # Branching on the first future event. Subsequent event continuation is
        # *also simulated* and scored; commit only the first action allocation.
        if len(look) >= 2:
            second = look[1]
            second_options = followup_options(motor, occupied | item.axes,
                                              second.chord_size, option)
            candidates = [((item, PlanItem(second.event_index, second.episode_time_s, f)),)
                          for f in second_options]
            if not candidates:
                candidates = [((item,),)]
        evaluated = []
        for candidate_wrap in candidates:
            trial = list(candidate_wrap[0])
            if len(look) >= 3 and len(trial) > 1:
                third = look[2]
                booked = occupied | {axis for p in trial for axis in p.axes}
                last = trial[-1].fingers
                alternatives = followup_options(motor, booked, third.chord_size, last, limit=1)
                if alternatives:
                    trial.append(PlanItem(third.event_index, third.episode_time_s, alternatives[0]))
            complete = committed + tuple(trial)
            evaluation = simulated_sequence(motor, complete,
                                           until_s=max(p.due_s for p in complete) + PREVIEW_TAIL)
            if len(trial) < len(look):
                # Missing continuation is a legitimate planning failure, not free abstention.
                evaluation["cost"] += 150 * sum(e.chord_size for e in look[len(trial):])
                evaluation["incomplete_continuation"] = True
            evaluated.append((evaluation["cost"], tuple(p.label for p in trial), evaluation, trial))
        score, _, best, trial = min(evaluated, key=lambda x: (x[0], x[1]))
        results.append({"finger": label, "eligible": True, "reason": "physical_lookahead",
                        "score": round(score, 3), "missing": best["missing"],
                        "extra": best["extra"], "error_ms": best["total_error_ms"],
                        "best_continuation": [p.label for p in trial[1:]],
                        "lookahead_depth": len(trial), "physical_preview": best,
                        "allocation": option})
    for rank, c in enumerate(sorted((r for r in results if r["eligible"]),
                                    key=lambda r: (r["score"], r["finger"])), 1):
        c["rank"] = rank
    return results


def run_teacher(segment, *, focus_from_s: float = 0.0, max_depth: int = 3):
    if not (2 <= max_depth <= 3):
        raise ValueError("max_depth must be 2 or 3")
    env = CoupledContactRealChartEnv(segment, control_dt_s=CTRL, physics_dt_s=PHYS)
    obs = env.reset()
    events = AutoplayTimingTeacher.from_segment(segment).events
    committed: dict[int, PlanItem] = {}
    rows = []
    selected_by_event = {}
    all_downs = []
    last_chosen = None
    max_steps = int((segment.duration_s + 2) / CTRL) + 200
    for _ in range(max_steps):
        now = env.privileged_episode_time_s()
        committed = {idx: item for idx, item in committed.items()
                     if item.due_s + PREVIEW_TAIL >= now
                     and not all(env._used[o] for o in events[idx].target_ordinals)}
        upcoming = [e for e in events if e.event_index not in selected_by_event
                    and e.episode_time_s + PREVIEW_TAIL >= now]
        due_now = [e for e in upcoming if e.episode_time_s <= now + LEAD + 1e-9]
        for event in due_now:
            if event.event_index in selected_by_event:
                continue
            if now < focus_from_s:
                # Explicitly fixed center-first prefix for reaching an in-chart physical state.
                occupied = {x for item in committed.values() for x in item.axes}
                choices = [a for a in finger_options(event.chord_size)
                           if eligible(env.motor, occupied, a)[0]]
                opts = sorted(choices, key=lambda v: (sum(int(f.axis.split("_")[1]) for f in v),
                                                             tuple(f.label for f in v)))
                options = () if not opts else opts[0]
                info = []
                basis = "fixed_center_first_prefix"
                depth = 1
                continuation = []
            else:
                look = tuple(e for e in upcoming if e.event_index >= event.event_index)
                evaluated = rank_first_choices(env.motor, tuple(committed.values()),
                                               look, max_depth=max_depth)
                viable = [x for x in evaluated if x["eligible"]]
                chosen = min(viable, key=lambda x: (x["score"], x["finger"])) if viable else None
                options = chosen["allocation"] if chosen else ()
                basis = "physical_lookahead" if chosen else "no_feasible_axis"
                depth = chosen.get("lookahead_depth", 0) if chosen else 0
                continuation = chosen.get("best_continuation", []) if chosen else []
                info = [{k: v for k, v in c.items() if k != "allocation"} for c in evaluated]
            item = PlanItem(event.event_index, event.episode_time_s, options)
            selected_by_event[event.event_index] = item
            if options:
                committed[event.event_index] = item
            if now >= focus_from_s:
                rows.append({"event_index": event.event_index, "ordinals": event.target_ordinals,
                             "due_ms": round(event.episode_time_s * 1000, 3),
                             "decision_ms": round(now * 1000, 3),
                             "chosen": item.label if options else None,
                             "fingers": [{"axis": f.axis, "mode": f.mode,
                                          "contacts": list(f.logical_keys)} for f in options],
                             "reason": basis, "depth": depth,
                             "continuation": continuation, "candidates": info})
        action = action_for(env.motor, tuple(committed.values()), now)
        step = env.step(action)
        for evt in env.last_events:
            if evt.event is KeyEvent.DOWN:
                all_downs.append({"time_ms": round(evt.time_s * 1000, 3), "contact": evt.key})
        obs = step.observation
        if step.done:
            break
    else:
        raise RuntimeError("Physical teacher exceeded step budget")
    for row in rows:
        relevant = [env._used[o] for o in row["ordinals"]]
        missed = [env._missed[o] for o in row["ordinals"]]
        row["outcome"] = "hit" if all(relevant) else "miss" if any(missed) else "unresolved"
        # Posthoc only: nearest actual contact KeyDown to target time is diagnostic.
        contacts = [c for f in row["fingers"] for c in f["contacts"]]
        matches = [d for d in all_downs if d["contact"] in contacts and
                   abs(d["time_ms"] - row["due_ms"]) < 35]
        row["keydowns"] = matches
        row["actual_error_ms"] = (round(min((d["time_ms"] - row["due_ms"] for d in matches),
                                             key=abs), 3) if matches else None)
    return {"stats": {"hits": env.stats.hits, "misses": env.stats.misses,
                      "too_early": env.stats.too_early_presses, "overloaded": env.stats.overloaded,
                      "x_accuracy_percent": env.stats.x_accuracy_percent,
                      "targets": env.stats.targets},
            "focus_from_s": focus_from_s, "focus_hits": sum(r["outcome"] == "hit" for r in rows),
            "focus_events": len(rows), "decisions": rows,
            "keydown_count": len(all_downs)}


def report_chart(chart_path: Path, *, end_s: float, focus_from_s: float = 0.0,
                 max_depth: int = 3):
    if end_s <= 0 or focus_from_s < 0 or end_s <= focus_from_s:
        raise ValueError("invalid chart window")
    data = chart_path.read_bytes()
    chart = compile_adofai(load_adofai(chart_path))
    segment = build_playable_segment(chart, end_s=min(end_s, chart.duration_s))
    oracle = AutoplayTimingTeacher.from_segment(segment)
    outcome = run_teacher(segment, focus_from_s=focus_from_s, max_depth=max_depth)
    return {"format": "dmdod-oracle-fingering-physical-v0.2",
            "oracle_assisted": True, "student_trained": False,
            "physical_axes": 12, "logical_contacts": 16,
            "chart": chart_path.name, "chart_sha256": hashlib.sha256(data).hexdigest(),
            "segment_end_s": segment.duration_s, "oracle_targets": oracle.total_targets,
            "lookahead_s": LOOKAHEAD, "max_depth": max_depth,
            "control_s": CTRL, "physics_s": PHYS,
            "teacher": outcome,
            "limitations": [
                "Only first choice is committed; future lookahead choices are counterfactual.",
                "Bounded diverse 3-choice continuation beam; not globally optimal.",
                "Actual chart scorer is unchanged and its Multipress semantics are incomplete.",
                "Physical profile is illustrative, not measured human finger anatomy.",
                "Oracle times are teacher-only and never constitute HUD-only inference.",
                "No Student training, original checkpoints, Stage A FAIL or validation modifications.",
            ]}


def render_html(report: dict) -> str:
    rows = report["teacher"]["decisions"]
    lane_rows = []
    min_ms = report["teacher"]["focus_from_s"] * 1000
    max_ms = report["segment_end_s"] * 1000
    for axis in AXES:
        dots = []
        for row in rows:
            if not any(f["axis"] == axis for f in row["fingers"]):
                continue
            relative = ((row["due_ms"] - min_ms) / max(1.0, max_ms - min_ms))*100
            dots.append(f'<a class="dot" style="left:{relative:.3f}%" href="#ev{row["event_index"]}" title="#{row["event_index"]} {row["chosen"]}"></a>')
        lane_rows.append(f'<div class="lane"><span>{escape(axis)}</span><div class="track">{"".join(dots)}</div></div>')
    body = []
    for d in rows:
        candidates = []
        for c in d["candidates"]:
            candidates.append('<tr' + (' class="chosen"' if c['finger'] == d['chosen'] else '') + '>'
                              +f'<td>{escape(c["finger"])}</td><td>{c.get("rank", "—")}</td>'
                              +f'<td>{c.get("score") if c.get("score") is not None else "—"}</td>'
                              +f'<td>{escape(c["reason"])}</td>'
                              +f'<td>{escape(" → ".join(c["best_continuation"]))}</td></tr>')
        body.append(f'<tr id="ev{d["event_index"]}"><td>#{d["event_index"]}</td>'
                    f'<td>{d["due_ms"]}</td><td>{escape(d["chosen"] or "未割当")}</td>'
                    f'<td>{d["depth"]}</td><td>{d.get("actual_error_ms") if d.get("actual_error_ms") is not None else "—"}</td><td>{escape(d["outcome"])}</td>'
                    f'<td><details><summary>候補比較 ({len(d["candidates"])})</summary>'
                    '<table><tr><th>指</th><th>順位</th><th>cost</th><th>理由</th><th>2・3打目の想定</th></tr>'
                    + ''.join(candidates)+'</table></details></td></tr>')
    metrics = report['teacher']['stats']
    return '''<!DOCTYPE html><html lang="ja"><meta charset="utf-8"><title>DMDOD Fingering Teacher v0.2</title>
<style>body{font:14px/1.5 system-ui,sans-serif;max-width:1200px;padding:1rem;margin:auto;color:#e5e7eb;background:#111827}
a{color:#93c5fd}h1{font-size:1.5rem}small{color:#9ca3af}.stat{display:inline-block;padding:.7rem;border:1px solid #374151;border-radius:8px;margin:.3rem}.lane{display:grid;grid-template-columns:100px 1fr;gap:8px;align-items:center}.track{height:20px;background:#374151;position:relative;margin:2px}.dot{position:absolute;top:3px;height:14px;width:7px;background:#fbbf24;border-radius:4px;transform:translateX(-50%)}
table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #374151;padding:6px;text-align:left}tr:target,.chosen{background:#25465c}details{max-width:680px}summary{cursor:pointer}div.scroll{overflow:auto}p.notice{border-left:3px solid #fbbf24;padding-left:12px}</style>
<h1>DMDOD Fingering Teacher v0.2 — 12軸・16接点</h1>'''+f'<p>{escape(report["chart"])} / {report["segment_end_s"]:.2f}s</p>' + f'<div class="stat">対象区間 {report["teacher"]["focus_hits"]}/{report["teacher"]["focus_events"]} Hit</div>' + ''.join(f'<span class="stat">{escape(k)}: {v}</span>' for k,v in metrics.items()) + '''<p class="notice">AutoPlay正解時刻を知る物理教師。Fingering Networkではない。先読みは最大3打・限られた候補のみ。指の採用理由を全て記録。</p>
<h2>物理軸別タイムライン</h2>''' + ''.join(lane_rows) + '<h2>採用された指と候補</h2><div class="scroll"><table><tr><th>イベント</th><th>due(ms)</th><th>採用指</th><th>先読み打数</th><th>実誤差(ms)</th><th>実判定</th><th>他の候補</th></tr>'+''.join(body)+'</table></div></html>'


def save_report(report, output: Path):
    html = output.with_suffix('.html')
    csv_path = output.with_suffix('.csv')
    for p in (output, html, csv_path):
        if p.exists():
            raise FileExistsError(f'write-once output exists: {p}')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    html.write_text(render_html(report), encoding='utf-8')
    with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['event_index','due_ms','chosen','depth','actual_error_ms','outcome','reason','continuation'])
        writer.writeheader()
        for row in report['teacher']['decisions']:
            writer.writerow({k: ' → '.join(row[k]) if k == 'continuation' else row.get(k)
                             for k in writer.fieldnames})
    return html, csv_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('chart', type=Path)
    p.add_argument('--end-s', type=float, default=15)
    p.add_argument('--focus-from-s', type=float, default=0)
    p.add_argument('--depth', type=int, choices=(2,3), default=3)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = report_chart(args.chart, end_s=args.end_s,
                          focus_from_s=args.focus_from_s, max_depth=args.depth)
    html, csv_path = save_report(report, args.output)
    print('stats:', report['teacher']['stats'])
    print('events:', len(report['teacher']['decisions']))
    print('report:', args.output, 'html:', html, 'csv:', csv_path)


if __name__ == '__main__':
    main()