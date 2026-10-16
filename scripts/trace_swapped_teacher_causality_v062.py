"""Stage C v0.6.2: reproduce note #614–615 swap forecast divergence.

Read-only diagnostic. The only alternate reservation is left_5 -> left_6
at 58.400s. Compare (a) fixed command, (b) recorded live controls,
(c) one-tick ablation, and (d) the full replanning shadow teacher.
No Student training; no changes to Motor, scorer, or frozen teacher.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.envs.contact_chart_16k import CoupledContactRealChartEnv
from dmdod.motor.contact_16k import ContactAction, PHYSICAL_AXES
from dmdod.motor.keyboard import KeyEvent
import compare_nerfed_tie_teacher as tie

AXIS = "left_6"
SOURCE = "left_5"
ORDINALS = (614, 615)
START_S = 58.4
FRAMES = 15


def rows(events):
    return [(round(e.time_s * 1000, 6), e.key, e.event.value) for e in events]


def replay(snapshot, actions, *, fixed=None, ablate=False):
    env = copy.deepcopy(snapshot)
    result = []
    for i, a in enumerate(actions):
        command = fixed if fixed is not None else a
        if ablate and i == 1:
            values = list(command.commands)
            values[PHYSICAL_AXES.index(AXIS)] = 1.
            command = ContactAction(tuple(values), command.thumb_modes, command.edge_enabled)
        result.extend(rows(env.step(command).logical_events))
    return result


def first_down(events):
    for at_ms, name, event in events:
        if name == "left_7" and event == KeyEvent.DOWN.value:
            return at_ms
    return None


def diagnose(chart: Path):
    parsed = compile_adofai(parse_adofai_text(chart.read_text(encoding="utf-8-sig")))
    segment = build_playable_segment(parsed, end_s=58.8)

    class SwappedTeacher(tie.TieAwareJointTeacher):
        def __init__(self):
            super().__init__(8, None)
            self.swap_time = None
            self.snapshot = None
            self.first_action = None
            self.recorded = []
            self.events = []

        def action(self, observed, now_s, remaining):
            before = {axis: tuple(plan.tokens) for axis, plan in self.reservations.items()}
            command = super().action(observed, now_s, remaining)
            created = [axis for axis, plan in self.reservations.items()
                       if before.get(axis) != tuple(plan.tokens)]
            if (self.swap_time is None and now_s >= START_S - 1e-8
                and SOURCE in created and AXIS not in self.reservations
                and tuple(self.reservations[SOURCE].tokens) == ORDINALS):
                self.snapshot = copy.deepcopy(self.motor)
                plan = self.reservations.pop(SOURCE)
                plan.key = AXIS
                self.reservations[AXIS] = plan
                values = list(command.commands)
                values[PHYSICAL_AXES.index(SOURCE)] = -1.
                values[PHYSICAL_AXES.index(AXIS)] = 1.
                command = ContactAction(tuple(values),
                                        command.thumb_modes, command.edge_enabled)
                self.swap_time = now_s
                self.first_action = command
            if self.swap_time is not None and len(self.recorded) < FRAMES:
                self.recorded.append(command)
            return command

    teacher = SwappedTeacher()
    env = CoupledContactRealChartEnv(segment, control_dt_s=.010, physics_dt_s=.001)
    observed = env.reset()
    teacher.motor = env.motor
    shadow = None
    for _ in range(int((segment.duration_s + 1) / .010) + 4):
        target = env.privileged_next_target()
        remaining = (tuple((int(t.ordinal), float(t.episode_time_s))
                           for t in segment.targets[int(target.ordinal):])
                     if target is not None else ())
        command = teacher.action(observed, env.privileged_episode_time_s(), remaining)
        if teacher.swap_time is not None and shadow is None:
            shadow = copy.deepcopy((teacher, env))
            first_command = command
        step = env.step(command)
        if teacher.swap_time is not None and len(teacher.events) < FRAMES:
            teacher.events.append(rows(env.last_events))
        observed = step.observation
        if step.done:
            break
    if shadow is None or len(teacher.recorded) != FRAMES or len(teacher.events) != FRAMES:
        raise AssertionError("target reservation was not fully captured")
    actual = [event for batch in teacher.events for event in batch]
    recorded_replay = replay(teacher.snapshot, teacher.recorded)
    if recorded_replay != actual:
        raise AssertionError("identical input replay mismatches actual events")

    shadow_teacher, shadow_env = shadow
    if shadow_teacher.motor is not shadow_env.motor:
        raise AssertionError("shadow teacher and simulator must share cloned motor")
    shadow_events = []
    shadow_actions = []
    command = first_command
    for tick in range(FRAMES):
        if tick:
            target = shadow_env.privileged_next_target()
            remaining = (tuple((int(t.ordinal), float(t.episode_time_s))
                               for t in segment.targets[int(target.ordinal):])
                         if target is not None else ())
            command = shadow_teacher.action(
                shadow_obs, shadow_env.privileged_episode_time_s(), remaining)
        shadow_actions.append(command)
        shadow_result = shadow_env.step(command)
        shadow_events.extend(rows(shadow_env.last_events))
        shadow_obs = shadow_result.observation
    if shadow_events != actual or shadow_actions != teacher.recorded:
        raise AssertionError("full replanning shadow diverged from actual execution")

    fixed = replay(teacher.snapshot, teacher.recorded, fixed=teacher.first_action)
    intervention = replay(teacher.snapshot, teacher.recorded, ablate=True)
    times = {"fixed_ms": first_down(fixed), "actual_ms": first_down(actual),
             "ablation_ms": first_down(intervention),
             "replanned_shadow_ms": first_down(shadow_events)}
    return {
        "format": "dmdod-v062-stage-c-causal-audit",
        "chart": str(chart), "swap_at_ms": round(teacher.swap_time*1000, 3),
        "source": SOURCE, "candidate": AXIS, "ordinals": ORDINALS,
        "same_commands_same_events": True, "replanned_shadow_exact": True,
        "keydowns": times,
        "tie_delay": [r for r in teacher.sample_inversions
                      if r["axis"] == AXIS and START_S*1000 <= r["now_ms"] <=
                      START_S*1000 + FRAMES*10],
        "actual_commands": [
            {"tick": tick, "candidate_value": a.commands[PHYSICAL_AXES.index(AXIS)]}
            for tick, a in enumerate(teacher.recorded)],
        "limitation": "The one-tick ablation freezes subsequent recorded teacher commands.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("chart", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("refusing to overwrite output")
    result = diagnose(args.chart)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
