from __future__ import annotations

"""Targeted, read-only diagnosis for saved Fingering causal-lead failures.

The policy, checkpoint, chart, Motor constants, and evaluator are reused
unchanged.  A callback observes the existing execution loop; it never changes
actions or policy inputs.  Oracle target ordinals/times appear only in this
diagnostic/evaluator output, not in Network context construction.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_fingering_causal_early_v02 as causal
import audit_oracle_fingering_v02 as teacher
import run_full_chart_fingering_replay_v02 as runner

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai


DEFAULT_CHART = Path(
    "data/DMDOD-v140-tuf-p11-p20/Validation/"
    "14483 - QuomodocunquizE - 없는닉이없네.adofai"
)
DEFAULT_CHECKPOINT = Path(
    "checkpoints/fingering/fingering_network_v01-4fa2ae1.pt"
)
DEFAULT_SOURCE_DIR = Path(
    "artifacts/fingering_network_v02/validation20-80-fast-r3/arms"
)
DEFAULT_OUTPUT = Path(
    "artifacts/fingering_network_v02/validation20-80-fast-r3/"
    "quomodocunquize-causal-failure-diagnostic.json"
)
POLICY_ARMS = (
    ("free", 0.0, "baseline"),
    ("free", 0.0, "early_decision"),
    ("free", 0.0, "causal_lead"),
    ("human_proxy", 0.5, "causal_lead"),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_saved_arm(source_dir: Path, chart_sha256: str,
                   policy: str, arm: str) -> dict:
    matches = list(source_dir.glob(
        f"chart-*-{chart_sha256[:16]}-{policy}-{arm}.json"
    ))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one saved {policy}/{arm} arm, found {len(matches)}"
        )
    payload = json.loads(matches[0].read_text(encoding="utf-8"))
    if payload["chart_sha256"] != chart_sha256:
        raise RuntimeError("saved arm chart SHA256 mismatch")
    return payload


def physical_snapshot(env) -> dict:
    motor = env.motor.motor
    obs = motor.observe()
    rows = {}
    for index, axis in enumerate(teacher.AXES):
        state = motor.body.states[index]
        rows[axis] = {
            "position_m": round(float(state.position_m), 9),
            "velocity_m_s": round(float(state.velocity_m_s), 9),
            "activation": round(float(state.activation), 9),
            "fatigue": round(float(state.fatigue), 9),
            "pressed": bool(obs.pressed_flags[index]),
        }
    logical = env.motor.observe()
    return {
        "episode_ms": round(env.privileged_episode_time_s() * 1000.0, 3),
        "axes": rows,
        "logical_pressed": [
            key for key, pressed in zip(logical.logical_keys,
                                        logical.logical_pressed) if pressed
        ],
    }


class WindowProbe:
    def __init__(self, event_times: dict[int, float], radius_s: float) -> None:
        self.event_times = event_times
        self.radius_s = radius_s
        self.frames: list[dict] = []
        self.decisions: list[dict] = []

    def _near(self, time_s: float) -> bool:
        return any(abs(time_s - due_s) <= self.radius_s
                   for due_s in self.event_times.values())

    def __call__(self, stage: str, payload: dict) -> None:
        now_s = float(payload["now_s"])
        if stage == "decision":
            event = payload["event"]
            if event.event_index not in self.event_times:
                return
            self.decisions.append({
                "event_index": event.event_index,
                "due_ms": round(event.episode_time_s * 1000.0, 3),
                "target_ordinals": list(event.target_ordinals),
                "chord_size": event.chord_size,
                "chosen": payload["decision"]["chosen"],
                "extra_lead_ms": payload["decision"]["extra_lead_ms"],
                "state": physical_snapshot(payload["env"]),
            })
            return
        if stage != "after_step" or not self._near(now_s):
            return
        action = payload["action"]
        self.frames.append({
            "control_start_ms": round(now_s * 1000.0, 3),
            "control_end_ms": round(
                payload["env"].privileged_episode_time_s() * 1000.0, 3
            ),
            "commands": {
                axis: float(value)
                for axis, value in zip(teacher.AXES, action.commands)
                if value != 0.0
            },
            "logical_events": [
                {
                    "time_ms": round(event.time_s * 1000.0, 6),
                    "contact": event.key,
                    "event": event.event.value,
                }
                for event in payload["logical_events"]
            ],
            "judgements": [
                {
                    "time_ms": round(float(row["time_s"]) * 1000.0, 6),
                    "contact": row["key"],
                    "target_ordinal_before_score": row["target_index"],
                    "margin": row["margin"],
                }
                for row in payload["judgements"]
            ],
            "new_miss_ordinals": list(payload["new_miss_ordinals"]),
            "new_too_early": int(payload["new_too_early"]),
            "state": physical_snapshot(payload["env"]),
        })


def compact_result(result: dict) -> dict:
    return {
        "stats": result["stats"],
        "termination_reason": result["termination_reason"],
        "interventions": result["interventions"],
        "decisions": result["decisions"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chart", type=Path, default=DEFAULT_CHART)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window-ms", type=float, default=80.0)
    parser.add_argument("--wall-timeout-s", type=float, default=120.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")

    chart_sha = file_sha256(args.chart)
    checkpoint_sha = file_sha256(args.checkpoint)
    saved_causal = load_saved_arm(
        args.source_dir, chart_sha, "free", "causal_lead"
    )
    failed = [
        int(row["event_index"])
        for row in saved_causal["result"]["decisions"]
        if row["outcome"] == "Miss"
    ]
    if not failed:
        raise RuntimeError("saved Free/Causal Lead arm has no failed events")

    segment = build_playable_segment(compile_adofai(load_adofai(args.chart)))
    events = causal.AutoplayTimingTeacher.from_segment(segment).events
    event_by_index = {event.event_index: event for event in events}
    event_times = {index: event_by_index[index].episode_time_s for index in failed}
    model = runner.load_model(args.checkpoint)
    outputs = {}
    for policy, alpha, arm in POLICY_ARMS:
        probe = WindowProbe(event_times, args.window_ms / 1000.0)
        result = causal.execute(
            segment, model, policy, alpha, arm,
            wall_timeout_s=args.wall_timeout_s, diagnostic_probe=probe,
        )
        if arm in ("baseline", "causal_lead"):
            saved = load_saved_arm(args.source_dir, chart_sha, policy, arm)
            expected = saved["result"]["stats"]
            if result["stats"] != expected:
                raise RuntimeError(
                    f"{policy}/{arm} rerun stats differ from saved arm"
                )
        outputs[f"{policy}/{arm}"] = {
            "result": compact_result(result),
            "probe_decisions": probe.decisions,
            "probe_frames": probe.frames,
        }

    payload = {
        "format": "dmdod-fingering-causal-failure-diagnostic-v0.2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only_diagnostic": True,
        "student_policy_inputs_changed": False,
        "chart": str(args.chart),
        "chart_sha256": chart_sha,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "failed_event_indices_from_saved_arm": failed,
        "window_ms": args.window_ms,
        "arms": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "failed_event_indices": failed,
        "arms": {key: value["result"]["stats"] for key, value in outputs.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
