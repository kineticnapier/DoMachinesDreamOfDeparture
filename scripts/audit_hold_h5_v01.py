from __future__ import annotations

"""Train-only Hold timing parity and deterministic judgement audit."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from dmdod.adofai.chart import _normalize_loose_json, parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.evaluation.hold_judgement_v01 import (
    HoldBehavior,
    HoldInputFrame,
    HoldIntervalContract,
    HoldJudgementStateMachine,
    TimeWindow,
)


FORMAT = "dmdod-hold-h5-train-audit-v0.1"
TRAIN_DIR = Path("data/DMDOD-v140-tuf-p11-p20/Train")
EXPECTED = {
    "04349 - ヒアソビ (feat. 初音ミク) - Zedwards.adofai":
        "baaf6bff3cd012d8b0e4f123f58868c1cd4f262a35394e32eb2ebcba898caab9",
    "10907 - Remote Access Trojan - 0mni & Trippy.adofai":
        "d7e5a1cb5abc8ea1104116ac154f5f1e0b9fd889fdf2650ed1fe1d27510f4b82",
    "13131 - 『계획 이상 有!』 (feat. 重音テト) - Team Name.adofai":
        "167e781dca406fcec3a0ee754a173d40a50b23f1714ee93758ca3eaf53564d64",
}
TOLERANCE_S = 1e-6


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _root(path: Path) -> dict:
    return json.loads(
        _normalize_loose_json(path.read_text(encoding="utf-8-sig")),
        strict=False,
    )


def _compile_root(root: dict):
    return compile_adofai(parse_adofai_text(json.dumps(root, ensure_ascii=False)))


def _timing_rows(path: Path) -> list[dict]:
    root = _root(path)
    original = _compile_root(root)
    rows = []
    actions = root.get("actions") or []
    for action_index, action in enumerate(actions):
        if not isinstance(action, dict) or action.get("eventType") != "Hold":
            continue
        if action.get("active", True) is False:
            continue
        floor_index = action.get("floor")
        duration = action.get("duration")
        if type(floor_index) is not int or type(duration) is not int:
            continue
        if not 0 <= floor_index < len(original.floors) - 1 or duration < 0:
            continue

        counterfactual_root = deepcopy(root)
        del counterfactual_root["actions"][action_index]
        counterfactual = _compile_root(counterfactual_root)
        floor = original.floors[floor_index]
        without = counterfactual.floors[floor_index]
        hold = floor.hold
        if hold is None:
            raise RuntimeError("active non-final Hold disappeared from compilation")
        expected_additional_s = duration * 120.0 / floor.bpm
        start_shift_s = floor.target_time_s - without.target_time_s
        exit_shift_s = floor.exit_time_s - without.exit_time_s
        next_target_shift_s = (
            original.floors[floor_index + 1].target_time_s
            - counterfactual.floors[floor_index + 1].target_time_s
        )
        episode_additional_s = hold.additional_time_s / (
            original.pitch_percent * 0.01
        )
        same_floor_actions = tuple(
            str(item.get("eventType"))
            for item in actions
            if isinstance(item, dict) and item.get("floor") == floor_index
        )
        errors = {
            "start_shift_s": abs(start_shift_s),
            "additional_s": abs(hold.additional_time_s - expected_additional_s),
            "exit_shift_s": abs(exit_shift_s - expected_additional_s),
            "next_target_shift_s": abs(next_target_shift_s - expected_additional_s),
        }
        rows.append({
            "action_index": action_index,
            "floor": floor_index,
            "duration": duration,
            "bpm_at_hold": floor.bpm,
            "pitch_percent": original.pitch_percent,
            "start_time_chart_s": hold.start_time_s,
            "release_time_chart_s": hold.end_time_s,
            "additional_rotation_rad": hold.additional_angle_rad,
            "expected_additional_chart_s": expected_additional_s,
            "compiled_additional_chart_s": hold.additional_time_s,
            "compiled_additional_episode_s": episode_additional_s,
            "counterfactual_start_shift_s": start_shift_s,
            "counterfactual_exit_shift_s": exit_shift_s,
            "counterfactual_next_target_shift_s": next_target_shift_s,
            "same_floor_actions": same_floor_actions,
            "has_setspeed": "SetSpeed" in same_floor_actions,
            "has_pause": "Pause" in same_floor_actions,
            "has_twirl": "Twirl" in same_floor_actions,
            "has_multiplanet": "MultiPlanet" in same_floor_actions,
            "midspin": floor.midspin,
            "max_code_parity_error_s": max(errors.values()),
            "code_parity": max(errors.values()) <= TOLERANCE_S,
        })
    return rows


def _judgement_cases() -> list[dict]:
    contract = HoldIntervalContract(
        "h", TimeWindow(1.0, 1.1), TimeWindow(2.0, 2.1)
    )

    def run(name: str, behavior: HoldBehavior, frames: tuple[HoldInputFrame, ...]):
        machine = HoldJudgementStateMachine(behavior)
        machine.arm(contract)
        updates = []
        for frame in frames:
            update = machine.observe(frame)
            updates.append(update)
            if update.state.value in ("completed", "failed"):
                break
        final = updates[-1]
        return {
            "name": name,
            "behavior": behavior.value,
            "states": [item.state.value for item in updates],
            "signals": [[signal.value for signal in item.signals] for item in updates],
            "final": final.state.value,
            "failure": final.failure.value if final.failure else None,
            "ordinary_hit_credit": sum(item.ordinary_hit_credit for item in updates),
            "hold_completion_credit": sum(
                item.hold_completion_credit for item in updates
            ),
        }

    cases = [
        run("normal", HoldBehavior.NORMAL, (
            HoldInputFrame(1.0, down=frozenset((1, 2))),
            HoldInputFrame(1.5, down=frozenset((7,))),
            HoldInputFrame(2.0, up=frozenset((1,))),
        )),
        run("normal_early_release", HoldBehavior.NORMAL, (
            HoldInputFrame(1.0, down=frozenset((1,))),
            HoldInputFrame(1.5, up=frozenset((1,))),
        )),
        run("can_hit_end", HoldBehavior.CAN_HIT_END, (
            HoldInputFrame(1.0, down=frozenset((1,))),
            HoldInputFrame(2.0, down=frozenset((8,))),
        )),
        run("no_hold_needed", HoldBehavior.NO_HOLD_NEEDED, (
            HoldInputFrame(1.0, down=frozenset((1,))),
            HoldInputFrame(1.2, up=frozenset((1,))),
            HoldInputFrame(2.0, down=frozenset((8,))),
        )),
        run("missed_start", HoldBehavior.NORMAL, (HoldInputFrame(1.11),)),
        run("missed_release", HoldBehavior.NORMAL, (
            HoldInputFrame(1.0, down=frozenset((1,))),
            HoldInputFrame(2.11),
        )),
    ]
    return cases


def run(train_dir: Path) -> dict:
    charts = []
    for filename, expected_hash in EXPECTED.items():
        path = train_dir / filename
        digest = _sha(path)
        if digest != expected_hash:
            raise ValueError(f"Train chart hash drift: {filename}")
        rows = _timing_rows(path)
        chart = _compile_root(_root(path))
        segment = build_playable_segment(chart)
        charts.append({
            "filename": filename,
            "sha256": digest,
            "compiled_floor_count": len(chart.floors),
            "ordinary_target_count": len(segment.targets),
            "hold_count": len(rows),
            "holds": rows,
        })
    holds = [row for chart in charts for row in chart["holds"]]
    cases = _judgement_cases()
    checks = {
        "fixed_train_charts_3": len(charts) == 3,
        "fixed_hold_actions_25": len(holds) == 25,
        "all_code_parity_within_1us": all(row["code_parity"] for row in holds),
        "duration_values_0_1_2_3": sorted({row["duration"] for row in holds})
        == [0, 1, 2, 3],
        "ordinary_and_completion_credit_separate": all(
            case["ordinary_hit_credit"] <= 1
            and case["hold_completion_credit"] <= 1
            for case in cases
        ),
        "normal_success": cases[0]["final"] == "completed",
        "normal_early_release_fails": cases[1]["failure"] == "early_release",
        "can_hit_end_success": cases[2]["final"] == "completed",
        "no_hold_needed_success": cases[3]["final"] == "completed",
        "missed_start_fails": cases[4]["failure"] == "start_missed",
        "missed_release_fails": cases[5]["failure"] == "release_missed",
    }
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Train80 fixed Hold charts only; no Validation20/Reserve20/FINAL",
        "verdict": "CODE_PARITY_PASS_RUNTIME_UNVERIFIED"
        if all(checks.values()) else "HOLD_H5_AUDIT_FAILED",
        "tolerance_s": TOLERANCE_S,
        "checks": checks,
        "summary": {
            "chart_count": len(charts),
            "hold_count": len(holds),
            "max_code_parity_error_s": max(
                row["max_code_parity_error_s"] for row in holds
            ),
            "duration_counts": {
                str(duration): sum(row["duration"] == duration for row in holds)
                for duration in sorted({row["duration"] for row in holds})
            },
            "combined_cases": {
                name: sum(row[name] for row in holds)
                for name in (
                    "has_setspeed", "has_pause", "has_twirl",
                    "has_multiplanet", "midspin",
                )
            },
        },
        "timing_evidence": {
            "status": "CODE_PARITY_ONLY",
            "reference": (
                "audited stock formula: each positive duration loop adds two "
                "crotchets at the Hold floor BPM"
            ),
            "live_game_capture": "UNVERIFIED",
            "charts": charts,
        },
        "judgement_evidence": {
            "status": "DETERMINISTIC_STATE_MACHINE_ONLY",
            "live_game_capture": "UNVERIFIED",
            "cases": cases,
            "consecutive_hold": "UNVERIFIED_FAIL_CLOSED",
        },
    }


def _write_new(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.train_dir)
    _write_new(args.output, result)
    print("verdict:", result["verdict"])
    print("summary:", result["summary"])
    print("checks:", result["checks"])


if __name__ == "__main__":
    main()
