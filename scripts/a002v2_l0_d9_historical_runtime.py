#!/usr/bin/env python3
"""D9: audit archived GitHub Actions runtime evidence without any learning."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

PROTOCOL = "docs/research-a-v2-l0-d9-historical-runtime-protocol.md"
JOBS = {
    "r1": (41309837474, 124857703147),
    "d3": (41369778494, 125037538287),
    "d5": (41380105087, 125068823089),
    "d7": (41391082538, 125102397739),
    "d8": (41397065687, 125120568955),
}
ENV_KEYS = ("PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
CHECK_FIELDS = ("runner_version", "image", "image_release", "python_version",
                "torch_version", *ENV_KEYS)
ORIGINAL_ENTROPY = 1.3561848871858129
NEW_ENTROPY = 1.3561848983669778
DEFAULT_ENTROPY = 1.356184897858743


def match(pattern: str, text: str) -> str | None:
    found = re.search(pattern, text, re.MULTILINE)
    return found.group(1) if found else None


def parse_log(data: bytes) -> dict:
    text = re.sub(r"\x1b\[[0-9;]*m", "", data.decode("utf-8-sig", errors="replace"))
    result = {
        "log_sha256": hashlib.sha256(data).hexdigest(),
        "runner_version": match(r"Current runner version:\s*'([^']+)'", text),
        "image": match(r"\bImage:\s*(ubuntu-[\d.]+)", text),
        "image_release": match(r"\bImage Release:\s*(https://\S+)", text),
        "python_version": match(r"/Python/(3\.11\.\d+)/x64", text),
        "torch_version": match(r"\btorch-(2\.5\.1\+cpu)\b", text),
        "cpu_model_in_log": match(r"(?im)^\s*(?:cpu_model|model name)\s*:\s*([^\r\n]+)", text),
        "aten_capability_in_log": match(r"(?im)^\s*ATEN_CPU_CAPABILITY\s*:\s*([^\r\n]+)", text),
    }
    for key in ENV_KEYS:
        result[key] = match(r"\b" + key + r":\s*([^\s]+)", text)
    return result


def audit(logs: dict[str, bytes], d7: dict, native: dict, default: dict) -> dict:
    report = {
        "schema": "research-a-v2-l0-d9-historical-runtime-audit-v1",
        "protocol": PROTOCOL,
        "preregistered_commit": "40b682d2c9b1ccdbf3514071617b791db38d4649",
        "classification": "invalid_archive_audit",
        "new_training_episodes": 0, "ppo_updates": 0,
        "new_physical_keydowns": 0,
        "historical_r2_gradients_verified": False,
        "formal_study_002_approved": False,
        "historical_jobs": {},
        "missing_or_conflicting_evidence": [],
    }
    issues = report["missing_or_conflicting_evidence"]
    for label, (run_id, job_id) in JOBS.items():
        data = logs.get(label)
        if data is None:
            issues.append(label + ": missing log")
            continue
        row = parse_log(data)
        row["run_id"] = run_id
        row["job_id"] = job_id
        row["job_url"] = (
            "https://github.com/kineticnapier/DoMachinesDreamOfDeparture/"
            "actions/runs/" + str(run_id) + "/job/" + str(job_id)
        )
        report["historical_jobs"][label] = row
        for field in CHECK_FIELDS:
            if row[field] is None:
                issues.append(label + ": missing " + field)
    if len(report["historical_jobs"]) == len(JOBS):
        report["all_five_job_logs_present"] = True
        report["visible_environment_consistent"] = all(
            len({report["historical_jobs"][label][field] for label in JOBS}) == 1
            for field in CHECK_FIELDS
        )
    else:
        report["all_five_job_logs_present"] = False
        report["visible_environment_consistent"] = False
    if not report["visible_environment_consistent"]:
        issues.append("visible environment differs or incomplete")

    try:
        if d7["classification"] != "initial_evaluation_diverged":
            raise ValueError("D7 classification")
        diff = d7["stage0_difference"]
        if (diff["expected"], diff["actual"]) != (ORIGINAL_ENTROPY, NEW_ENTROPY):
            raise ValueError("D7 entropy")
        if (native["stage0"]["argmax"]["mean_entropy"] != NEW_ENTROPY or
                default["stage0"]["argmax"]["mean_entropy"] != DEFAULT_ENTROPY):
            raise ValueError("D8 entropy")
        if native["initial_state_sha256"] != default["initial_state_sha256"]:
            raise ValueError("D8 initial weights")
        if native["runtime"]["cpu_model"] != default["runtime"]["cpu_model"]:
            raise ValueError("D8 two conditions CPU model")
        if native["runtime"]["cpu_capability"] == default["runtime"]["cpu_capability"]:
            raise ValueError("D8 capability not distinct")
        if d7["runtime"]["cpu_model"] != native["runtime"]["cpu_model"]:
            raise ValueError("D7/D8 CPU model")
        for label, record in (("d7", d7), ("d8", native)):
            runtime = record["runtime"]
            if (runtime["torch"] != "2.5.1+cpu" or runtime["torch_threads"] != 1 or
                    not runtime["mkldnn_enabled"]):
                raise ValueError(label + " torch/runtime settings")
        report["d7_d8_cpu_model"] = native["runtime"]["cpu_model"]
        report["d7_d8_same_cpu_model"] = True
        report["d8_cpu_capability"] = {
            "native": native["runtime"]["cpu_capability"],
            "default": default["runtime"]["cpu_capability"],
        }
        report["stage0_entropy"] = {
            "original_r1_and_d3": ORIGINAL_ENTROPY,
            "d7_and_d8_native": NEW_ENTROPY,
            "d8_default": DEFAULT_ENTROPY,
            "d7_minus_original": NEW_ENTROPY - ORIGINAL_ENTROPY,
            "d8_default_minus_native": DEFAULT_ENTROPY - NEW_ENTROPY,
        }
        report["archived_d8_physical"] = {
            mode: {key: native["stage0"][mode][key] for key in
                   ("hits", "extra_keydowns", "physical_left", "physical_right", "targets")}
            for mode in ("argmax", "sampled")
        }
        for label in ("d7", "d8"):
            report["historical_jobs"][label]["cpu_model_from_artifact"] = (
                d7["runtime"]["cpu_model"] if label == "d7" else
                native["runtime"]["cpu_model"]
            )
    except (KeyError, TypeError, ValueError) as exc:
        issues.append("archived D7/D8 data mismatch: " + str(exc))

    report["unknown_historical_cpu_models"] = [
        label for label in ("r1", "d3", "d5")
        if label not in report["historical_jobs"] or
        not report["historical_jobs"][label]["cpu_model_in_log"]
    ]
    report["unknown_historical_aten_dispatch"] = [
        label for label in ("r1", "d3", "d5", "d7")
        if label not in report["historical_jobs"] or
        not report["historical_jobs"][label]["aten_capability_in_log"]
    ]
    if not issues:
        report["classification"] = (
            "historical_cpu_unresolved"
            if report["unknown_historical_cpu_models"] or
               report["unknown_historical_aten_dispatch"]
            else "runtime_evidence_complete"
        )
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", action="append", required=True, help="label=path")
    p.add_argument("--d7", type=Path, required=True)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--default", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("refuse to overwrite D9 output")
    files = {}
    for spec in args.log:
        label, sep, name = spec.partition("=")
        if not sep or label not in JOBS or label in files:
            p.error("invalid or duplicate --log: " + spec)
        path = Path(name)
        if path.exists():
            files[label] = path.read_bytes()
    report = audit(
        files, json.loads(args.d7.read_text(encoding="utf-8")),
        json.loads(args.native.read_text(encoding="utf-8")),
        json.loads(args.default.read_text(encoding="utf-8"))
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print("D9_RESULT", json.dumps({
        "classification": report["classification"],
        "visible_environment_consistent": report["visible_environment_consistent"],
        "unknown_historical_cpu_models": report["unknown_historical_cpu_models"],
        "unknown_historical_aten_dispatch": report["unknown_historical_aten_dispatch"],
        "issues": report["missing_or_conflicting_evidence"],
        "new_training_episodes": 0,
    }, ensure_ascii=False), flush=True)
    if report["classification"] == "invalid_archive_audit":
        raise SystemExit("D9 invalid archive audit; preserve negative evidence")


if __name__ == "__main__":
    main()
