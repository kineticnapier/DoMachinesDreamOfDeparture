from __future__ import annotations

"""Frozen DMDOGO v0.8 run provenance, independent of the frozen runner.

Checks ALL tracked Python/TOML source present at the frozen commit within
src/dmdod and scripts, not merely the planner and runner, and records a
dependency/environment fingerprint. This proves source consistency against
the frozen Git tree under ordinary accidental-change assumptions, NOT a
bit-for-bit match to the historical bool machine environment.
"""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

FROZEN_HEAD = "45f30bcaf968281400e2c2913a96dd95515222d2"
WRAPPER_FORMAT = "dmdogo-v08-verified-physical-run-v1"
MINIMUM_REUSE_MS = 80.0
PLANNING_MODE = "gesture_v08"
ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED", "PYTHONPATH",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS",
)


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo,
                          check=True, text=True, capture_output=True).stdout.strip()


def frozen_tracked_sources(repo: Path) -> dict[str, str]:
    """Git SHA1 blobs for every frozen Python/TOML/lock runtime path."""
    entries = _git(repo, "ls-tree", "-r", FROZEN_HEAD,
                   "--", "src/dmdod", "scripts", "pyproject.toml", "uv.lock")
    sources = {}
    for line in entries.splitlines():
        meta, path = line.split("\t", 1)
        parts = meta.split(" ")
        if len(parts) != 3 or parts[1] != "blob":
            continue
        if path.endswith((".py", ".toml", ".lock")):
            sources[path] = parts[2]
    if len(sources) < 10:
        raise RuntimeError("frozen dependency tree incomplete; refusing verification")
    return dict(sorted(sources.items()))


def frozen_code_digest(repo: Path) -> str:
    return digest_json(frozen_tracked_sources(repo))


def verify_source_tree(repo: Path) -> dict:
    expected = frozen_tracked_sources(repo)
    mismatches = []
    for name, reference_sha in expected.items():
        if not (repo / name).is_file():
            mismatches.append(name + ": missing")
            continue
        actual = _git(repo, "hash-object", f"--path={name}", name)
        if actual != reference_sha:
            mismatches.append(name + ": modified")
    # New .py under runtime package can shadow imports despite same old blobs.
    for path in (repo / "src" / "dmdod").rglob("*.py"):
        rel = path.relative_to(repo).as_posix()
        if rel not in expected:
            mismatches.append(rel + ": added runtime module")
    if mismatches:
        raise RuntimeError(
            "frozen implementation mismatch (" + str(len(mismatches)) + "): "
            + "; ".join(mismatches[:15]))
    return {
        "frozen_source_commit": FROZEN_HEAD,
        "actual_git_head": _git(repo, "rev-parse", "HEAD"),
        "code_digest": digest_json(expected),
        "tracked_runtime_files_verified": len(expected),
        "working_runtime_sources_match": True,
    }


def environment_identity() -> dict:
    dists = {}
    for item in importlib.metadata.distributions():
        name = item.metadata.get("Name") or "unnamed"
        dists[name.lower()] = item.version
    return {
        "python_version": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "dependencies": dict(sorted(dists.items())),
        "selected_environment": {key: os.environ.get(key) for key in ENV_KEYS},
    }


def motor_defaults() -> dict:
    # Called AFTER verifying the frozen source tree. No changes to Motor.
    import audit_oracle_fingering_v02 as teacher
    from dmdod.planning.motor_lead_compensation_v04 import MotorLeadCompensator
    from dmdod.planning.gesture_chunk_planner_v08 import GestureChunkPlannerV08
    from dmdod.planning.pattern_motor_adapter_v06 import PatternMotorMapping

    lead = MotorLeadCompensator()
    planner = GestureChunkPlannerV08()
    mapping = PatternMotorMapping.default_eight()
    return {
        "control_dt_s": teacher.CTRL,
        "physics_dt_s": teacher.PHYS,
        "preview_tail_s": teacher.PREVIEW_TAIL,
        "minimum_reuse_ms": MINIMUM_REUSE_MS,
        "planning_mode": PLANNING_MODE,
        "stop_after_s": None,
        "max_prediction_s": lead.max_prediction_s,
        "legacy_lead_s": lead.legacy_lead_s,
        "planner_beam_width": planner.beam_width,
        "planner_max_chunk_events": planner.max_chunk_events,
        "planner_restart_gap_s": planner.restart_gap_s,
        "physical_axes": list(mapping.physical_axes),
        "logical_keys": list(mapping.logical_keys),
    }


def preflight(repo: Path) -> dict:
    sources = verify_source_tree(repo)
    env = environment_identity()
    defaults = motor_defaults()
    return {
        "format": WRAPPER_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": sources,
        "environment_digest": digest_json(env),
        "environment": env,
        "settings": defaults,
        "settings_digest": digest_json(defaults),
    }
