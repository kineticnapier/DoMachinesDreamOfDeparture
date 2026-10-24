from __future__ import annotations

"""Launch frozen v0.8 physical Motor and save a separate verified sidecar.

Does not edit the frozen runner or permit per-case hyperparameters. A strict
subprocess timeout covers compile + Planner + Motor, unlike the frozen
runner's inner timeout. Runs only a preselected Train chart.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from dmdogo_v08_provenance import (
    FROZEN_HEAD, MINIMUM_REUSE_MS, PLANNING_MODE, WRAPPER_FORMAT,
    preflight, sha256_file, digest_json,
)


def case_from_manifest(manifest: dict, sha: str) -> dict:
    if manifest.get("format") != "dmdogo-v08-frozen-train-generalization-manifest-v1":
        raise ValueError("unexpected selection manifest")
    if manifest.get("frozen_implementation_commit") != FROZEN_HEAD:
        raise ValueError("selection manifest targets another implementation")
    matched = [case for case in manifest.get("cases", [])
               if case.get("sha256") == sha]
    if len(matched) != 1:
        raise ValueError("case sha256 must match exactly one selected Train chart")
    return matched[0]


def validate_chart(case: dict, root: Path) -> Path:
    if root.name.lower() != "train" or not root.is_dir():
        raise ValueError("manifest does not identify an existing Train root")
    path = Path(case["path"]).resolve()
    if not path.is_file() or path.suffix.lower() != ".adofai":
        raise ValueError("selected chart missing")
    if not path.is_relative_to(root):
        raise ValueError("selected chart is not under Train")
    if any(item.lower() in {"validation", "validation20", "final", "holdout"}
           for item in path.relative_to(root).parts):
        raise ValueError("selected chart belongs to a forbidden subset")
    if sha256_file(path) != case["sha256"]:
        raise ValueError("selected chart content has changed")
    return path


def validate_provenance_against_manifest(provenance: dict, manifest: dict) -> None:
    actual = provenance["source"]["code_digest"]
    frozen = manifest.get("frozen_code_digest")
    if not isinstance(frozen, str) or not frozen:
        raise ValueError("selection manifest lacks full frozen_code_digest; regenerate with updated selector")
    if actual != frozen:
        raise ValueError("frozen source-tree digest differs from selection manifest")
    settings = provenance["settings"]
    if (settings.get("minimum_reuse_ms") != MINIMUM_REUSE_MS or
            settings.get("planning_mode") != PLANNING_MODE or
            settings.get("stop_after_s") is not None):
        raise ValueError("physical settings differ from frozen v0.8 defaults")


def _write_json_once(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(body, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _update_own_sidecar(path: Path, body: dict) -> None:
    # Exclusively replaces this launch's own initial in_progress manifest.
    staged = path.with_name(path.name + ".new")
    _write_json_once(staged, body)
    os.replace(staged, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--case-sha", required=True)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--status-root", required=True, type=Path)
    parser.add_argument("--wall-timeout-s", type=float, default=1200.)
    args = parser.parse_args()
    if not 1 <= args.wall_timeout_s <= 7200:
        parser.error("wall-timeout-s must be 1..7200 seconds")
    repo = Path(__file__).resolve().parents[1]
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        case = case_from_manifest(manifest, args.case_sha)
        root = Path(manifest["train_root"]).resolve()
        chart = validate_chart(case, root)
        provenance = preflight(repo)
        validate_provenance_against_manifest(provenance, manifest)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    prefix = args.case_sha[:12]
    results_dir = args.results_root / prefix
    status = args.status_root / f"dmdogo-v08-train-{prefix}.status.json"
    sidecar = args.results_root / "_provenance" / f"{prefix}.json"
    if any(p.exists() for p in (results_dir, status, sidecar)):
        parser.error("refusing to overwrite prior result/status/provenance")
    identity = {
        **provenance,
        "chart": case["chart"],
        "chart_sha256": case["sha256"],
        "selection_manifest_sha256": sha256_file(args.manifest),
        "selection_manifest_frozen_code_digest": manifest["frozen_code_digest"],
        "state": "in_progress",
        "result_directory": str(results_dir),
        "status_file": str(status),
        "runner_exit_code": None,
        "summary_sha256": None,
    }
    _write_json_once(sidecar, identity)
    command = [
        sys.executable, str(repo / "scripts" / "run_dmdogo_v06_pattern_first.py"),
        "--planning-mode", PLANNING_MODE,
        "--minimum-reuse-ms", str(MINIMUM_REUSE_MS),
        "--chart", str(chart), "--output-dir", str(results_dir),
        "--status-file", str(status),
        "--wall-timeout-s", str(args.wall_timeout_s),
    ]
    outcome = "runner_error"
    exit_code = 1
    try:
        completed = subprocess.run(command, cwd=repo,
                                   timeout=args.wall_timeout_s, check=False)
        exit_code = completed.returncode
        outcome = "success" if exit_code == 0 else "runner_error"
    except subprocess.TimeoutExpired:
        exit_code = 124
        outcome = "outer_timeout"
    finally:
        summary = results_dir / "summary.json"
        identity.update({
            "state": outcome,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "runner_exit_code": exit_code,
            "summary_sha256": sha256_file(summary) if summary.is_file() else None,
        })
        _update_own_sidecar(sidecar, identity)
    print(json.dumps({
        "chart": case["chart"], "exit_code": exit_code,
        "outcome": outcome, "provenance": str(sidecar),
    }, ensure_ascii=False), flush=True)
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
