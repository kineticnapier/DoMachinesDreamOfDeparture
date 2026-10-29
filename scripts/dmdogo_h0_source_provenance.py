from __future__ import annotations

"""Identify the Python audit implementation used for a newly written H0 report.

This is separate from the frozen v0.8 source-tree verification. It records
loaded repository-local source bytes and the current Git state without changing
any research calculations or existing report files.
"""

import hashlib
from pathlib import Path
import subprocess
import sys


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def capture_audit_source_provenance(repo_root: Path) -> dict:
    """Capture local loaded Python sources, HEAD and worktree dirtiness."""
    root = repo_root.resolve()
    sources: dict[str, str] = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if (
            path.suffix != ".py"
            or not relative.parts
            or relative.parts[0] not in ("scripts", "src")
        ):
            continue
        sources[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()

    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    return {
        "git_head": _git(root, "rev-parse", "HEAD"),
        "worktree_dirty": bool(status),
        "worktree_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "loaded_local_python_sha256": dict(sorted(sources.items())),
    }
