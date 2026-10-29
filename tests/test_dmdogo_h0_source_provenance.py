from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys
from types import ModuleType

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import dmdogo_h0_source_provenance as provenance


def test_loaded_source_hash_changes_when_source_changes(tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    path = scripts / "sample.py"
    path.write_text("VALUE = 1\n", encoding="utf-8")
    module = ModuleType("_provenance_test_sample")
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        provenance,
        "_git",
        lambda root, *args: "0" * 40 if args[0] == "rev-parse" else "",
    )

    before = provenance.capture_audit_source_provenance(tmp_path)
    key = "scripts/sample.py"
    assert before["loaded_local_python_sha256"][key] == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    assert before["worktree_dirty"] is False

    path.write_text("VALUE = 2\n", encoding="utf-8")
    after = provenance.capture_audit_source_provenance(tmp_path)
    assert before["loaded_local_python_sha256"][key] != (
        after["loaded_local_python_sha256"][key]
    )


def test_git_state_is_recorded_without_mutation(tmp_path, monkeypatch):
    calls = []

    def fake_git(root, *args):
        calls.append(args)
        if args[0] == "rev-parse":
            return "a" * 40
        return " M scripts/example.py"

    monkeypatch.setattr(provenance, "_git", fake_git)
    snapshot = provenance.capture_audit_source_provenance(tmp_path)
    assert snapshot["git_head"] == "a" * 40
    assert snapshot["worktree_dirty"] is True
    assert snapshot["worktree_status_sha256"] == hashlib.sha256(
        b" M scripts/example.py"
    ).hexdigest()
    assert calls == [
        ("status", "--porcelain=v1", "--untracked-files=all"),
        ("rev-parse", "HEAD"),
    ]


def test_h010_h011_h012_report_sources_recorded():
    import audit_dmdogo_v09_h010_all_quick_seams as h010
    import audit_dmdogo_v09_h011_adaptive_release as h011
    import audit_dmdogo_v09_h012_synthetic_counterexamples as h012

    root = SCRIPTS.parent
    snapshot = provenance.capture_audit_source_provenance(root)
    assert len(snapshot["git_head"]) == 40
    for module in (h010, h011, h012):
        path = Path(module.__file__).resolve()
        relative = path.relative_to(root).as_posix()
        assert snapshot["loaded_local_python_sha256"][relative] == (
            hashlib.sha256(path.read_bytes()).hexdigest()
        )

        # Verify the report-building path contains the new provenance field,
        # without running any expensive H0 research experiment.
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert any(
            isinstance(node, ast.Dict)
            and any(
                isinstance(key, ast.Constant)
                and key.value == "audit_source_provenance"
                for key in node.keys
            )
            for node in ast.walk(tree)
        )
