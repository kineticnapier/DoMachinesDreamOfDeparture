from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import dmdogo_v08_provenance as provenance
from dmdogo_v08_provenance import FROZEN_HEAD, digest_json
from run_dmdogo_v08_verified_train_case import (
    case_from_manifest, validate_chart, validate_provenance_against_manifest,
)


def _sandbox_git(monkeypatch, repo):
    entries = {}
    for i in range(12):
        path = f"src/dmdod/runtime_{i}.py"
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"#{i}\n", encoding="utf-8")
        entries[path] = f"{i + 1:040x}"
    def fake_git(_root, *args):
        if args[0] == "ls-tree":
            return "\n".join(f"100644 blob {v}\t{k}" for k, v in entries.items())
        if args[0] == "rev-parse":
            return "actual-head"
        if args[0] == "hash-object":
            path = args[-1]
            return entries[path] if (repo / path).read_text(
                encoding="utf-8") == (
                f"#{int(Path(path).stem.split('_')[-1])}\n") else "0" * 40
        raise AssertionError(args)
    monkeypatch.setattr(provenance, "_git", fake_git)
    return entries


def test_frozen_source_tree_verifies_every_runtime_dependency(monkeypatch, tmp_path):
    entries = _sandbox_git(monkeypatch, tmp_path)
    output = provenance.verify_source_tree(tmp_path)
    assert output["tracked_runtime_files_verified"] == len(entries)
    assert output["working_runtime_sources_match"] is True
    assert output["code_digest"] == digest_json(dict(sorted(entries.items())))


def test_changed_frozen_dependency_rejected(monkeypatch, tmp_path):
    _sandbox_git(monkeypatch, tmp_path)
    (tmp_path / "src/dmdod/runtime_9.py").write_text("#different\n")
    with pytest.raises(RuntimeError, match="runtime_9"):
        provenance.verify_source_tree(tmp_path)


def test_new_runtime_module_rejected(monkeypatch, tmp_path):
    _sandbox_git(monkeypatch, tmp_path)
    (tmp_path / "src/dmdod/new_shadow.py").write_text("#shadow\n")
    with pytest.raises(RuntimeError, match="added runtime module"):
        provenance.verify_source_tree(tmp_path)


def _selection(chart, sha):
    return {
        "format": "dmdogo-v08-frozen-train-generalization-manifest-v1",
        "frozen_implementation_commit": FROZEN_HEAD,
        "frozen_code_digest": "f" * 64,
        "cases": [{"chart": chart.name, "path": str(chart),
                   "sha256": sha, "event_count": 20}],
    }


def test_only_preselected_real_train_chart_is_accepted(tmp_path):
    root = tmp_path / "Train"
    root.mkdir()
    chart = root / "example.adofai"
    chart.write_bytes(b"chart bytes")
    sha = hashlib.sha256(b"chart bytes").hexdigest()
    selection = _selection(chart, sha)
    assert case_from_manifest(selection, sha)["sha256"] == sha
    assert validate_chart(selection["cases"][0], root) == chart.resolve()
    with pytest.raises(ValueError, match="exactly one"):
        case_from_manifest(selection, "0" * 64)
    chart.write_bytes(b"modified chart")
    with pytest.raises(ValueError, match="changed"):
        validate_chart(selection["cases"][0], root)


def test_chart_outside_train_is_rejected(tmp_path):
    train = tmp_path / "Train"
    train.mkdir()
    outside = tmp_path / "outside.adofai"
    outside.write_bytes(b"outside")
    sha = hashlib.sha256(b"outside").hexdigest()
    with pytest.raises(ValueError, match="not under Train"):
        validate_chart(_selection(outside, sha)["cases"][0], train)


def test_launcher_preflight_rejects_old_manifest_and_nonfrozen_reuse():
    selection = {"frozen_code_digest": "f" * 64}
    p = {"source": {"code_digest": "f" * 64},
         "settings": {"planning_mode": "gesture_v08",
                      "minimum_reuse_ms": 80.,
                      "stop_after_s": None}}
    validate_provenance_against_manifest(p, selection)
    p["settings"]["minimum_reuse_ms"] = 1.
    with pytest.raises(ValueError):
        validate_provenance_against_manifest(p, selection)
    p["settings"]["minimum_reuse_ms"] = 80.
    with pytest.raises(ValueError, match="lacks full"):
        validate_provenance_against_manifest(p, {})
