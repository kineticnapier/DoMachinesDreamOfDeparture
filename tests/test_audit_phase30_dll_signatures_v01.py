from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_phase30_dll_signatures_v01.py"


def _load():
    spec = importlib.util.spec_from_file_location("phase30", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audited_decompiled_sources_have_exact_signatures_and_order() -> None:
    source_root = Path(r"F:\adofai-code")
    if not source_root.exists():
        pytest.skip("Decompiled ADOFAI source tree F:\\adofai-code is unavailable")
    result = _load().audit(source_root)
    assert result["verdict"] == "DLL_SIGNATURES_VERIFIED"
    assert all(row["verified"] for row in result["signatures"])
    assert result["control_flow"]["simulated_update_order_verified"]
    assert result["control_flow"]["no_fail_die_hit_order_verified"]
    assert result["runtime_parity"] == "RUNTIME_PARITY_UNVERIFIED"


def test_installed_identity_is_not_claimed_as_codex_observation() -> None:
    source_root = Path(r"F:\adofai-code")
    if not source_root.exists():
        pytest.skip("Decompiled ADOFAI source tree F:\\adofai-code is unavailable")
    result = _load().audit(source_root)
    assert result["installed_dll_identity"] == "USER_CONFIRMED_NOT_CODEX_EXECUTED"
