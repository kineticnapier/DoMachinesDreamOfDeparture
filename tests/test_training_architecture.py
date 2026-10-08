from __future__ import annotations

import ast
from pathlib import Path


def test_stable_training_package_does_not_depend_on_versioned_scripts() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "dmdod" / "training"
    offenders: list[str] = []

    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.append(node.module)
                imported.extend(alias.name for alias in node.names)
            if any("train_real_chart_v" in name for name in imported):
                offenders.append(path.name)
                break

    assert offenders == [], (
        "stable training modules must not import historical versioned scripts: "
        + ", ".join(offenders)
    )


def test_stable_entrypoint_name_is_not_versioned() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "train_real_chart.py").is_file()
    assert (root / "scripts" / "run_training.ps1").is_file()


def test_dmdod_package_root_stays_small() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "dmdod"
    root_files = sorted(
        path.name
        for path in root.glob("*.py")
    )
    assert root_files == [
        "__init__.py",
        "calibration.py",
        "profiles.py",
    ]


def test_legacy_flat_module_imports_alias_structured_modules() -> None:
    import dmdod.adofai_chart as legacy_chart
    import dmdod.body as legacy_body
    import dmdod.fly_connectome_policy as legacy_connectome
    import dmdod.n_key_training as legacy_training

    from dmdod.adofai import chart
    from dmdod.connectome import fly_policy
    from dmdod.motor import body
    from dmdod.training import n_key

    assert legacy_chart is chart
    assert legacy_body is body
    assert legacy_connectome is fly_policy
    assert legacy_training is n_key

    assert legacy_chart.__legacy_target__ == "dmdod.adofai.chart"
    assert legacy_body.__legacy_target__ == "dmdod.motor.body"
    assert legacy_connectome.__legacy_target__ == "dmdod.connectome.fly_policy"
    assert legacy_training.__legacy_target__ == "dmdod.training.n_key"


def test_stable_runner_registers_boundary_trust_mode() -> None:
    from dmdod.training.runner import _RUNNERS

    assert "budget_action_trust" in _RUNNERS
    assert "budget_survival_trust" in _RUNNERS
    assert "budget_boundary_trust" in _RUNNERS
    assert "trajectory_probe" in _RUNNERS
