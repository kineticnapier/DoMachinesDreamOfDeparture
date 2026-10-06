from __future__ import annotations

from pathlib import Path


def test_stable_training_package_does_not_depend_on_versioned_scripts() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "dmdod" / "training"
    offenders: list[str] = []

    for path in sorted(root.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "train_real_chart_v" in source:
            offenders.append(path.name)

    assert offenders == [], (
        "stable training modules must not import historical versioned scripts: "
        + ", ".join(offenders)
    )


def test_stable_entrypoint_name_is_not_versioned() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "train_real_chart.py").is_file()
    assert (root / "scripts" / "run_training.ps1").is_file()
