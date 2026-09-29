from __future__ import annotations

import hashlib
import io
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


ROLE_NAMES = ("Train", "Validation", "Final")


@dataclass(frozen=True, slots=True)
class DatasetChart:
    role: str
    name: str
    source_label: str
    resolved_path: str
    content_sha256: str


@dataclass(frozen=True, slots=True)
class MultiChartDataset:
    root: str
    train: tuple[DatasetChart, ...]
    validation: tuple[DatasetChart, ...]
    final: tuple[DatasetChart, ...]

    @property
    def all_charts(self) -> tuple[DatasetChart, ...]:
        return (*self.train, *self.validation, *self.final)

    def signature(self) -> dict:
        def encode(items: tuple[DatasetChart, ...]) -> list[dict[str, str]]:
            return [
                {
                    "name": item.name,
                    "sha256": item.content_sha256,
                    "source": item.source_label,
                }
                for item in items
            ]

        return {
            "train": encode(self.train),
            "validation": encode(self.validation),
            "final": encode(self.final),
        }


def discover_multichart_dataset(root: str | Path) -> MultiChartDataset:
    source = Path(root).expanduser().resolve()
    if source.is_dir():
        roles = _discover_directory_dataset(source)
    elif source.is_file() and source.suffix.lower() == ".zip":
        roles = _discover_bundle_zip(source)
    else:
        raise ValueError(f"dataset root must be a directory or .zip bundle: {source}")

    train = tuple(roles["Train"])
    validation = tuple(roles["Validation"])
    final = tuple(roles["Final"])
    if not train:
        raise ValueError("dataset has no Train charts")
    if not validation:
        raise ValueError("dataset has no Validation charts")
    if not final:
        raise ValueError("dataset has no Final charts")

    hashes: dict[str, str] = {}
    for chart in (*train, *validation, *final):
        previous = hashes.get(chart.content_sha256)
        if previous is not None:
            raise ValueError(
                f"same chart content appears more than once: {previous!r} and {chart.source_label!r}"
            )
        hashes[chart.content_sha256] = chart.source_label

    return MultiChartDataset(str(source), train, validation, final)


def _discover_directory_dataset(root: Path) -> dict[str, list[DatasetChart]]:
    children = {child.name.casefold(): child for child in root.iterdir() if child.exists()}
    roles: dict[str, list[DatasetChart]] = {}
    for role in ROLE_NAMES:
        role_dir = children.get(role.casefold())
        if role_dir is None or not role_dir.is_dir():
            raise ValueError(f"dataset directory is missing {role}/")
        charts: list[DatasetChart] = []
        for item in sorted(role_dir.iterdir(), key=lambda value: value.name.casefold()):
            if item.name.startswith("."):
                continue
            if item.is_file() and item.suffix.lower() not in {".zip", ".adofai"}:
                continue
            charts.append(_chart_from_filesystem_item(role, item))
        roles[role] = charts
    return roles


def _chart_from_filesystem_item(role: str, item: Path) -> DatasetChart:
    if item.is_dir():
        candidates = sorted(item.rglob("*.adofai"))
        chosen = _choose_adofai_name([str(path.relative_to(item)).replace("\\", "/") for path in candidates])
        path = item / Path(chosen)
        raw = path.read_bytes()
        return _materialize_chart(role, item.name, str(path), raw)

    if item.suffix.lower() == ".adofai":
        return _materialize_chart(role, item.stem, str(item), item.read_bytes(), existing_path=item)

    if item.suffix.lower() == ".zip":
        return _chart_from_zip_bytes(role, item.stem, str(item), item.read_bytes())

    raise ValueError(f"unsupported chart source: {item}")


def _discover_bundle_zip(bundle_path: Path) -> dict[str, list[DatasetChart]]:
    result = {role: [] for role in ROLE_NAMES}
    with zipfile.ZipFile(bundle_path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        for role in ROLE_NAMES:
            groups: dict[str, list[str]] = {}
            for name in names:
                parts = [part for part in name.replace("\\", "/").split("/") if part]
                lowered = [part.casefold() for part in parts]
                try:
                    role_index = lowered.index(role.casefold())
                except ValueError:
                    continue
                if role_index + 1 >= len(parts):
                    continue
                first = parts[role_index + 1]
                groups.setdefault(first, []).append(name)

            for first, group_names in sorted(groups.items(), key=lambda item: item[0].casefold()):
                if len(group_names) == 1 and first.lower().endswith(".zip"):
                    nested_name = group_names[0]
                    nested_raw = archive.read(nested_name)
                    result[role].append(
                        _chart_from_zip_bytes(
                            role,
                            Path(first).stem,
                            f"{bundle_path}::{nested_name}",
                            nested_raw,
                        )
                    )
                    continue

                adofai_names = [name for name in group_names if name.lower().endswith(".adofai")]
                chosen = _choose_adofai_name(adofai_names)
                raw = archive.read(chosen)
                label = f"{bundle_path}::{chosen}"
                result[role].append(_materialize_chart(role, Path(first).stem, label, raw))

    return result


def _chart_from_zip_bytes(role: str, name: str, source_label: str, raw_zip: bytes) -> DatasetChart:
    try:
        with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
            candidates = [entry for entry in archive.namelist() if entry.lower().endswith(".adofai")]
            chosen = _choose_adofai_name(candidates)
            raw = archive.read(chosen)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid chart zip: {source_label}") from exc
    return _materialize_chart(role, name, f"{source_label}::{chosen}", raw)


def _choose_adofai_name(candidates: list[str]) -> str:
    if not candidates:
        raise ValueError("chart source contains no .adofai file")
    if len(candidates) == 1:
        return candidates[0]

    preferred = [
        name
        for name in candidates
        if Path(name).name.casefold() in {"main.adofai", "level.adofai"}
    ]
    if len(preferred) == 1:
        return preferred[0]

    # Many published ADOFAI chart packages ship one playable chart plus an
    # editor backup. If exactly one non-backup file remains, that is the chart
    # the game/editor would normally present to the player.
    non_backup = [
        name
        for name in candidates
        if "backup" not in Path(name).stem.casefold()
        and "autosave" not in Path(name).stem.casefold()
    ]
    if len(non_backup) == 1:
        return non_backup[0]

    raise ValueError(
        "chart source contains multiple .adofai files and no unique playable candidate: "
        + ", ".join(candidates)
    )


def _materialize_chart(
    role: str,
    name: str,
    source_label: str,
    raw: bytes,
    *,
    existing_path: Path | None = None,
) -> DatasetChart:
    digest = hashlib.sha256(raw).hexdigest()
    if existing_path is not None:
        resolved = existing_path.resolve()
    else:
        cache_dir = Path(tempfile.gettempdir()) / "dmdod_chart_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        resolved = cache_dir / f"{digest}.adofai"
        if not resolved.exists() or resolved.read_bytes() != raw:
            resolved.write_bytes(raw)

    return DatasetChart(
        role=role,
        name=name,
        source_label=source_label,
        resolved_path=str(resolved),
        content_sha256=digest,
    )
