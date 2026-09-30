from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from .adofai_chart import parse_adofai_bytes
from .tuf_dataset import TufCandidate, chart_filename, choose_diverse, download_chart


@dataclass(frozen=True, slots=True)
class BalancedDatasetSelection:
    train: tuple[TufCandidate, ...]
    validation: tuple[TufCandidate, ...]
    final: tuple[TufCandidate, ...]
    relaxed_creator_overlap: bool


@dataclass(frozen=True, slots=True)
class BalancedPreflightResult:
    selection: BalancedDatasetSelection
    rejected: tuple[tuple[int, str], ...]
    raw_by_level_id: dict[int, bytes]


def _difficulty_pool(candidates: Iterable[TufCandidate], difficulty: int) -> tuple[TufCandidate, ...]:
    label = f"p{difficulty}"
    return tuple(
        candidate
        for candidate in candidates
        if candidate.difficulty_name.strip().casefold() == label
    )


def _creator_ids(items: Iterable[TufCandidate]) -> set[int]:
    return {creator_id for item in items for creator_id in item.creator_ids}


def _select_role(
    pool: tuple[TufCandidate, ...],
    *,
    difficulties: tuple[int, ...],
    per_difficulty: int,
    excluded_level_ids: set[int],
    excluded_song_keys: set[str],
    excluded_creator_ids: set[int],
) -> tuple[TufCandidate, ...] | None:
    selected: list[TufCandidate] = []
    level_ids = set(excluded_level_ids)
    song_keys = set(excluded_song_keys)

    for difficulty in difficulties:
        band = _difficulty_pool(pool, difficulty)
        chosen = choose_diverse(
            band,
            per_difficulty,
            excluded_level_ids=level_ids,
            excluded_song_keys=song_keys,
            excluded_creator_ids=excluded_creator_ids,
        )
        if len(chosen) != per_difficulty:
            return None
        selected.extend(chosen)
        level_ids.update(item.level_id for item in chosen)
        song_keys.update(item.song_key for item in chosen)

    return tuple(selected)


def select_balanced_dataset(
    candidates: Iterable[TufCandidate],
    *,
    min_p_difficulty: int = 7,
    max_p_difficulty: int = 10,
    train_per_difficulty: int = 16,
    validation_per_difficulty: int = 4,
    final_per_difficulty: int = 2,
) -> BalancedDatasetSelection:
    """Select exact per-P quotas with song-disjoint Train/Validation/Final roles.

    Final is selected first, then Validation, then Train. Creator identity is kept
    disjoint across roles when possible. If an exact quota is impossible under
    that creator constraint, only creator isolation is relaxed; level and song
    identity remain hard-disjoint across all roles and difficulties.
    """

    if min_p_difficulty <= 0 or max_p_difficulty < min_p_difficulty:
        raise ValueError("P difficulty range must satisfy 1 <= min <= max")
    if train_per_difficulty <= 0 or validation_per_difficulty <= 0 or final_per_difficulty <= 0:
        raise ValueError("per-difficulty quotas must be positive")

    pool = tuple(candidates)
    difficulties = tuple(range(min_p_difficulty, max_p_difficulty + 1))

    required = train_per_difficulty + validation_per_difficulty + final_per_difficulty
    for difficulty in difficulties:
        available = len(_difficulty_pool(pool, difficulty))
        if available < required:
            raise ValueError(
                f"not enough P{difficulty} candidates: {available} < {required} required"
            )

    used_ids: set[int] = set()
    used_songs: set[str] = set()

    final = _select_role(
        pool,
        difficulties=difficulties,
        per_difficulty=final_per_difficulty,
        excluded_level_ids=used_ids,
        excluded_song_keys=used_songs,
        excluded_creator_ids=set(),
    )
    if final is None:
        raise ValueError("not enough song-disjoint candidates for Final quotas")
    used_ids.update(item.level_id for item in final)
    used_songs.update(item.song_key for item in final)

    relaxed = False
    validation = _select_role(
        pool,
        difficulties=difficulties,
        per_difficulty=validation_per_difficulty,
        excluded_level_ids=used_ids,
        excluded_song_keys=used_songs,
        excluded_creator_ids=_creator_ids(final),
    )
    if validation is None:
        relaxed = True
        validation = _select_role(
            pool,
            difficulties=difficulties,
            per_difficulty=validation_per_difficulty,
            excluded_level_ids=used_ids,
            excluded_song_keys=used_songs,
            excluded_creator_ids=set(),
        )
    if validation is None:
        raise ValueError("not enough song-disjoint candidates for Validation quotas")
    used_ids.update(item.level_id for item in validation)
    used_songs.update(item.song_key for item in validation)

    train = _select_role(
        pool,
        difficulties=difficulties,
        per_difficulty=train_per_difficulty,
        excluded_level_ids=used_ids,
        excluded_song_keys=used_songs,
        excluded_creator_ids=_creator_ids((*final, *validation)),
    )
    if train is None:
        relaxed = True
        train = _select_role(
            pool,
            difficulties=difficulties,
            per_difficulty=train_per_difficulty,
            excluded_level_ids=used_ids,
            excluded_song_keys=used_songs,
            excluded_creator_ids=set(),
        )
    if train is None:
        raise ValueError("not enough song-disjoint candidates for Train quotas")

    return BalancedDatasetSelection(
        train=train,
        validation=validation,
        final=final,
        relaxed_creator_overlap=relaxed,
    )


def select_parser_compatible_balanced_dataset(
    candidates: Iterable[TufCandidate],
    *,
    min_p_difficulty: int = 7,
    max_p_difficulty: int = 10,
    train_per_difficulty: int = 16,
    validation_per_difficulty: int = 4,
    final_per_difficulty: int = 2,
    timeout_s: float = 60.0,
    progress: Callable[[str], None] | None = None,
) -> BalancedPreflightResult:
    """Re-select after rejecting any chart the current DMDOD parser cannot load."""

    pool = list(candidates)
    rejected: list[tuple[int, str]] = []
    raw_by_level_id: dict[int, bytes] = {}

    while True:
        selection = select_balanced_dataset(
            pool,
            min_p_difficulty=min_p_difficulty,
            max_p_difficulty=max_p_difficulty,
            train_per_difficulty=train_per_difficulty,
            validation_per_difficulty=validation_per_difficulty,
            final_per_difficulty=final_per_difficulty,
        )
        selected = (*selection.final, *selection.validation, *selection.train)
        bad_ids: set[int] = set()

        for candidate in selected:
            if candidate.level_id in raw_by_level_id:
                continue
            if progress is not None:
                progress(
                    f"preflight #{candidate.level_id} {candidate.song} "
                    f"({candidate.difficulty_name})"
                )
            try:
                raw = download_chart(candidate, timeout_s=timeout_s)
                parse_adofai_bytes(raw, source_path=f"TUF:{candidate.level_id}")
            except Exception as exc:  # network/format/parser failures all disqualify the chart
                reason = str(exc).strip() or type(exc).__name__
                rejected.append((candidate.level_id, reason))
                bad_ids.add(candidate.level_id)
                if progress is not None:
                    progress(f"reject #{candidate.level_id}: {reason}")
            else:
                raw_by_level_id[candidate.level_id] = raw

        if not bad_ids:
            return BalancedPreflightResult(
                selection=selection,
                rejected=tuple(rejected),
                raw_by_level_id=raw_by_level_id,
            )

        pool = [candidate for candidate in pool if candidate.level_id not in bad_ids]
        for level_id in bad_ids:
            raw_by_level_id.pop(level_id, None)


def _manifest_item(candidate: TufCandidate, path: str) -> dict:
    result = asdict(candidate)
    result["quality_score"] = round(candidate.quality_score, 6)
    result["ranked"] = candidate.ranked
    result["path"] = path
    return result


def build_balanced_dataset(
    selection: BalancedDatasetSelection,
    *,
    raw_by_level_id: dict[int, bytes],
    output: str | Path,
    metadata: dict | None = None,
    download_delay_s: float = 0.0,
    timeout_s: float = 60.0,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Write a three-way TUF dataset using preflight bytes when available."""

    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    partial = output_path.with_name(output_path.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"partial output already exists: {partial}")

    role_items = {
        "Train": selection.train,
        "Validation": selection.validation,
        "Final": selection.final,
    }
    written: dict[str, list[dict]] = {role: [] for role in role_items}

    try:
        for role, items in role_items.items():
            role_dir = partial / role
            role_dir.mkdir(parents=True, exist_ok=True)
            for index, candidate in enumerate(items, 1):
                if progress is not None:
                    progress(
                        f"{role} {index:02d}/{len(items):02d} "
                        f"#{candidate.level_id} {candidate.difficulty_name} {candidate.song}"
                    )
                raw = raw_by_level_id.get(candidate.level_id)
                if raw is None:
                    raw = download_chart(candidate, timeout_s=timeout_s)
                filename = chart_filename(candidate)
                (role_dir / filename).write_bytes(raw)
                written[role].append(
                    _manifest_item(candidate, str(Path(role) / filename))
                )
                if download_delay_s > 0.0 and candidate.level_id not in raw_by_level_id:
                    time.sleep(download_delay_s)

        manifest = {
            "format": 2,
            "source": "TUF API v2",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "selection": {
                "train_count": len(selection.train),
                "validation_count": len(selection.validation),
                "final_count": len(selection.final),
                "relaxed_creator_overlap": selection.relaxed_creator_overlap,
            },
            "metadata": metadata or {},
            "Train": written["Train"],
            "Validation": written["Validation"],
            "Final": written["Final"],
        }
        (partial / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        partial.rename(output_path)
    except Exception:
        # Keep partial data for inspection instead of silently deleting downloads.
        raise

    return output_path
