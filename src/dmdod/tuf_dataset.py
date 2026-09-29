from __future__ import annotations

import io
import json
import math
import re
import shutil
import time
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .multichart_dataset import _choose_adofai_name


API_BASE = "https://api.tuforums.com"
LEVELS_URL = f"{API_BASE}/v2/database/levels"
USER_AGENT = "DMDOD-TUF-Dataset-Builder/0.1"
_WINDOWS_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass(frozen=True, slots=True)
class TufCandidate:
    level_id: int
    song: str
    artist: str
    creator: str
    creator_ids: tuple[int, ...]
    song_id: int | None
    bpm: float
    tilecount: int
    midspin_count: int
    length_ms: float
    unique_clears: int
    clears: int
    downloads: int
    curated: bool
    difficulty_id: int
    difficulty_name: str
    file_id: str
    download_url: str

    @property
    def ranked(self) -> bool:
        return self.difficulty_id != 0 and self.difficulty_name.casefold() != "unranked"

    @property
    def quality_score(self) -> float:
        # Dataset-quality heuristic, intentionally dominated by community evidence
        # rather than raw popularity alone. Curation is a large fixed bonus while
        # clear/download counts are logarithmic so one viral chart cannot swamp
        # the whole selection.
        return (
            (8.0 if self.curated else 0.0)
            + 4.0 * math.log1p(max(0, self.unique_clears))
            + 1.25 * math.log1p(max(0, self.clears))
            + 0.40 * math.log1p(max(0, self.downloads))
            + (1.0 if self.ranked else 0.0)
        )

    @property
    def song_key(self) -> str:
        if self.song_id is not None:
            return f"id:{self.song_id}"
        return f"text:{self.artist.casefold()}::{self.song.casefold()}"

    @property
    def bpm_bin(self) -> str:
        bpm = self.bpm
        if bpm < 100:
            return "<100"
        if bpm < 140:
            return "100-139"
        if bpm < 180:
            return "140-179"
        if bpm < 220:
            return "180-219"
        if bpm < 300:
            return "220-299"
        return "300+"

    @property
    def length_bin(self) -> str:
        seconds = self.length_ms / 1000.0
        if seconds < 60:
            return "<60s"
        if seconds < 120:
            return "60-119s"
        if seconds < 240:
            return "120-239s"
        return "240s+"

    @property
    def density_bin(self) -> str:
        seconds = max(1e-9, self.length_ms / 1000.0)
        rate = self.tilecount / seconds
        if rate < 4:
            return "<4tps"
        if rate < 7:
            return "4-7tps"
        if rate < 11:
            return "7-11tps"
        return "11+tps"


@dataclass(frozen=True, slots=True)
class DatasetSelection:
    train: tuple[TufCandidate, ...]
    validation: tuple[TufCandidate, ...]
    relaxed_creator_overlap: bool


def candidate_from_api(raw: dict) -> TufCandidate | None:
    try:
        level_id = int(raw["id"])
    except (KeyError, TypeError, ValueError):
        return None

    if bool(raw.get("isDeleted")) or bool(raw.get("isHidden")):
        return None

    file_id = str(raw.get("fileId") or "").strip()
    download_url = str(raw.get("dlLink") or "").strip()
    if not file_id and not download_url:
        return None

    try:
        bpm = float(raw.get("bpm") or 0.0)
        tilecount = int(raw.get("tilecount") or 0)
        length_ms = float(raw.get("levelLengthInMs") or 0.0)
    except (TypeError, ValueError):
        return None
    if bpm <= 0.0 or tilecount <= 0 or length_ms <= 0.0:
        return None

    credits = raw.get("levelCredits") or []
    creator_ids: list[int] = []
    for credit in credits:
        if not isinstance(credit, dict):
            continue
        role = str(credit.get("role") or "").casefold()
        if role not in {"charter", "vfxer"}:
            continue
        creator = credit.get("creator") or {}
        try:
            creator_id = int(creator.get("id", credit.get("creatorId")))
        except (TypeError, ValueError):
            continue
        creator_ids.append(creator_id)

    difficulty = raw.get("difficulty") or {}
    try:
        difficulty_id = int(difficulty.get("id", raw.get("diffId", 0)) or 0)
    except (TypeError, ValueError):
        difficulty_id = 0

    try:
        song_id_raw = raw.get("songId")
        song_id = None if song_id_raw is None else int(song_id_raw)
    except (TypeError, ValueError):
        song_id = None

    curated = bool(raw.get("isCurated") or raw.get("curation") or raw.get("curations"))
    return TufCandidate(
        level_id=level_id,
        song=str(raw.get("song") or "Untitled"),
        artist=str(raw.get("artist") or "Unknown artist"),
        creator=str(raw.get("creator") or "Unknown creator"),
        creator_ids=tuple(sorted(set(creator_ids))),
        song_id=song_id,
        bpm=bpm,
        tilecount=tilecount,
        midspin_count=int(raw.get("midspinCount") or 0),
        length_ms=length_ms,
        unique_clears=int(raw.get("uniqueClears") or 0),
        clears=int(raw.get("clears") or 0),
        downloads=int(raw.get("downloadCount") or 0),
        curated=curated,
        difficulty_id=difficulty_id,
        difficulty_name=str(difficulty.get("name") or "Unranked"),
        file_id=file_id,
        download_url=download_url,
    )


def _request_bytes(url: str, *, timeout_s: float = 30.0, retries: int = 4) -> bytes:
    request = Request(
        url,
        headers={
            "Accept": "application/json,text/plain,*/*",
            "User-Agent": USER_AGENT,
        },
    )
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            with urlopen(request, timeout=timeout_s) as response:
                return response.read()
        except HTTPError as exc:
            last_error = exc
            if exc.code not in {408, 429, 500, 502, 503, 504} or attempt + 1 >= retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            try:
                wait_s = float(retry_after) if retry_after is not None else 0.5 * (2**attempt)
            except ValueError:
                wait_s = 0.5 * (2**attempt)
            time.sleep(min(8.0, max(0.1, wait_s)))
        except URLError as exc:
            last_error = exc
            if attempt + 1 >= retries:
                raise
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    assert last_error is not None
    raise last_error


def _request_json(url: str, *, timeout_s: float = 30.0, retries: int = 4) -> dict:
    raw = _request_bytes(url, timeout_s=timeout_s, retries=retries)
    decoded = json.loads(raw.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"expected JSON object from {url}")
    return decoded


def fetch_candidates(
    *,
    page_size: int = 100,
    scan_limit: int = 0,
    min_unique_clears: int = 0,
    request_delay_s: float = 0.10,
    timeout_s: float = 30.0,
    progress: Callable[[int, int | None, int], None] | None = None,
) -> list[TufCandidate]:
    """Fetch anonymous TUF level metadata and return usable chart candidates.

    ``scan_limit=0`` scans the complete public level database. Selection happens
    locally so it does not depend on undocumented server-side sort semantics.
    """

    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if scan_limit < 0 or min_unique_clears < 0:
        raise ValueError("scan_limit/min_unique_clears must be non-negative")

    offset = 0
    scanned = 0
    total: int | None = None
    candidates: list[TufCandidate] = []
    seen_ids: set[int] = set()

    while True:
        remaining = scan_limit - scanned if scan_limit else page_size
        limit = min(page_size, remaining) if scan_limit else page_size
        if limit <= 0:
            break
        url = f"{LEVELS_URL}?{urlencode({'offset': offset, 'limit': limit})}"
        payload = _request_json(url, timeout_s=timeout_s)
        rows = payload.get("results") or []
        if not isinstance(rows, list):
            raise ValueError("TUF level search returned non-list results")
        try:
            total = int(payload.get("total")) if payload.get("total") is not None else total
        except (TypeError, ValueError):
            pass

        for raw in rows:
            if scan_limit and scanned >= scan_limit:
                break
            scanned += 1
            if not isinstance(raw, dict):
                continue
            candidate = candidate_from_api(raw)
            if candidate is None or candidate.level_id in seen_ids:
                continue
            if candidate.unique_clears < min_unique_clears:
                continue
            seen_ids.add(candidate.level_id)
            candidates.append(candidate)

        if progress is not None:
            progress(scanned, total, len(candidates))
        if scan_limit and scanned >= scan_limit:
            break
        if not rows or not bool(payload.get("hasMore")):
            break
        offset += len(rows)
        if request_delay_s > 0.0:
            time.sleep(request_delay_s)

    return candidates


def _pick_score(
    candidate: TufCandidate,
    *,
    bpm_counts: Counter[str],
    length_counts: Counter[str],
    density_counts: Counter[str],
    difficulty_counts: Counter[int],
    creator_counts: Counter[int],
) -> float:
    novelty = (
        1.25 / (1 + bpm_counts[candidate.bpm_bin])
        + 0.90 / (1 + length_counts[candidate.length_bin])
        + 0.70 / (1 + density_counts[candidate.density_bin])
        + 1.00 / (1 + difficulty_counts[candidate.difficulty_id])
    )
    creator_repeat = max((creator_counts[value] for value in candidate.creator_ids), default=0)
    return candidate.quality_score + novelty - 1.35 * creator_repeat


def choose_diverse(
    candidates: Iterable[TufCandidate],
    count: int,
    *,
    excluded_level_ids: set[int] | None = None,
    excluded_song_keys: set[str] | None = None,
    excluded_creator_ids: set[int] | None = None,
) -> tuple[TufCandidate, ...]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return ()

    excluded_level_ids = excluded_level_ids or set()
    excluded_song_keys = excluded_song_keys or set()
    excluded_creator_ids = excluded_creator_ids or set()
    remaining = {
        item.level_id: item
        for item in candidates
        if item.level_id not in excluded_level_ids
        and item.song_key not in excluded_song_keys
        and not set(item.creator_ids).intersection(excluded_creator_ids)
    }

    chosen: list[TufCandidate] = []
    bpm_counts: Counter[str] = Counter()
    length_counts: Counter[str] = Counter()
    density_counts: Counter[str] = Counter()
    difficulty_counts: Counter[int] = Counter()
    creator_counts: Counter[int] = Counter()
    used_song_keys: set[str] = set()

    while len(chosen) < count:
        eligible = [item for item in remaining.values() if item.song_key not in used_song_keys]
        if not eligible:
            break
        best = max(
            eligible,
            key=lambda item: (
                _pick_score(
                    item,
                    bpm_counts=bpm_counts,
                    length_counts=length_counts,
                    density_counts=density_counts,
                    difficulty_counts=difficulty_counts,
                    creator_counts=creator_counts,
                ),
                item.curated,
                item.unique_clears,
                item.clears,
                item.downloads,
                -item.level_id,
            ),
        )
        chosen.append(best)
        remaining.pop(best.level_id, None)
        used_song_keys.add(best.song_key)
        bpm_counts[best.bpm_bin] += 1
        length_counts[best.length_bin] += 1
        density_counts[best.density_bin] += 1
        difficulty_counts[best.difficulty_id] += 1
        for creator_id in best.creator_ids:
            creator_counts[creator_id] += 1

    return tuple(chosen)


def select_dataset(
    candidates: Iterable[TufCandidate],
    *,
    train_count: int,
    validation_count: int,
) -> DatasetSelection:
    pool = tuple(candidates)
    if train_count <= 0 or validation_count <= 0:
        raise ValueError("train_count and validation_count must be positive")

    validation = choose_diverse(pool, validation_count)
    if len(validation) != validation_count:
        raise ValueError(f"not enough candidates for Validation: {len(validation)}/{validation_count}")

    validation_ids = {item.level_id for item in validation}
    validation_songs = {item.song_key for item in validation}
    validation_creators = {creator_id for item in validation for creator_id in item.creator_ids}
    train = choose_diverse(
        pool,
        train_count,
        excluded_level_ids=validation_ids,
        excluded_song_keys=validation_songs,
        excluded_creator_ids=validation_creators,
    )
    relaxed = False
    if len(train) != train_count:
        # With a tiny/capped metadata scan the strict creator split can make the
        # requested size impossible. Song identity remains a hard split; creator
        # isolation is the only constraint that may be relaxed, and we record it.
        relaxed = True
        train = choose_diverse(
            pool,
            train_count,
            excluded_level_ids=validation_ids,
            excluded_song_keys=validation_songs,
        )
    if len(train) != train_count:
        raise ValueError(f"not enough candidates for Train: {len(train)}/{train_count}")
    return DatasetSelection(train=train, validation=validation, relaxed_creator_overlap=relaxed)


def sanitize_filename(value: str, *, max_length: int = 140) -> str:
    cleaned = _WINDOWS_INVALID.sub("_", value).strip(" .")
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        cleaned = "chart"
    return cleaned[:max_length].rstrip(" .") or "chart"


def chart_filename(candidate: TufCandidate) -> str:
    stem = sanitize_filename(f"{candidate.level_id:05d} - {candidate.song} - {candidate.creator}")
    return f"{stem}.adofai"


def _looks_like_adofai(raw: bytes) -> bool:
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(value, dict) and (
        isinstance(value.get("angleData"), list) or isinstance(value.get("pathData"), str)
    )


def download_chart(
    candidate: TufCandidate,
    *,
    timeout_s: float = 60.0,
) -> bytes:
    direct_url = f"{API_BASE}/v2/database/levels/{candidate.level_id}/level.adofai"
    try:
        raw = _request_bytes(direct_url, timeout_s=timeout_s)
        if _looks_like_adofai(raw):
            return raw
    except (HTTPError, URLError, ValueError):
        pass

    if not candidate.download_url:
        raise ValueError(f"level {candidate.level_id} has no usable chart download")
    raw_zip = _request_bytes(candidate.download_url, timeout_s=timeout_s)
    try:
        with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
            names = [name for name in archive.namelist() if name.casefold().endswith(".adofai")]
            chosen = _choose_adofai_name(names)
            raw = archive.read(chosen)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"TUF level {candidate.level_id} fallback is not a valid ZIP") from exc
    if not _looks_like_adofai(raw):
        raise ValueError(f"TUF level {candidate.level_id} downloaded invalid .adofai data")
    return raw


def copy_final_source(source: str | Path, destination: Path) -> int:
    source_path = Path(source).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0

    if source_path.is_dir():
        root = source_path / "Final" if (source_path / "Final").is_dir() else source_path
        for item in root.iterdir():
            target = destination / item.name
            if item.is_dir():
                shutil.copytree(item, target)
                copied += 1
            elif item.suffix.casefold() in {".adofai", ".zip"}:
                shutil.copy2(item, target)
                copied += 1
        if copied == 0:
            raise ValueError(f"Final source contains no chart items: {source_path}")
        return copied

    if source_path.is_file() and source_path.suffix.casefold() == ".zip":
        with zipfile.ZipFile(source_path) as archive:
            groups: set[str] = set()
            for info in archive.infolist():
                if info.is_dir():
                    continue
                parts = [part for part in info.filename.replace("\\", "/").split("/") if part]
                lowered = [part.casefold() for part in parts]
                if "final" not in lowered:
                    continue
                index = lowered.index("final")
                relative_parts = parts[index + 1 :]
                if not relative_parts:
                    continue
                relative = Path(*relative_parts)
                if any(part in {"..", "."} for part in relative.parts):
                    continue
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(info))
                groups.add(relative.parts[0])
            copied = len(groups)
        if copied == 0:
            raise ValueError(f"dataset ZIP contains no Final/ charts: {source_path}")
        return copied

    raise ValueError(f"--final-source must be a dataset directory, Final directory, or .zip: {source_path}")


def _candidate_manifest(candidate: TufCandidate, path: str) -> dict:
    result = asdict(candidate)
    result["quality_score"] = round(candidate.quality_score, 6)
    result["ranked"] = candidate.ranked
    result["path"] = path
    return result


def build_dataset(
    selection: DatasetSelection,
    *,
    output: str | Path,
    final_source: str | Path,
    download_delay_s: float = 0.15,
    timeout_s: float = 60.0,
    metadata: dict | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")

    partial = output_path.with_name(output_path.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"partial output already exists: {partial}")
    train_dir = partial / "Train"
    validation_dir = partial / "Validation"
    final_dir = partial / "Final"
    train_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)
    final_dir.mkdir(parents=True)

    written: dict[str, list[dict]] = {"Train": [], "Validation": []}
    try:
        for role, items, role_dir in (
            ("Validation", selection.validation, validation_dir),
            ("Train", selection.train, train_dir),
        ):
            for index, candidate in enumerate(items, 1):
                if progress is not None:
                    progress(
                        f"{role} {index:02d}/{len(items):02d} "
                        f"#{candidate.level_id} {candidate.song} "
                        f"curated={candidate.curated} uniqueClears={candidate.unique_clears}"
                    )
                raw = download_chart(candidate, timeout_s=timeout_s)
                filename = chart_filename(candidate)
                destination = role_dir / filename
                destination.write_bytes(raw)
                written[role].append(_candidate_manifest(candidate, str(Path(role) / filename)))
                if download_delay_s > 0.0:
                    time.sleep(download_delay_s)

        final_count = copy_final_source(final_source, final_dir)
        manifest = {
            "format": 1,
            "source": "TUF API v2",
            "api_base": API_BASE,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "selection": {
                "train_count": len(selection.train),
                "validation_count": len(selection.validation),
                "relaxed_creator_overlap": selection.relaxed_creator_overlap,
            },
            "metadata": metadata or {},
            "Train": written["Train"],
            "Validation": written["Validation"],
            "Final": {"copied_from": str(final_source), "chart_items": final_count},
        }
        (partial / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        partial.rename(output_path)
    except Exception:
        # Keep .partial for post-mortem/resume-by-hand instead of deleting data.
        raise
    return output_path
