from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable
from urllib.parse import urlencode

from .tuf_dataset import LEVELS_URL, TufCandidate, _request_json, candidate_from_api


MAX_P_DIFFICULTY = 6
_P_DIFFICULTY_RE = re.compile(r"^P([1-9][0-9]*)$", re.IGNORECASE)


def is_v090_difficulty_eligible(candidate: TufCandidate) -> bool:
    """Return whether a TUF chart is in the v0.9 curriculum band.

    v0.9 intentionally trains only on ordinary P difficulties up through P6.
    G/U/special/unranked charts and P7+ are excluded before quality/diversity
    selection so extreme charts cannot consume training slots just for novelty.
    """

    match = _P_DIFFICULTY_RE.fullmatch(candidate.difficulty_name.strip())
    return match is not None and 1 <= int(match.group(1)) <= MAX_P_DIFFICULTY


def _rows_from_payload(payload: dict) -> list[dict]:
    rows = payload.get("results") or []
    if not isinstance(rows, list):
        raise ValueError("TUF level search returned non-list results")
    return [row for row in rows if isinstance(row, dict)]


def _payload_total(payload: dict) -> int | None:
    try:
        value = payload.get("total")
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def fetch_candidates_parallel(
    *,
    page_size: int = 500,
    workers: int = 6,
    scan_limit: int = 0,
    min_unique_clears: int = 0,
    request_delay_s: float = 0.02,
    timeout_s: float = 30.0,
    progress: Callable[[int, int | None, int], None] | None = None,
) -> list[TufCandidate]:
    """Fetch TUF level metadata concurrently while respecting server page clamps.

    The first request is intentionally oversized. TUF currently clamps large
    ``limit`` values, so the actual number of rows returned by that request is
    treated as the effective page size. Remaining offsets are then fetched in
    parallel. Selection still happens locally because server-side sort/facet
    parameters are not reliable enough for dataset construction.

    Only P1-P6 charts enter the returned candidate pool. This curriculum gate is
    applied before clear-count and curation ranking.
    """

    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if scan_limit < 0 or min_unique_clears < 0:
        raise ValueError("scan_limit/min_unique_clears must be non-negative")
    if request_delay_s < 0.0:
        raise ValueError("request_delay_s must be non-negative")

    first_limit = min(page_size, scan_limit) if scan_limit else page_size
    first_url = f"{LEVELS_URL}?{urlencode({'offset': 0, 'limit': first_limit})}"
    first_payload = _request_json(first_url, timeout_s=timeout_s)
    first_rows = _rows_from_payload(first_payload)
    server_total = _payload_total(first_payload)

    if not first_rows:
        if progress is not None:
            progress(0, server_total, 0)
        return []

    # If TUF clamps limit=500/1000 to 100, len(first_rows) tells us the real
    # stride. Using the requested limit here would skip most of the database.
    effective_page_size = len(first_rows)
    if server_total is None:
        target_total = scan_limit if scan_limit else effective_page_size
    else:
        target_total = min(server_total, scan_limit) if scan_limit else server_total

    seen_ids: set[int] = set()
    candidates: list[TufCandidate] = []
    scanned = 0

    def consume(rows: list[dict], remaining: int | None = None) -> None:
        nonlocal scanned
        if remaining is not None:
            rows = rows[:remaining]
        for raw in rows:
            scanned += 1
            candidate = candidate_from_api(raw)
            if candidate is None or candidate.level_id in seen_ids:
                continue
            if not is_v090_difficulty_eligible(candidate):
                continue
            if candidate.unique_clears < min_unique_clears:
                continue
            seen_ids.add(candidate.level_id)
            candidates.append(candidate)

    consume(first_rows, target_total)
    if progress is not None:
        progress(scanned, target_total, len(candidates))

    if scanned >= target_total or not bool(first_payload.get("hasMore")):
        return sorted(candidates, key=lambda item: item.level_id)

    offsets = list(range(effective_page_size, target_total, effective_page_size))

    def fetch_page(offset: int) -> tuple[int, list[dict]]:
        if request_delay_s > 0.0:
            # A small per-request pause avoids an instantaneous burst while still
            # allowing independent HTTP requests to overlap substantially.
            time.sleep(request_delay_s)
        limit = min(effective_page_size, target_total - offset)
        url = f"{LEVELS_URL}?{urlencode({'offset': offset, 'limit': limit})}"
        payload = _request_json(url, timeout_s=timeout_s)
        return offset, _rows_from_payload(payload)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tuf-scan") as executor:
        futures = {executor.submit(fetch_page, offset): offset for offset in offsets}
        for future in as_completed(futures):
            _offset, rows = future.result()
            remaining = max(0, target_total - scanned)
            if remaining == 0:
                continue
            consume(rows, remaining)
            if progress is not None:
                progress(scanned, target_total, len(candidates))

    # Completion order is nondeterministic; stabilize the candidate pool so any
    # exact score ties in later selection are reproducible.
    return sorted(candidates, key=lambda item: item.level_id)
