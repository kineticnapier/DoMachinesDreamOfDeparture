from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import dmdod.tuf_dataset_parallel as parallel


def _row(level_id: int) -> dict:
    return {
        "id": level_id,
        "song": f"Song {level_id}",
        "artist": "Artist",
        "creator": f"Creator {level_id}",
        "songId": level_id,
        "bpm": 180,
        "tilecount": 600,
        "midspinCount": 0,
        "levelLengthInMs": 120000,
        "uniqueClears": level_id,
        "clears": level_id,
        "downloadCount": 10,
        "isCurated": False,
        "fileId": f"file-{level_id}",
        "dlLink": f"https://api.tuforums.com/cdn/file-{level_id}",
        "difficulty": {"id": 10, "name": "D10"},
        "levelCredits": [
            {"role": "charter", "creator": {"id": 1000 + level_id}},
        ],
    }


def test_parallel_scan_uses_actual_first_page_size_as_stride(monkeypatch) -> None:
    calls: list[tuple[int, int]] = []

    def fake_request_json(url: str, *, timeout_s: float = 30.0, retries: int = 4) -> dict:
        query = parse_qs(urlparse(url).query)
        offset = int(query["offset"][0])
        limit = int(query["limit"][0])
        calls.append((offset, limit))

        # Simulate TUF clamping the oversized first request to two rows.
        if offset == 0:
            rows = [_row(1), _row(2)]
        elif offset == 2:
            rows = [_row(3), _row(4)]
        elif offset == 4:
            rows = [_row(5)]
        else:
            rows = []
        return {
            "results": rows,
            "total": 5,
            "hasMore": offset + len(rows) < 5,
        }

    monkeypatch.setattr(parallel, "_request_json", fake_request_json)

    result = parallel.fetch_candidates_parallel(
        page_size=500,
        workers=2,
        request_delay_s=0.0,
    )

    assert [item.level_id for item in result] == [1, 2, 3, 4, 5]
    assert (0, 500) in calls
    assert (2, 2) in calls
    assert (4, 1) in calls
