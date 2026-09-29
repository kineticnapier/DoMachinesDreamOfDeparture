from __future__ import annotations

import dmdod.tuf_dataset_preflight as preflight
from dmdod.tuf_dataset import TufCandidate


def _candidate(level_id: int, *, score_clears: int) -> TufCandidate:
    return TufCandidate(
        level_id=level_id,
        song=f"Song {level_id}",
        artist="Artist",
        creator=f"Creator {level_id}",
        creator_ids=(1000 + level_id,),
        song_id=2000 + level_id,
        bpm=180.0,
        tilecount=600,
        midspin_count=0,
        length_ms=120_000.0,
        unique_clears=score_clears,
        clears=score_clears,
        downloads=score_clears,
        curated=False,
        difficulty_id=5,
        difficulty_name="P5",
        file_id=f"file-{level_id}",
        download_url=f"https://api.tuforums.com/cdn/file-{level_id}",
    )


def test_preflight_rejects_pathdata_and_promotes_replacement(monkeypatch) -> None:
    candidates = [
        _candidate(1, score_clears=1000),
        _candidate(2, score_clears=900),
        _candidate(3, score_clears=800),
        _candidate(4, score_clears=700),
    ]

    def fake_download(candidate: TufCandidate, *, timeout_s: float = 60.0) -> bytes:
        if candidate.level_id == 1:
            return b'{"pathData":"RULD","settings":{"bpm":120}}'
        return b'{"angleData":[0,90,180],"settings":{"bpm":120}}'

    monkeypatch.setattr(preflight, "download_chart", fake_download)

    result = preflight.select_parser_compatible_dataset(
        candidates,
        train_count=1,
        validation_count=1,
    )

    selected_ids = {
        item.level_id for item in (*result.selection.validation, *result.selection.train)
    }
    rejected_ids = {level_id for level_id, _reason in result.rejected}

    assert 1 in rejected_ids
    assert 1 not in selected_ids
    assert len(selected_ids) == 2
