from __future__ import annotations

from collections import Counter

from dmdod.tuf_balanced_dataset import select_balanced_dataset
from dmdod.tuf_dataset import TufCandidate


def _candidate(level_id: int, difficulty: int, index: int) -> TufCandidate:
    return TufCandidate(
        level_id=level_id,
        song=f"P{difficulty} Song {index}",
        artist="Artist",
        creator=f"Creator {level_id}",
        creator_ids=(100000 + level_id,),
        song_id=200000 + level_id,
        bpm=140.0 + index,
        tilecount=800 + index,
        midspin_count=index % 5,
        length_ms=120000.0 + index * 1000.0,
        unique_clears=100 - index,
        clears=200 - index,
        downloads=500 - index,
        curated=index % 7 == 0,
        difficulty_id=difficulty,
        difficulty_name=f"P{difficulty}",
        file_id=f"file-{level_id}",
        download_url=f"https://example.invalid/{level_id}.zip",
    )


def test_balanced_p7_p10_selection_has_exact_role_quotas() -> None:
    candidates = []
    level_id = 1
    for difficulty in range(7, 11):
        for index in range(30):
            candidates.append(_candidate(level_id, difficulty, index))
            level_id += 1

    selection = select_balanced_dataset(candidates)

    assert len(selection.train) == 64
    assert len(selection.validation) == 16
    assert len(selection.final) == 8
    assert not selection.relaxed_creator_overlap

    for items, expected in (
        (selection.train, 16),
        (selection.validation, 4),
        (selection.final, 2),
    ):
        counts = Counter(item.difficulty_name for item in items)
        assert counts == {"P7": expected, "P8": expected, "P9": expected, "P10": expected}

    train_ids = {item.level_id for item in selection.train}
    validation_ids = {item.level_id for item in selection.validation}
    final_ids = {item.level_id for item in selection.final}
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(final_ids)
    assert validation_ids.isdisjoint(final_ids)

    train_songs = {item.song_key for item in selection.train}
    validation_songs = {item.song_key for item in selection.validation}
    final_songs = {item.song_key for item in selection.final}
    assert train_songs.isdisjoint(validation_songs)
    assert train_songs.isdisjoint(final_songs)
    assert validation_songs.isdisjoint(final_songs)
