from __future__ import annotations

import io
import zipfile

import pytest
import dmdod.data.tuf as tuf_module
from pathlib import Path

from dmdod.tuf_dataset import (
    TufCandidate,
    candidate_from_api,
    copy_final_source,
    sanitize_filename,
    select_dataset,
)


def _candidate(
    level_id: int,
    *,
    curated: bool = False,
    unique_clears: int = 10,
    clears: int = 12,
    creator_id: int | None = None,
    song_id: int | None = None,
    bpm: float = 180.0,
    length_ms: float = 120_000.0,
    difficulty_id: int = 10,
) -> TufCandidate:
    creator_id = level_id if creator_id is None else creator_id
    song_id = level_id if song_id is None else song_id
    return TufCandidate(
        level_id=level_id,
        song=f"Song {song_id}",
        artist=f"Artist {song_id}",
        creator=f"Creator {creator_id}",
        creator_ids=(creator_id,),
        song_id=song_id,
        bpm=bpm,
        tilecount=600,
        midspin_count=0,
        length_ms=length_ms,
        unique_clears=unique_clears,
        clears=clears,
        downloads=100,
        curated=curated,
        difficulty_id=difficulty_id,
        difficulty_name=f"D{difficulty_id}",
        file_id=f"file-{level_id}",
        download_url=f"https://api.tuforums.com/cdn/file-{level_id}",
    )


def test_quality_score_prioritizes_curation_and_clear_evidence():
    base = _candidate(1, curated=False, unique_clears=10)
    curated = _candidate(2, curated=True, unique_clears=10)
    more_clears = _candidate(3, curated=False, unique_clears=100)

    assert curated.quality_score > base.quality_score
    assert more_clears.quality_score > base.quality_score


def test_selection_keeps_validation_song_and_creator_disjoint_from_train():
    pool = [
        _candidate(
            index,
            curated=index % 2 == 0,
            unique_clears=10 + index,
            creator_id=100 + index,
            song_id=200 + index,
            bpm=90.0 + index * 35.0,
            length_ms=45_000.0 + index * 40_000.0,
            difficulty_id=index,
        )
        for index in range(1, 10)
    ]

    selection = select_dataset(pool, train_count=5, validation_count=2)
    assert len(selection.validation) == 2
    assert len(selection.train) == 5
    assert not selection.relaxed_creator_overlap

    validation_songs = {item.song_key for item in selection.validation}
    train_songs = {item.song_key for item in selection.train}
    validation_creators = {value for item in selection.validation for value in item.creator_ids}
    train_creators = {value for item in selection.train for value in item.creator_ids}
    assert validation_songs.isdisjoint(train_songs)
    assert validation_creators.isdisjoint(train_creators)


def test_candidate_from_api_detects_curation_and_metadata():
    raw = {
        "id": 14370,
        "song": "Arche",
        "artist": "Camellia",
        "creator": "Slime0205 & Geon Pi",
        "songId": 351,
        "bpm": 200,
        "tilecount": 4075,
        "midspinCount": 64,
        "levelLengthInMs": 307468.756,
        "uniqueClears": 25,
        "clears": 30,
        "downloadCount": 100,
        "isCurated": True,
        "fileId": "abc",
        "dlLink": "https://api.tuforums.com/cdn/abc",
        "difficulty": {"id": 10010, "name": "UQ0 (U1~U4)"},
        "levelCredits": [
            {"role": "charter", "creator": {"id": 2624}},
            {"role": "vfxer", "creator": {"id": 2257}},
        ],
    }
    item = candidate_from_api(raw)
    assert item is not None
    assert item.curated
    assert item.creator_ids == (2257, 2624)
    assert item.song_id == 351
    assert item.bpm == 200.0
    assert item.tilecount == 4075
    assert item.ranked


def test_sanitize_filename_is_windows_safe():
    value = sanitize_filename('  123: A/B*? <chart> | "x".  ')
    assert value == "123_ A_B__ _chart_ _ _x_"
    assert not any(character in value for character in '<>:"/\\|?*')


def test_copy_final_source_extracts_final_from_bundle_zip(tmp_path: Path):
    bundle = tmp_path / "old-dataset.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("DMDOD/Train/A/level.adofai", "train")
        archive.writestr("DMDOD/Final/Celica/level.adofai", "celica")
        archive.writestr("DMDOD/Final/Aftershine/level.adofai", "aftershine")

    destination = tmp_path / "Final"
    copied = copy_final_source(bundle, destination)
    assert copied == 2
    assert (destination / "Celica" / "level.adofai").read_text() == "celica"
    assert (destination / "Aftershine" / "level.adofai").read_text() == "aftershine"


@pytest.mark.parametrize("raw", [
    b'{"angleData":[0,90],"settings":{"bpm":120}}',
    b'{"angleData":[0,90,],"settings":{"bpm":120,},}',
    b'{// comment\\n"angleData":[0,90]}',
    b'{"angleData":[0],"settings":{"songFilename":"line1\\nline2.ogg"}}'.replace(b"\\\\n", b"\\n"),
    b"\\xff\\xfe" + '{"angleData":[0,90,]}'.encode("utf-16-le"),
    b"\\xfe\\xff" + '{"angleData":[0,90,]}'.encode("utf-16-be"),
])
def test_download_chart_accepts_every_parser_supported_document_encoding(monkeypatch, raw):
    # A valid direct response must not be discarded in favor of the ZIP URL.
    candidate = _candidate(501)
    requested = []

    def fake_request(url, *, timeout_s):
        requested.append(url)
        if url != f"{tuf_module.API_BASE}/v2/database/levels/501/level.adofai":
            raise AssertionError("unexpected fallback download")
        return raw

    monkeypatch.setattr(tuf_module, "_request_bytes", fake_request)
    assert tuf_module.download_chart(candidate) == raw
    assert len(requested) == 1


def test_download_chart_accepts_loose_json_inside_zip(monkeypatch):
    candidate = _candidate(502)
    chart = b"\\xff\\xfe" + '{/*note*/"angleData":[0,90,]}'.encode("utf-16-le")
    archive_data = io.BytesIO()
    with zipfile.ZipFile(archive_data, "w") as archive:
        archive.writestr("level.adofai", chart)
    requested = []

    def fake_request(url, *, timeout_s):
        requested.append(url)
        return b"not a chart" if url.endswith("/level.adofai") else archive_data.getvalue()

    monkeypatch.setattr(tuf_module, "_request_bytes", fake_request)
    assert tuf_module.download_chart(candidate) == chart
    assert len(requested) == 2


def test_download_chart_preserves_legacy_pathdata_for_preflight():
    from dmdod.adofai.chart import parse_adofai_bytes

    legacy = b'{"pathData":"RULD"}'
    assert tuf_module._looks_like_adofai(legacy)
    with pytest.raises(ValueError, match="legacy pathData"):
        parse_adofai_bytes(legacy)


@pytest.mark.parametrize("raw", [
    b"not JSON",
    b'{"angleData":null}',
    b'{"settings":{"bpm":120}}',
    b'{"pathData":123}',
    b"[]",
])
def test_download_chart_rejects_invalid_document_shapes(raw):
    assert not tuf_module._looks_like_adofai(raw)
