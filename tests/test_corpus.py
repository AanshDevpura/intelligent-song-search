import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

from songsearch import corpus
from songsearch.data import load_songs
from songsearch.evaluation import load_judgments


def source_fixture(tmp_path, monkeypatch, rows):
    raw = tmp_path / "raw.csv"
    with raw.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["artist", "song", "link", "text"])
        writer.writeheader()
        writer.writerows(rows)
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {"sha256": hashlib.sha256(raw.read_bytes()).hexdigest(), "expected_rows": len(rows)}
        )
    )
    monkeypatch.setattr(corpus, "SOURCE", source)
    return raw, tmp_path / "catalogue.jsonl"


def test_full_import_is_deterministic_and_reports_rejections(tmp_path, monkeypatch):
    row = dict(
        artist="A", song="Song", link="/a/a/song.html", text="original lyrics\r\nsecond line"
    )
    raw, output = source_fixture(
        tmp_path,
        monkeypatch,
        [row, {**row, "song": "Cover", "link": "/cover"}, {**row, "text": ""}],
    )
    report = corpus.prepare(output, raw_path=raw)
    first = output.read_bytes()
    assert report["source_rows"] == 3
    assert report["imported_rows"] == 2
    assert report["duplicate_lyrics_preserved"] == 1
    assert report["rejected_rows"] == {"missing_artist_title_or_lyrics": 1}
    assert all(s.lyrics and not s.description and not s.tags for s in load_songs(output))
    assert corpus.prepare(output, raw_path=raw)["normalized_sha256"] == report["normalized_sha256"]
    assert first == output.read_bytes()


def test_checksum_failure_preserves_catalogue(tmp_path, monkeypatch):
    raw, output = source_fixture(tmp_path, monkeypatch, [])
    output.write_text("previous catalogue")
    raw.write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        corpus.prepare(output, raw_path=raw)
    assert output.read_text() == "previous catalogue"


def test_duplicate_identity_preserves_catalogue(tmp_path, monkeypatch):
    row = dict(artist="A", song="Song", link="/a", text="lyrics")
    raw, output = source_fixture(tmp_path, monkeypatch, [row, row])
    output.write_text("previous catalogue")
    with pytest.raises(ValueError, match="Duplicate"):
        corpus.prepare(output, raw_path=raw)
    assert output.read_text() == "previous catalogue"


@pytest.mark.skipif(
    os.environ.get("SONGSEARCH_TEST_CORPUS") != "1",
    reason="Opt-in check of the complete downloaded lyrics dataset",
)
def test_entire_corpus_and_judgments():
    root = Path(__file__).resolve().parents[1]
    songs = load_songs(root / "data/songs.jsonl")
    assert len(songs) == 57650
    assert len({s.id for s in songs}) == 57650
    assert all(s.lyrics and not s.description and not s.tags for s in songs)
    cases = load_judgments(root / "data/judgments.json", {s.id for s in songs})
    assert len(cases) == 25
    assert all(len(c["relevant"]) == 10 for c in cases)
