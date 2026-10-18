"""Reproducible full-corpus download and deterministic import using the standard library."""

import csv
import hashlib
import json
import os
import tempfile
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

SOURCE = Path(__file__).resolve().parent / "dataset-source.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(destination: str | Path, *, raw_path: str | Path | None = None) -> dict:
    """Import every source row; record invalid rows without inserting invented text/metadata."""
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix != ".jsonl":
        raise ValueError("Full-corpus destination must be a .jsonl file")
    raw = Path(raw_path) if raw_path else destination.parent / "raw" / "song_data.csv"
    if not raw.exists():
        if raw_path:
            raise ValueError(f"Raw dataset does not exist: {raw}")
        raw.parent.mkdir(parents=True, exist_ok=True)
        temporary = raw.with_suffix(".download")
        try:
            with (
                urllib.request.urlopen(source["url"], timeout=120) as response,
                temporary.open("wb") as out,
            ):
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            if sha256_file(temporary) != source["sha256"]:
                raise ValueError("Dataset checksum mismatch; download was not imported")
            temporary.replace(raw)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256_file(raw) != source["sha256"]:
        raise ValueError("Raw dataset checksum does not match the pinned source")
    fd, name = tempfile.mkstemp(dir=destination.parent, suffix=".jsonl")
    rejected = Counter()
    seen, artists, lyric_hashes = set(), set(), set()
    total = imported = 0
    try:
        with (
            os.fdopen(fd, "w", encoding="utf-8") as out,
            raw.open(encoding="utf-8-sig", newline="") as stream,
        ):
            reader = csv.DictReader(stream)
            if not {"artist", "song", "link", "text"} <= set(reader.fieldnames or []):
                raise ValueError("Unexpected dataset schema")
            for row in reader:
                total += 1
                artist, title, lyrics = (row[key].strip() for key in ("artist", "song", "text"))
                if not all((artist, title, lyrics)):
                    rejected["missing_artist_title_or_lyrics"] += 1
                    continue
                path = row["link"].strip()
                url = "https://www.lyricsfreak.com/" + path.lstrip("/")
                if urllib.parse.urlsplit(url).hostname != "www.lyricsfreak.com":
                    raise ValueError("Unexpected corpus source link")
                identity = "\0".join((artist.casefold(), title.casefold(), path))
                sid = "lf-" + hashlib.sha256(identity.encode()).hexdigest()[:20]
                if sid in seen:
                    raise ValueError(f"Duplicate source identity at row {total}")
                lyrics = lyrics.replace("\r\n", "\n").replace("\r", "\n")
                record = dict(
                    id=sid,
                    title=title,
                    artist=artist,
                    lyrics=lyrics,
                    source_url=url,
                )
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                seen.add(sid)
                artists.add(artist)
                lyric_hashes.add(hashlib.sha256(lyrics.encode()).hexdigest())
                imported += 1
        if total != source["expected_rows"] or not imported:
            raise ValueError(f"Unexpected source row count: {total}")
        Path(name).replace(destination)
    finally:
        Path(name).unlink(missing_ok=True)
    report = {
        **source,
        "source_rows": total,
        "imported_rows": imported,
        "rejected_rows": dict(rejected),
        "unique_artists": len(artists),
        "unique_lyrics": len(lyric_hashes),
        "duplicate_lyrics_preserved": imported - len(lyric_hashes),
        "normalized_sha256": sha256_file(destination),
        "indexed_fields": ["title", "artist", "lyrics"],
        "missing_metadata": ["genre", "year"],
        "description_policy": "No generated summaries or theme labels in the retrieval index.",
    }
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
