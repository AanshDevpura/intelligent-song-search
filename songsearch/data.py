"""Strict, deterministic CSV/JSON/JSONL ingestion. Never silently drop bad records."""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Song:
    id: str
    title: str
    artist: str
    lyrics: str = ""
    source_url: str = ""

    @property
    def text(self) -> str:
        return "\n".join((self.title, self.artist, self.lyrics))

    def public(self) -> dict:
        return asdict(self)


def load_songs(path: str | Path) -> list[Song]:
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if path.suffix.lower() == ".csv":
            rows = list(csv.DictReader(stream))
        elif path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in stream if line.strip()]
        elif path.suffix.lower() == ".json":
            rows = json.load(stream)
        else:
            raise ValueError("Dataset must be .csv, .json or .jsonl")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Dataset must be a nonempty list of song objects")
    songs, seen = [], set()
    allowed = set(Song.__dataclass_fields__)
    for i, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Record {i}: expected an object")
        unknown = set(row) - allowed
        if unknown:
            raise ValueError(f"Record {i}: unknown fields {sorted(unknown)}")
        values = {}
        for key in allowed:
            value = row.get(key, "")
            if not isinstance(value, str):
                raise ValueError(f"Record {i}: {key} must be text")
            values[key] = value.strip()
        for key in ("id", "title", "artist"):
            if not values[key]:
                raise ValueError(f"Record {i}: {key} is required")
        if not values["lyrics"]:
            raise ValueError(f"Record {i}: lyrics is required")
        if values["source_url"]:
            source = urlsplit(values["source_url"])
            if source.scheme != "https" or not source.netloc or source.username or source.password:
                raise ValueError(f"Record {i}: source_url must be a public HTTPS URL")
        if values["id"] in seen:
            raise ValueError(f"Record {i}: duplicate id {values['id']!r}")
        songs.append(Song(**values))
        seen.add(values["id"])
    return songs
