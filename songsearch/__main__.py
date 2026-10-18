"""Start the web app, preparing the full catalogue on first launch."""

import os
from pathlib import Path

import uvicorn

from .api import create_app
from .corpus import prepare
from .data import load_songs
from .engine import SearchEngine
from .evaluation import load_judgments


def main():
    data = Path("data/songs.jsonl")
    if not data.exists():
        prepare(data)
    songs = load_songs(data)
    engine = SearchEngine(songs, cache_dir=".cache/embeddings")
    judgments = load_judgments("data/judgments.json", {s.id for s in songs})
    uvicorn.run(
        create_app(engine, judgments), host="127.0.0.1", port=int(os.getenv("PORT", "8000"))
    )


if __name__ == "__main__":
    main()
