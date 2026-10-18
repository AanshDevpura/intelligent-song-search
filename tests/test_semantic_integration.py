"""Opt-in integration check against the actual pinned transformer (downloads once)."""

import os

import pytest

from songsearch.data import Song
from songsearch.engine import SearchEngine, SentenceEncoder


@pytest.mark.skipif(
    os.environ.get("SONGSEARCH_TEST_MODEL") != "1",
    reason="Set SONGSEARCH_TEST_MODEL=1 to test downloaded model weights",
)
def test_real_model_paraphrase_chunking_and_cache(tmp_path):
    encoder = SentenceEncoder()
    songs = [
        Song(
            "regret",
            "Empty Rooms",
            "A",
            "I wish I had never said goodbye. I miss the person I let go. "
            "I want to undo the day our love ended.",
        ),
        Song(
            "party",
            "Friday",
            "B",
            "Dancing together with friends, laughing and celebrating all night.",
        ),
        Song(
            "nature",
            "Orchard",
            "C",
            "Green leaves and apple trees grow in the warm summer sunshine.",
        ),
    ]
    engine = SearchEngine(songs, encoder=encoder, cache_dir=tmp_path)
    assert (
        engine.search("sadness and remorse over a failed romance", mode="semantic")[0]["id"]
        == "regret"
    )
    chunks = encoder.chunks("A long road through the mountains. " * 200)
    assert len(chunks) > 1
    assert all(
        len(encoder.model.tokenizer.encode(c)) <= encoder.model.max_seq_length for c in chunks
    )
    cached = SearchEngine(songs, encoder=encoder, cache_dir=tmp_path)
    assert cached.search(
        "sadness and remorse over a failed romance", mode="semantic"
    ) == engine.search("sadness and remorse over a failed romance", mode="semantic")
