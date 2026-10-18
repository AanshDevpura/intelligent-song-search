import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from songsearch.api import create_app
from songsearch.data import Song, load_songs
from songsearch.engine import SearchEngine
from songsearch.evaluation import evaluate, load_judgments, metrics


@pytest.fixture
def songs():
    return [
        Song("sad", "Farewell", "Ada", "I regret leaving you alone", "Folk", 2020),
        Song("joy", "Reunion", "Ben", "We dance together with friends", "Pop", 2023),
        Song("old", "Winter", "Ada", "Dark days without you", "Folk"),
    ]


class Encoder:
    """A controlled model: 'sorrow' is close to regret without lexical overlap."""

    identity = "test-model@1"

    def __init__(self):
        self.calls = 0

    def chunks(self, text):
        return [text]

    def encode(self, texts):
        self.calls += 1
        return np.array(
            [
                [1, 0, 0]
                if "regret" in t or "sorrow" in t
                else [0, 1, 0]
                if "dance" in t
                else [0, 0, 1]
                for t in texts
            ],
            dtype=np.float32,
        )


def test_keyword_order_and_unknown_query(songs):
    engine = SearchEngine(songs, backend="keyword")
    assert engine.search("regret leaving", mode="keyword")[0]["id"] == "sad"
    assert engine.search("qwertyzz", mode="keyword") == []
    assert engine.search("the song about", mode="keyword") == []
    with pytest.raises(ValueError, match="require"):
        engine.search("love")


def test_semantic_retrieves_without_overlap(songs):
    engine = SearchEngine(songs, encoder=Encoder())
    result = engine.search("sorrow", mode="semantic")
    assert result[0]["id"] == "sad"
    assert result[0]["keyword_score"] == 0
    assert result[0]["semantic_score"] == 1
    assert engine.search("sorrow", alpha=1)[0]["id"] == "sad"
    assert engine.search("sorrow", alpha=0) == []


def test_hybrid_fusion_endpoints_and_determinism(songs):
    engine = SearchEngine(songs, encoder=Encoder())
    for mode, alpha in [("keyword", 0), ("semantic", 1)]:
        expected = [r["id"] for r in engine.search("regret dance", mode=mode)]
        assert [r["id"] for r in engine.search("regret dance", alpha=alpha)] == expected
    assert engine.search("regret dance") == engine.search("regret dance")


def test_filters_and_unknown_year(songs):
    engine = SearchEngine(songs, backend="keyword")
    assert engine.search("you", mode="keyword") == []  # stop word
    assert engine.search("regret", mode="keyword", genre="POP") == []
    assert engine.search("regret", mode="keyword", artist="ad", year_from=2010)[0]["id"] == "sad"
    assert engine.search("Dark", mode="keyword", year_to=2025) == []
    assert engine.search("Dark", mode="keyword", genre="FOLK")[0]["id"] == "old"
    with pytest.raises(ValueError):
        engine.search("regret", mode="keyword", year_from=2024, year_to=2000)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query": " "},
        {"query": "x" * 2001},
        {"query": "x", "k": 0},
        {"query": "x", "mode": "bad"},
        {"query": "x", "alpha": float("nan")},
    ],
)
def test_invalid_search(songs, kwargs):
    with pytest.raises(ValueError):
        SearchEngine(songs, backend="keyword").search(**kwargs)


def test_cache_reused_and_invalidated(songs, tmp_path):
    first = Encoder()
    SearchEngine(songs, encoder=first, cache_dir=tmp_path)
    assert first.calls == 1
    second = Encoder()
    SearchEngine(songs, encoder=second, cache_dir=tmp_path)
    assert second.calls == 0
    changed = songs + [Song("new", "New", "Ada", "New lyrics")]
    SearchEngine(changed, encoder=second, cache_dir=tmp_path)
    assert second.calls == 1
    for file in tmp_path.iterdir():
        file.write_bytes(b"damaged cache")
    third = Encoder()
    SearchEngine(songs, encoder=third, cache_dir=tmp_path)
    assert third.calls == 1


def test_chunk_aggregation(songs):
    class ChunkEncoder(Encoder):
        def chunks(self, text):
            return ["ordinary", text]

    engine = SearchEngine(songs, encoder=ChunkEncoder())
    result = engine.search("sorrow", mode="semantic")
    assert result[0]["id"] == "sad"
    assert len({r["id"] for r in result}) == len(result)


def test_ingestion_formats_and_validation(songs, tmp_path):
    import csv

    for suffix in ["json", "jsonl", "csv"]:
        path = tmp_path / f"songs.{suffix}"
        rows = [s.public() for s in songs]
        if suffix == "json":
            path.write_text(json.dumps(rows))
        elif suffix == "jsonl":
            path.write_text("\n".join(json.dumps(r) for r in rows))
        else:
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
        assert load_songs(path) == songs
    path = tmp_path / "bad.json"
    for data in [
        [],
        [songs[0].public()] * 2,
        [{"id": "missing"}],
        [{**songs[0].public(), "year": True}],
        [{**songs[0].public(), "lyrics": None}],
        [{**songs[0].public(), "extra": "unknown"}],
    ]:
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            load_songs(path)


def test_metrics_denominators_and_full_mrr():
    assert metrics(["no", "a", "b"], {"a", "b"}, 2) == {
        "precision@2": 0.5,
        "recall@2": 0.5,
        "mrr": 0.5,
    }
    assert metrics(["a"], {"a", "b"}, 5)["precision@5"] == 0.2
    assert metrics([], {"a"}, 5)["mrr"] == 0
    assert metrics(["x", "y", "a"], {"a"}, 2)["mrr"] == 1 / 3


def test_evaluation_beyond_ui_limit():
    songs = [Song(str(i), f"Title {i}", "Artist", "needle") for i in range(150)]
    report = evaluate(
        SearchEngine(songs, backend="keyword"), [{"query": "needle", "relevant": ["99"]}], k=5
    )
    assert report["configurations"][0]["metrics"]["mrr"] == 1 / 150


def test_judgments_and_evaluation(songs, tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps([{"query": "sorrow", "relevant": ["sad"]}]))
    cases = load_judgments(path, {s.id for s in songs})
    report = evaluate(SearchEngine(songs, encoder=Encoder()), cases, k=2)
    assert len(report["configurations"]) == 3
    assert report["configurations"][1]["metrics"]["mrr"] == 1
    path.write_text(json.dumps([{"query": "sorrow", "relevant": ["unknown"]}]))
    with pytest.raises(ValueError, match="Unknown"):
        load_judgments(path, {s.id for s in songs})


def test_api(songs):
    client = TestClient(
        create_app(
            SearchEngine(songs, backend="keyword"), [{"query": "regret", "relevant": ["sad"]}]
        )
    )
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/api/catalogue").json()["modes"] == ["keyword"]
    assert (
        client.get("/api/search", params={"q": "regret", "mode": "keyword"}).json()["results"][0][
            "id"
        ]
        == "sad"
    )
    for params in [{"q": "x", "k": 0}, {"q": "x", "mode": "semantic"}, {"q": " "}]:
        assert client.get("/api/search", params=params).status_code == 422
    report = client.get("/api/evaluate").json()
    assert report["configurations"][0]["metrics"]["mrr"] == 1
    missing = TestClient(create_app(SearchEngine(songs, backend="keyword")))
    assert missing.get("/api/evaluate").status_code == 404


def test_summary_catalogue_and_source_validation(tmp_path):
    path = tmp_path / "songs.json"
    summary = dict(
        id="real",
        title="Someone Like You",
        artist="Adele",
        description="A reflective song about a former relationship.",
        source_url="https://music.apple.com/us/album/example/1",
    )
    path.write_text(json.dumps([summary]))
    song = load_songs(path)[0]
    assert song.lyrics == ""
    assert SearchEngine([song], backend="keyword").search("reflective", mode="keyword")
    for changes in [
        dict(description=""),
        dict(source_url="javascript:alert(1)"),
        dict(source_url="https://user:password@example.com"),
    ]:
        path.write_text(json.dumps([{**summary, **changes}]))
        with pytest.raises(ValueError):
            load_songs(path)


def test_evaluation_ranking_does_not_materialize_full_lyrics(songs):
    engine = SearchEngine(songs, backend="keyword")
    rendered = engine.search("regret", mode="keyword")
    ranked = engine.search("regret", mode="keyword", details=False)
    assert [r["id"] for r in rendered] == [r["id"] for r in ranked]
    assert set(ranked[0]) == {"id", "title", "artist", "score"}


def test_custom_evaluation_and_reference_names(songs):
    client = TestClient(create_app(SearchEngine(songs, encoder=Encoder())))
    assert client.get("/api/judgments").json() == []
    body = {"cases": [{"query": "sorrow", "relevant": ["sad", "old"]}], "k": 10}
    response = client.post("/api/evaluate", json=body)
    assert response.status_code == 200
    report = response.json()
    assert report["cases"][0]["expected"] == [
        {"id": "sad", "title": "Farewell", "artist": "Ada"},
        {"id": "old", "title": "Winter", "artist": "Ada"},
    ]
    assert report["configurations"][1]["metrics"]["precision@10"] == 0.1
    assert report["configurations"][1]["metrics"]["recall@10"] == 0.5
    for changes in [
        {"cases": []},
        {"cases": [{"query": "x", "relevant": ["missing"]}]},
        {"cases": [{"query": "x", "relevant": ["sad", "sad"]}]},
        {"cases": body["cases"] * 2},
        {"k": 51},
        {"alpha": 2},
        {"extra": True},
    ]:
        assert client.post("/api/evaluate", json={**body, **changes}).status_code == 422
    fixed = TestClient(create_app(SearchEngine(songs, backend="keyword"), body["cases"]))
    assert fixed.get("/api/judgments").json() == report["cases"]


def test_http_search_supports_fifty_results():
    songs = [Song(str(i), str(i), "Artist", "needle") for i in range(60)]
    client = TestClient(create_app(SearchEngine(songs, backend="keyword")))
    response = client.get("/api/search", params={"q": "needle", "mode": "keyword", "k": 50})
    assert response.status_code == 200
    assert response.json()["count"] == 50
    assert len({s["id"] for s in response.json()["results"]}) == 50
    assert client.get("/api/search", params={"q": "needle", "k": 51}).status_code == 422
