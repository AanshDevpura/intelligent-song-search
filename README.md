# Song Search

Find songs by describing their lyrics, story, or mood. Python handles retrieval and evaluation; the UI uses plain HTML, CSS, and JavaScript.

## Run

Requires Python 3.10+.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[semantic,dev]'
python -m songsearch
```

Open http://127.0.0.1:8000. First launch downloads and verifies the full dataset, then builds the embedding index. Later launches reuse `.cache/`. Set `PORT=8766` to use another port.

## Search and evaluation

Search up to 50 results using BM25 keyword matching, semantic similarity, or a hybrid of both. An optional artist filter narrows the catalogue.

The Evaluation tab compares methods against 25 test queries with ten expected songs each. For a custom test, mark search results relevant and click **Evaluate my labels**. Your labels are saved in this browser.

Relevance labels are hand-selected and opinion-based. Reviewing all 57,650 songs for every query was impractical, so the selections are imperfect and other songs may also be relevant. See the [evaluation report](reports/REPORT.md).

## Data and method

The complete [Spotify Million Song dataset mirror](https://huggingface.co/datasets/sebastiandizon/spotify-million-song) contains 57,650 lyric records from 643 artist names. The downloader pins the source revision and verifies its checksum. Data, model weights, and indexes are downloaded locally rather than committed to Git.

Semantic search uses `all-MiniLM-L6-v2`, overlapping lyric chunks, and maximum cosine similarity per song. Hybrid search combines semantic and BM25 ranks with weighted reciprocal rank fusion (semantic weight 0.65, rank constant 60). Genre and release years are unavailable in this dataset.

Code is MIT licensed. Dataset credits and transcriptions may contain errors; the source's CC0 declaration does not independently clear underlying lyric rights.

## Tests

```bash
pytest -q
ruff check songsearch tests
ruff format --check songsearch tests
```

After the dataset and model are cached, run integration checks with `HF_HUB_OFFLINE=1 SONGSEARCH_TEST_MODEL=1 SONGSEARCH_TEST_CORPUS=1 pytest -q`.

