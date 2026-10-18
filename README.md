# Song Search

Find songs by describing their lyrics, story, or mood. Python handles retrieval and evaluation; the UI uses plain HTML, CSS, and JavaScript.

## Run

Requires Python 3.10+. Run from the project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[semantic,dev]'
python -m songsearch
```

Open http://127.0.0.1:8000 once the terminal reports that Uvicorn is running. First launch requires internet: it downloads and verifies the full dataset, downloads the embedding model, and builds the index before starting the server. Indexing can take several minutes and is slower on a CPU; progress appears in the terminal. Later launches reuse `.cache/`.

Use `PORT=8766 python -m songsearch` for another port. After setup, `HF_HUB_OFFLINE=1 python -m songsearch` uses the cached model without contacting Hugging Face. Keep `data/songs.jsonl` and `.cache/` to avoid repeating download/indexing work.

## Usage

### Search

Enter a description such as “a sad song about regretting a breakup,” choose a method, and click **Search**. Results can include up to 50 songs. The optional artist filter matches part of an artist's name, ignoring case, and applies before ranking.

- **Keyword:** BM25 matching against title, artist, and lyrics; useful when the description includes recognizable words.
- **Semantic:** Embedding similarity; useful for themes or paraphrases without exact lyric overlap.
- **Hybrid:** Combines both rankings. **Semantic weight** controls their contributions: 1 uses only semantic ranks, 0 uses only keyword ranks, and the default is 0.65.

Expand **Lyrics and scores** to inspect a result's lyrics, matched words, cosine similarity, and BM25 score. Scores are ranking signals, not probabilities, and their scales differ across methods.

### Evaluation

Open **Evaluation**, choose **Configured queries**, select K and the hybrid weight, then click **Run evaluation**. The test set has 25 queries with ten expected songs each. Expand a query to compare its expected songs with each method's results and per-query metrics.

For a custom test, search a description and mark relevant results. Open **Relevance labels** and click **Evaluate my labels**, or select **My labels** in Evaluation. Clicking **Relevant ✓** again removes a label. Labels persist in this browser; custom evaluation includes all saved queries and searches the full catalogue, regardless of the search screen's artist filter.

Precision@K divides relevant hits by K; Recall@K divides hits by the number of labeled songs. MRR averages the reciprocal rank of the first relevant result across queries, using each full positive-score ranking rather than stopping at K. The displayed metrics are macro averages across queries.

Labels are hand-selected and opinion-based. Reviewing all 57,650 songs for every query was impractical, so the selections are imperfect and other songs may also be relevant. All expected records count equally; list order does not affect the metrics, and cover versions count as separate records. See the [evaluation report](reports/REPORT.md).

## Implementation

The pipeline is: **source CSV → validated song records → BM25 postings and chunk embeddings → ranked songs → browser UI**.

| File | Responsibility |
| --- | --- |
| [`__main__.py`](songsearch/__main__.py) | Prepare missing data, load the index and judgments, and start the server. |
| [`corpus.py`](songsearch/corpus.py), [`data.py`](songsearch/data.py) | Download/checksum the source, create stable song IDs, write JSONL and a manifest, and validate records. |
| [`engine.py`](songsearch/engine.py) | Tokenization, BM25, chunk embeddings, artist filtering, caching, and ranking. |
| [`evaluation.py`](songsearch/evaluation.py) | Validate relevance IDs and compute per-query and aggregate metrics. |
| [`api.py`](songsearch/api.py), [`static/`](songsearch/static/) | Serve the FastAPI endpoints and UI; JavaScript submits searches/evaluations and stores custom labels locally. |

**Keyword retrieval** case-folds Unicode word tokens and removes a small English stop-word list. BM25 uses term-frequency postings, document-length normalization, `k1=1.2`, and `b=0.75`.

**Semantic retrieval** embeds title, artist, and lyrics with the pinned `all-MiniLM-L6-v2` model. Tokenizer-bounded chunks overlap by 32 tokens to cover long lyrics. Embeddings are L2-normalized; a song's score is its maximum chunk cosine similarity to the query. NumPy scores the chunks directly, without a vector database. The embedding cache is keyed by song records and model identity, so changing either rebuilds the index.

**Hybrid retrieval** uses weighted reciprocal rank fusion: `alpha / (60 + semantic_rank) + (1 - alpha) / (60 + keyword_rank)`. A missing rank contributes zero. Nonpositive results are excluded, and ties break by stable song ID. Evaluation requests compact full rankings without returning lyrics for every song.

The UI uses `GET /api/search`, `GET /api/judgments`, and `GET`/`POST /api/evaluate` for configured/custom evaluation. Interactive API documentation is available at `/docs` while the server is running. Configured labels live in [`data/judgments.json`](data/judgments.json).

## Dataset

The complete [Spotify Million Song dataset mirror](https://huggingface.co/datasets/sebastiandizon/spotify-million-song) contains 57,650 lyric records from 643 artist names. The source revision and SHA-256 are pinned in [`dataset-source.json`](songsearch/dataset-source.json); the import manifest records counts and the normalized-file checksum. Genre and release years are unavailable. Data, model weights, and indexes are downloaded locally rather than committed to Git.

Code is MIT licensed. Dataset credits and transcriptions may contain errors; the source's CC0 declaration does not independently clear underlying lyric rights.

## Tests

```bash
pytest -q
ruff check songsearch tests
ruff format --check songsearch tests
```

Default tests use small fixtures and a controlled encoder to check ingestion, caching, retrieval/fusion, filtering, metrics, API validation, and startup without downloading the corpus or model. After the app has prepared both, run the actual-model and full-corpus integration checks:

```bash
HF_HUB_OFFLINE=1 SONGSEARCH_TEST_MODEL=1 SONGSEARCH_TEST_CORPUS=1 pytest -q
```
