# Song Search

A small Python project for finding songs from descriptions of their lyrics, themes, or moods. The UI is plain HTML, CSS, and JavaScript. No frontend framework, vector database, API key, or hosted inference service.

## Run

Requires Python 3.10+. Run commands from this directory.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[semantic,dev]'
songsearch prepare
songsearch serve --judgments data/judgments.json
```

Open **http://127.0.0.1:8000**.

`prepare` downloads the pinned source CSV, verifies its SHA-256, imports every valid row, and builds embeddings. It takes several minutes on a GPU and longer on a CPU. Models and indexes are cached under `.cache/`; progress is printed while indexing. Subsequent starts reuse the index. `HF_HUB_OFFLINE=1` works after model download; `SONGSEARCH_DEVICE=cpu` forces CPU and `SONGSEARCH_MODEL_CACHE` overrides the model-cache directory.

The source ZIP excludes the local corpus, model cache, and virtual environment. The downloader recreates the complete dataset; it does not select a small sample.

## Dataset

The default is the complete [Spotify Million Song dataset mirror](https://huggingface.co/datasets/sebastiandizon/spotify-million-song): **57,650 full-lyrics records**, 643 artist names, 100,334 indexed chunks. Its name does not describe its actual size. All rows imported; no songs were dropped. The corpus has 57,484 distinct normalized lyric texts; 166 duplicate texts are preserved because different records may represent different versions.

The index uses **title, artist, and lyrics**, without generated summaries or theme labels. Genre and release year are absent in this source. Artist filtering is available in the UI; optional genre/year filters remain supported by the API/CLI for other datasets. Source transcriptions and credits can contain errors.

The pinned revision and source checksum live in `songsearch/dataset-source.json`. `data/songs.manifest.json` records import counts and the normalized-file checksum. The upstream dataset card declares CC0; that declaration is not independent clearance of underlying lyric rights. The code's MIT license does not relicense third-party lyrics or model weights.

## Search

```bash
songsearch search 'regretting the end of a relationship' --mode semantic -k 5
songsearch search 'finding hope after difficult times' --mode hybrid --alpha 0.65
songsearch search 'missing someone' --artist Adele
```

Global options precede the command:

```bash
songsearch --backend keyword search 'lonely city'
songsearch --data /path/to/songs.csv index
```

The keyword backend needs only the base install (`pip install -e .`) and explicitly disables semantic/hybrid search. Search modes:

- **Keyword:** BM25, Unicode word tokens, case folding, English stop words; `k1=1.2`, `b=0.75`.
- **Semantic:** normalized `all-MiniLM-L6-v2` embeddings, overlapping tokenizer-bounded chunks, maximum chunk cosine per song.
- **Hybrid:** weighted reciprocal rank fusion, `alpha/(60 + semantic_rank) + (1-alpha)/(60 + keyword_rank)`, default alpha 0.65.

Filters apply before ranking. Nonpositive scores are excluded; ties sort by stable ID. Scores are ranking signals, not relevance probabilities. Retrieval is exact and in memory. Compact NumPy postings reduce BM25 memory; embeddings are batched and cached using a dataset/model fingerprint. Cache loading disallows pickle and cache writes are atomic. The model revision is pinned; `--model` and `--revision` allow alternatives. This English encoder has not been validated for multilingual retrieval.

## Evaluation

```bash
songsearch evaluate --judgments data/judgments.json -k 5 \
  --output reports/my-evaluation.json
```

The **Evaluation** tab lists all 25 configured queries with ten expected songs each (250 query–song judgments). Expand a query to see titles and artists before running the comparison. The ordered lists are assistant-curated expectations, including broader thematic matches and some cover versions, not human-validated or exhaustive ground truth. Order is an editorial preference; the binary metrics treat all ten songs equally. Expanded labels were curated by content, not copied from retrieval output; the initial single-reference benchmark had already been inspected. No ranking weights were tuned on the expanded set.

For a custom evaluation, search a description, mark relevant songs, and click **Evaluate my labels** under **Relevance labels**. The app evaluates all saved query labels directly against the full catalogue and displays metrics, expected songs, and retrieved results. You can also choose **My labels** in Evaluation. Labels persist in this browser per catalogue. Search supports up to 50 results, so you can judge a larger candidate list. Label/report JSON downloads remain optional; no CLI or server restart is needed for custom evaluation.

The CLI also accepts judgment files shaped like `[{"query": "a description", "relevant": ["song-id"]}]`.

Precision@K uses K as its denominator, Recall@K uses all labeled relevant IDs, and MRR uses the first relevant item over the full positive-score ranking. Aggregate scores are macro averages. Unknown/duplicate IDs and empty relevance sets are rejected. Evaluation retrieves compact rankings without serializing full lyrics; repeated configured API evaluations are cached.

Measured results and limitations: [reports/REPORT.md](reports/REPORT.md). Raw per-query scores: [reports/full-corpus-evaluation.json](reports/full-corpus-evaluation.json).

## Other datasets and API

Import UTF-8 CSV, JSON arrays, or JSONL using `id`, `title`, `artist`, `lyrics`, `genre`, `year`, `description`, `tags`, `source_url`. Require unique ID, title, artist, and lyrics or description. Missing optional text defaults to empty; year is a four-digit integer or null. Unknown fields and malformed records fail explicitly. An optional source URL must use HTTPS without credentials.

The server binds to localhost and has no authentication. API documentation is at `/docs`:

- `GET /api/health`
- `GET /api/catalogue`
- `GET /api/search?q=...&mode=hybrid&k=10&alpha=0.65` (optional `artist`, `genre`, `year_from`, `year_to`; HTTP K capped at 50)
- `GET /api/judgments` (configured queries with expected song titles/artists)
- `GET /api/evaluate?k=10&alpha=0.65` (requires configured judgments)
- `POST /api/evaluate` with `{ "cases": [{ "query": "...", "relevant": ["song-id"] }], "k": 10, "alpha": 0.65 }` (custom evaluation; up to 100 queries, K 1–50)

## Verification and files

```bash
pytest -q
ruff check songsearch tests
ruff format --check songsearch tests
# Actual-model and full-corpus checks:
HF_HUB_OFFLINE=1 SONGSEARCH_TEST_MODEL=1 SONGSEARCH_TEST_CORPUS=1 pytest -q
```

**Verified: 25 tests pass with both integration checks enabled**, Ruff checks pass, and the desktop/mobile UI has been checked in the browser.

Normal tests run without downloading a model or corpus; opt-in tests require `prepare`. Tests cover ingestion, checksums, cache behavior, retrieval/fusion, metadata filters, metrics, complete-corpus integrity, and API contracts. CI covers Python 3.10/3.12; remote CI has not been run.

`corpus.py` handles the download/import; `data.py` validates records; `engine.py` implements retrieval; `evaluation.py` computes metrics; `cli.py` and `api.py` expose them. UI files are under `songsearch/static/`. The only runtime base dependencies are NumPy, FastAPI, and Uvicorn; sentence-transformers is the semantic extra.

Inspired by [intelligent-movie-search](https://github.com/sydneyaurelius/intelligent-movie-search), implemented independently.
