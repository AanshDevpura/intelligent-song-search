"""BM25, chunked sentence embeddings and weighted reciprocal rank fusion."""

import hashlib
import json
import os
import re
import sys
import tempfile
import threading
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from .data import Song

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Pin the default model's weights/tokenizer for reproducible experiments.
REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
STOP = set(
    "a an the of in on at to for and or is it its this that with about song songs "
    "i me my you your we our they their be was are as from by some something".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[^\W_]+(?:'[^\W_]+)?", text.casefold()) if t not in STOP]


class SentenceEncoder:
    def __init__(self, model: str = MODEL, revision: str | None = REVISION):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install semantic support: pip install -e '.[semantic]' "
                "or run with --backend keyword"
            ) from exc
        self.model = SentenceTransformer(
            model,
            revision=revision,
            trust_remote_code=False,
            device=os.environ.get("SONGSEARCH_DEVICE"),
            cache_folder=os.environ.get("SONGSEARCH_MODEL_CACHE", ".cache/models"),
        )
        self.identity = f"{model}@{revision}"
        print(f"Embedding model device: {self.model.device}", file=sys.stderr, flush=True)
        self.lock = threading.Lock()

    def chunks(self, text: str) -> list[str]:
        tokenizer = self.model.tokenizer
        tokens = tokenizer.encode(text, add_special_tokens=False)
        width = min(self.model.max_seq_length, 512) - tokenizer.num_special_tokens_to_add()
        overlap = min(32, width // 4)
        return [
            tokenizer.decode(tokens[i : i + width], skip_special_tokens=True)
            for i in range(0, len(tokens), width - overlap)
        ] or [text]

    def encode(self, texts: list[str]) -> np.ndarray:
        with self.lock:
            return self.model.encode(
                texts,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )


def normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class SearchEngine:
    def __init__(
        self,
        songs: list[Song],
        *,
        backend: str = "semantic",
        cache_dir: str | Path | None = None,
        encoder=None,
        model: str = MODEL,
        revision: str | None = REVISION,
    ):
        if backend not in {"semantic", "keyword"}:
            raise ValueError("backend must be semantic or keyword")
        if not songs or len({s.id for s in songs}) != len(songs):
            raise ValueError("Provide a nonempty catalogue with unique song IDs")
        self.songs = songs
        self.catalogue_id = hashlib.sha256("\0".join(s.id for s in songs).encode()).hexdigest()
        self.backend = backend
        self.encoder = None
        counts = [Counter(tokenize(s.text)) for s in songs]
        lengths = np.array([sum(c.values()) for c in counts], dtype=float)
        self.postings = {}
        for i, count in enumerate(counts):
            for term, frequency in count.items():
                self.postings.setdefault(term, []).append((i, frequency))
        self.postings = {
            term: np.asarray(items, dtype=np.int32).T for term, items in self.postings.items()
        }
        del counts
        self.length_norm = 1.2 * (0.25 + 0.75 * lengths / max(lengths.mean(), 1))
        self.embeddings = None
        self.owners = None
        if backend == "semantic":
            self.encoder = encoder if encoder is not None else SentenceEncoder(model, revision)
            fingerprint = hashlib.sha256(f"index-v2:{self.encoder.identity}".encode())
            for song in songs:
                fingerprint.update(
                    json.dumps(song.public(), sort_keys=True, ensure_ascii=False).encode()
                )
            self.fingerprint = fingerprint.hexdigest()
            cache = Path(cache_dir) / f"{self.fingerprint}.npz" if cache_dir else None
            if cache and cache.exists():
                try:
                    with np.load(cache, allow_pickle=False) as archive:
                        vectors, owners = archive["embeddings"], archive["owners"]
                    if (
                        vectors.ndim != 2
                        or owners.ndim != 1
                        or len(vectors) != len(owners)
                        or not len(owners)
                        or vectors.shape[1] == 0
                        or owners.dtype.kind not in "iu"
                        or not np.isfinite(vectors).all()
                        or set(owners.tolist()) != set(range(len(songs)))
                    ):
                        raise ValueError("Invalid embedding cache")
                    self.embeddings, self.owners = normalize(vectors), owners
                except (ValueError, KeyError, OSError, EOFError):
                    pass  # Rebuild a damaged cache; no pickle is ever deserialized.
            if self.embeddings is None:
                chunks, owners = [], []
                for i, song in enumerate(songs):
                    parts = self.encoder.chunks(song.text)
                    chunks.extend(parts)
                    owners.extend([i] * len(parts))
                blocks = []
                for start in range(0, len(chunks), 512):
                    blocks.append(normalize(self.encoder.encode(chunks[start : start + 512])))
                    if len(chunks) > 512:
                        print(
                            f"Embedding chunks: {min(start + 512, len(chunks))}/{len(chunks)}",
                            file=sys.stderr,
                            flush=True,
                        )
                self.embeddings = np.concatenate(blocks)
                del blocks, chunks
                self.owners = np.asarray(owners, dtype=np.int32)
                if cache:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    fd, name = tempfile.mkstemp(dir=cache.parent, suffix=".npz")
                    try:
                        with os.fdopen(fd, "wb") as stream:
                            np.savez_compressed(
                                stream, embeddings=self.embeddings, owners=self.owners
                            )
                        os.replace(name, cache)
                    finally:
                        if os.path.exists(name):
                            os.unlink(name)

    def keyword_scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.songs))
        for term in set(tokenize(query)):
            posting = self.postings.get(term)
            if posting is None:
                continue
            indices, frequencies = posting
            idf = np.log(1 + (len(self.songs) - len(indices) + 0.5) / (len(indices) + 0.5))
            scores[indices] += idf * frequencies * 2.2 / (frequencies + self.length_norm[indices])
        return scores

    def search(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        k: int = 10,
        alpha: float = 0.65,
        genre: str = "",
        artist: str = "",
        year_from: int | None = None,
        year_to: int | None = None,
        details: bool = True,
    ) -> list[dict]:
        query = query.strip()
        if not query or len(query) > 2000:
            raise ValueError("Query must contain 1–2000 characters")
        if mode not in {"keyword", "semantic", "hybrid"}:
            raise ValueError("Unknown search mode")
        if not 1 <= k <= max(100, len(self.songs)) or not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("k must be positive and within catalogue/UI bounds; alpha must be 0–1")
        if year_from is not None and year_to is not None and year_from > year_to:
            raise ValueError("Start year must not exceed end year")
        if self.backend == "keyword" and mode != "keyword":
            raise ValueError("Semantic and hybrid search require the semantic backend")
        candidates = [
            i
            for i, s in enumerate(self.songs)
            if (not genre or s.genre.casefold() == genre.casefold())
            and (not artist or artist.casefold() in s.artist.casefold())
            and (year_from is None or s.year is not None and s.year >= year_from)
            and (year_to is None or s.year is not None and s.year <= year_to)
        ]
        if not candidates:
            return []
        lexical = self.keyword_scores(query)
        semantic = np.zeros(len(self.songs))
        if mode != "keyword":
            query_vector = self.query_vector(query)
            semantic.fill(-1)
            np.maximum.at(semantic, self.owners, self.embeddings @ query_vector)

        # Filter before ranking. A filtered item never consumes an RRF rank.
        def ranks(scores):
            order = sorted(
                (i for i in candidates if scores[i] > 0),
                key=lambda i: (-float(scores[i]), self.songs[i].id),
            )
            return {i: rank for rank, i in enumerate(order, 1)}

        lex_ranks, sem_ranks = ranks(lexical), ranks(semantic)
        if mode == "keyword":
            scores = {i: float(lexical[i]) for i in lex_ranks}
        elif mode == "semantic":
            scores = {i: float(semantic[i]) for i in sem_ranks}
        else:
            scores = {
                i: (alpha / (60 + sem_ranks[i]) if i in sem_ranks else 0)
                + ((1 - alpha) / (60 + lex_ranks[i]) if i in lex_ranks else 0)
                for i in set(lex_ranks) | set(sem_ranks)
            }
        ordered = sorted(
            (i for i in scores if scores[i] > 0), key=lambda i: (-scores[i], self.songs[i].id)
        )[:k]
        if not details:
            return [
                {
                    "id": self.songs[i].id,
                    "title": self.songs[i].title,
                    "artist": self.songs[i].artist,
                    "score": round(scores[i], 6),
                }
                for i in ordered
            ]
        return [
            {
                **self.songs[i].public(),
                "score": round(scores[i], 6),
                "semantic_score": round(float(semantic[i]), 6) if mode != "keyword" else None,
                "keyword_score": round(float(lexical[i]), 6),
                "matched_terms": sorted(set(tokenize(query)) & set(tokenize(self.songs[i].text))),
            }
            for i in ordered
        ]

    @lru_cache(maxsize=128)
    def query_vector(self, query: str) -> np.ndarray:
        return normalize(self.encoder.encode([query]))[0]

    def info(self) -> dict:
        return {
            "count": len(self.songs),
            "catalogue_id": self.catalogue_id,
            "lyrics_count": sum(bool(s.lyrics) for s in self.songs),
            "year_count": sum(s.year is not None for s in self.songs),
            "backend": self.backend,
            "model": self.encoder.identity if self.encoder else None,
            "genres": sorted({s.genre for s in self.songs if s.genre}),
            "modes": ["keyword", "semantic", "hybrid"] if self.encoder else ["keyword"],
        }
