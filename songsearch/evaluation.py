"""Evaluate explicit relevance judgments; never infer labels from rankings."""

import hashlib
import json
from pathlib import Path


def load_judgments(path: str | Path, song_ids: set[str]) -> list[dict]:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_judgments(cases, song_ids)


def validate_judgments(cases, song_ids: set[str]) -> list[dict]:
    if not isinstance(cases, list) or not cases:
        raise ValueError("Judgments must be a nonempty JSON array")
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each judgment must be an object")
        query, relevant = case.get("query"), case.get("relevant")
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("Every judgment needs a query of 1–2000 characters")
        if query in seen:
            raise ValueError(f"Duplicate evaluation query: {query}")
        seen.add(query)
        if (
            not isinstance(relevant, list)
            or not relevant
            or any(not isinstance(x, str) for x in relevant)
            or len(set(relevant)) != len(relevant)
        ):
            raise ValueError("Each query needs nonempty, unique relevant song IDs")
        if set(relevant) - song_ids:
            raise ValueError(f"Unknown relevant song IDs: {sorted(set(relevant) - song_ids)}")
    return cases


def references(engine, cases):
    songs = {s.id: s for s in engine.songs}
    return [
        {
            "query": c["query"],
            "expected": [
                {"id": sid, "title": songs[sid].title, "artist": songs[sid].artist}
                for sid in c["relevant"]
            ],
        }
        for c in cases
    ]


def metrics(ranked: list[str], relevant: set[str], k: int) -> dict:
    if k < 1 or not relevant or len(set(ranked)) != len(ranked):
        raise ValueError("Metrics need k > 0, relevant IDs and unique ranked IDs")
    hits = len(set(ranked[:k]) & relevant)
    reciprocal = next((1 / i for i, sid in enumerate(ranked, 1) if sid in relevant), 0.0)
    return {f"precision@{k}": hits / k, f"recall@{k}": hits / len(relevant), "mrr": reciprocal}


def evaluate(engine, cases: list[dict], *, k: int = 5, alpha: float = 0.65) -> dict:
    if not 1 <= k <= 100 or not cases:
        raise ValueError("Evaluation needs cases and k of 1–100")
    validate_judgments(cases, {s.id for s in engine.songs})
    reports = []
    for mode in engine.info()["modes"]:
        rows = []
        for case in cases:
            results = engine.search(
                case["query"], mode=mode, k=len(engine.songs), alpha=alpha, details=False
            )
            ids = [r["id"] for r in results]
            rows.append(
                {
                    "query": case["query"],
                    "relevant": case["relevant"],
                    **metrics(ids, set(case["relevant"]), k),
                    "results": [
                        {
                            "id": r["id"],
                            "title": r["title"],
                            "artist": r["artist"],
                            "score": r["score"],
                            "relevant": r["id"] in case["relevant"],
                        }
                        for r in results[:k]
                    ],
                }
            )
        reports.append(
            {
                "mode": mode,
                "metrics": {
                    key: sum(r[key] for r in rows) / len(rows)
                    for key in (f"precision@{k}", f"recall@{k}", "mrr")
                },
                "queries": rows,
            }
        )
    return {
        "catalogue_size": len(engine.songs),
        "catalogue_id": engine.catalogue_id,
        "judgments_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
        "query_count": len(cases),
        "cases": references(engine, cases),
        "k": k,
        "alpha": alpha,
        "model": engine.info()["model"],
        "mrr_scope": "full positive-score ranking",
        "configurations": reports,
    }
