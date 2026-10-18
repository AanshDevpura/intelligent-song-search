"""Read-only HTTP API and framework-free UI, served by the same Python process."""

from functools import lru_cache
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .evaluation import evaluate, references

STATIC = Path(__file__).parent / "static"


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cases: list[dict] = Field(min_length=1, max_length=100)
    k: int = Field(default=10, ge=1, le=50)
    alpha: float = Field(default=0.65, ge=0, le=1)


def create_app(engine, judgments=None) -> FastAPI:
    app = FastAPI(title="Songsearch", version="1.0.0")

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/catalogue")
    def catalogue():
        return {**engine.info(), "evaluation_available": bool(judgments)}

    @app.get("/api/search")
    def search(
        q: str = Query(min_length=1, max_length=2000),
        mode: str = "hybrid",
        k: int = Query(10, ge=1, le=50),
        alpha: float = Query(0.65, ge=0, le=1),
        artist: str = "",
    ):
        started = perf_counter()
        try:
            results = engine.search(
                q,
                mode=mode,
                k=k,
                alpha=alpha,
                artist=artist,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {
            "query": q.strip(),
            "mode": mode,
            "count": len(results),
            "elapsed_ms": round((perf_counter() - started) * 1000, 1),
            "results": results,
        }

    @app.get("/api/judgments")
    def expected():
        return references(engine, judgments or [])

    @app.post("/api/evaluate")
    def custom_evaluation(body: EvaluationRequest):
        try:
            return evaluate(engine, body.cases, k=body.k, alpha=body.alpha)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/evaluate")
    @lru_cache(maxsize=8)
    def benchmark(k: int = Query(5, ge=1, le=50), alpha: float = Query(0.65, ge=0, le=1)):
        if not judgments:
            raise HTTPException(404, "No relevance judgments configured")
        return evaluate(engine, judgments, k=k, alpha=alpha)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
