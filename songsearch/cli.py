import argparse
import json
import sys
from pathlib import Path

from .data import load_songs
from .engine import MODEL, REVISION, SearchEngine
from .evaluation import evaluate, load_judgments


def main():
    parser = argparse.ArgumentParser(description="Songsearch: search by meaning, mood or lyrics")
    parser.add_argument(
        "--data", default="data/songs.jsonl", help="CSV, JSON or JSONL song catalogue"
    )
    parser.add_argument("--backend", choices=["semantic", "keyword"], default="semantic")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", help="Model revision; default model uses pinned weights")
    parser.add_argument("--cache", default=".cache/embeddings")
    sub = parser.add_subparsers(dest="command", required=True)
    preparation = sub.add_parser(
        "prepare", help="Download and import the full pinned corpus, then build the index"
    )
    preparation.add_argument("--raw", help="Use a previously downloaded source CSV")
    sub.add_parser("index", help="Validate catalogue and build/cache embeddings")
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--mode", choices=["keyword", "semantic", "hybrid"])
    search.add_argument("-k", type=int, default=10)
    search.add_argument("--alpha", type=float, default=0.65)
    search.add_argument("--genre", default="")
    search.add_argument("--artist", default="")
    search.add_argument("--year-from", type=int)
    search.add_argument("--year-to", type=int)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--judgments", default="data/judgments.json")
    evaluation.add_argument("-k", type=int, default=5)
    evaluation.add_argument("--alpha", type=float, default=0.65)
    evaluation.add_argument("--output", help="Write JSON report")
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--judgments", help="Enable evaluation tab with this judgments JSON")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            from .corpus import prepare

            print(json.dumps(prepare(args.data, raw_path=args.raw), indent=2), file=sys.stderr)
        if not Path(args.data).exists():
            raise ValueError(f"Dataset missing: {args.data}. Run songsearch prepare first.")
        songs = load_songs(args.data)
        revision = args.revision or (REVISION if args.model == MODEL else None)
        engine = SearchEngine(
            songs, backend=args.backend, cache_dir=args.cache, model=args.model, revision=revision
        )
        if args.command in {"index", "prepare"}:
            report = engine.info()
        elif args.command == "search":
            report = engine.search(
                args.query,
                mode=args.mode or ("hybrid" if args.backend == "semantic" else "keyword"),
                k=args.k,
                alpha=args.alpha,
                genre=args.genre,
                artist=args.artist,
                year_from=args.year_from,
                year_to=args.year_to,
            )
        elif args.command == "evaluate":
            cases = load_judgments(args.judgments, {s.id for s in songs})
            report = evaluate(engine, cases, k=args.k, alpha=args.alpha)
        else:
            import uvicorn

            from .api import create_app

            cases = (
                load_judgments(args.judgments, {s.id for s in songs}) if args.judgments else None
            )
            uvicorn.run(create_app(engine, cases), host=args.host, port=args.port)
            return
        output = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        if getattr(args, "output", None):
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(output + "\n", encoding="utf-8")
        print(output)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"songsearch: {exc}", file=sys.stderr)
        sys.exit(2)
