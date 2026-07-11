"""Command-line interface for production newspaper analysis runs."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from newsvlm_analysis import __version__
from newsvlm_analysis.analysis_run import run_offline_analysis
from newsvlm_analysis.persistent_index import PersistentLexicalIndex, inspect_persistent_index
from newsvlm_analysis.queries import iter_query_specs, query_specs_from_strings
from newsvlm_analysis.validation import validate_analysis_run_bundle


def _json_print(payload: object, *, stream: object = sys.stdout) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), file=stream)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="newspaper-analysis",
        description="Build and query evidence-first offline newspaper analysis runs.",
    )
    parser.add_argument("--version", action="version", version=f"newspaper-analysis {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Build, query, evaluate, and validate an analysis run bundle.")
    run.add_argument("--parser-run-dir", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    query_input = run.add_mutually_exclusive_group(required=True)
    query_input.add_argument("--query", action="append", help="Query text; repeat for multiple questions.")
    query_input.add_argument("--queries-jsonl", type=Path, help="JSONL query manifest with optional relevance labels.")
    run.add_argument("--top-k", type=int, default=10)
    run.add_argument("--chunk-words", type=int, default=220)
    run.add_argument("--overlap-words", type=int, default=40)

    search = commands.add_parser("search", help="Search an existing persistent analysis index.")
    search.add_argument("--index", type=Path, required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=10)

    inspect = commands.add_parser("inspect-index", help="Print index contract, size, and corpus counts.")
    inspect.add_argument("--index", type=Path, required=True)

    validate = commands.add_parser("validate-run", help="Validate an offline analysis run bundle.")
    validate.add_argument("--run-dir", type=Path, required=True)
    validate.add_argument("--warnings-as-errors", action="store_true")
    validate.add_argument("--output-json", type=Path)
    return parser


def _run_command(args: argparse.Namespace) -> int:
    if args.queries_jsonl is not None:
        specs = list(iter_query_specs(args.queries_jsonl))
    else:
        specs = query_specs_from_strings(args.query or [])
    summary = run_offline_analysis(
        parser_run_dir=args.parser_run_dir,
        run_dir=args.run_dir,
        query_specs=specs,
        top_k=args.top_k,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
    )
    _json_print(summary)
    return 0 if summary.get("status") == "ok" else 1


def _search_command(args: argparse.Namespace) -> int:
    with PersistentLexicalIndex(args.index) as index:
        hits = index.search(args.query, top_k=args.top_k)
    _json_print(
        {
            "query": args.query,
            "top_k": args.top_k,
            "hit_count": len(hits),
            "hits": [asdict(hit) for hit in hits],
        }
    )
    return 0


def _validate_command(args: argparse.Namespace) -> int:
    report = validate_analysis_run_bundle(
        args.run_dir,
        warnings_are_errors=args.warnings_as_errors,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _json_print(report)
    return 0 if report["status"] != "error" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _run_command(args)
        if args.command == "search":
            return _search_command(args)
        if args.command == "inspect-index":
            _json_print(inspect_persistent_index(args.index))
            return 0
        if args.command == "validate-run":
            return _validate_command(args)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        _json_print(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            stream=sys.stderr,
        )
        return 2
    parser.error(f"unsupported command: {args.command}")
    return 2
