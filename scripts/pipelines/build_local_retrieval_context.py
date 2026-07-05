#!/usr/bin/env python3
"""Build API-free retrieval context rows from local transcript text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from newsvlm_analysis.local_retrieval import (
    LexicalIndex,
    iter_jsonl_documents,
    iter_text_documents,
    read_queries,
    retrieval_hits_to_jsonl_rows,
    write_jsonl,
)
from newsvlm_analysis.evidence import (
    build_evidence_contexts,
    iter_fused_page_documents,
    iter_parser_run_documents,
    parser_run_provenance,
    write_evidence_contexts_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-jsonl", type=Path, help="JSONL documents with id/text fields.")
    source.add_argument("--input-dir", type=Path, help="Directory of .txt/.md documents.")
    source.add_argument("--fused-pages", type=Path, help="Parser fused-page JSON file or directory.")
    source.add_argument("--parser-run-dir", type=Path, help="Parser run bundle with outputs/fused_pages.")
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument("--query", help="Single query string.")
    query.add_argument("--queries-jsonl", type=Path, help="JSONL or plain-text query file.")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--query-field", default="query")
    parser.add_argument("--chunk-words", type=int, default=220)
    parser.add_argument("--overlap-words", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output-format",
        choices=["hits", "contexts"],
        default="hits",
        help="Write flat retrieval hits or evidence-context packets.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.input_jsonl:
        documents = list(
            iter_jsonl_documents(
                args.input_jsonl,
                id_field=args.id_field,
                text_field=args.text_field,
            )
        )
        input_mode = "input_jsonl"
        source_provenance = {"input_jsonl": str(args.input_jsonl)}
    elif args.input_dir:
        documents = list(iter_text_documents(args.input_dir))
        input_mode = "input_dir"
        source_provenance = {"input_dir": str(args.input_dir)}
    elif args.fused_pages:
        documents = list(iter_fused_page_documents(args.fused_pages))
        input_mode = "fused_pages"
        source_provenance = {"fused_pages": str(args.fused_pages)}
    else:
        documents = list(iter_parser_run_documents(args.parser_run_dir))
        input_mode = "parser_run"
        source_provenance = parser_run_provenance(args.parser_run_dir)

    queries = [args.query] if args.query else list(read_queries(args.queries_jsonl, query_field=args.query_field))
    if args.output_format == "contexts":
        contexts = build_evidence_contexts(
            documents=documents,
            queries=queries,
            top_k=args.top_k,
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
            provenance={"input_mode": input_mode, **source_provenance},
        )
        written = write_evidence_contexts_jsonl(args.output_jsonl, contexts)
        chunk_count = contexts[0].provenance["chunk_count"] if contexts else 0
    else:
        index = LexicalIndex.from_documents(
            documents,
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
        )
        rows = (
            row
            for query in queries
            for row in retrieval_hits_to_jsonl_rows(index.search(query, top_k=args.top_k))
        )
        written = write_jsonl(args.output_jsonl, rows)
        chunk_count = len(index.chunks)
    print(
        json.dumps(
            {
                "documents": len(documents),
                "chunks": chunk_count,
                "queries": len(queries),
                "rows_written": written,
                "output_jsonl": str(args.output_jsonl),
                "output_format": args.output_format,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
