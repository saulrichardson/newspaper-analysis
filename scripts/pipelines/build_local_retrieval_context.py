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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-jsonl", type=Path, help="JSONL documents with id/text fields.")
    source.add_argument("--input-dir", type=Path, help="Directory of .txt/.md documents.")
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
    else:
        documents = list(iter_text_documents(args.input_dir))

    index = LexicalIndex.from_documents(
        documents,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
    )
    queries = [args.query] if args.query else list(read_queries(args.queries_jsonl, query_field=args.query_field))
    rows = (
        row
        for query in queries
        for row in retrieval_hits_to_jsonl_rows(index.search(query, top_k=args.top_k))
    )
    written = write_jsonl(args.output_jsonl, rows)
    print(
        json.dumps(
            {
                "documents": len(documents),
                "chunks": len(index.chunks),
                "queries": len(queries),
                "rows_written": written,
                "output_jsonl": str(args.output_jsonl),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
