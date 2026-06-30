from __future__ import annotations

import json
from pathlib import Path

from newsvlm_analysis.local_retrieval import (
    LexicalIndex,
    SourceDocument,
    iter_jsonl_documents,
    read_queries,
    retrieval_hits_to_jsonl_rows,
    write_jsonl,
)


def test_lexical_index_ranks_policy_specific_context_first() -> None:
    index = LexicalIndex.from_documents(
        [
            SourceDocument(
                doc_id="zoning",
                text="The city council adopted a zoning ordinance limiting apartment height near transit.",
                metadata={"city": "Exampleville"},
            ),
            SourceDocument(
                doc_id="weak",
                text="The newspaper published an ordinance about parade permits before the baseball scores.",
                metadata={"city": "Exampleville"},
            ),
        ],
        chunk_words=50,
        overlap_words=0,
    )

    hits = index.search("zoning ordinance apartment height", top_k=2)

    assert hits[0].source_id == "zoning"
    assert hits[0].score > hits[-1].score
    assert hits[0].metadata["city"] == "Exampleville"


def test_jsonl_document_and_query_roundtrip(tmp_path: Path) -> None:
    docs_path = tmp_path / "docs.jsonl"
    docs_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "doc-a", "text": "zoning board variance hearing", "year": 1947}),
                json.dumps({"id": "doc-b", "text": "public school budget hearing", "year": 1948}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    queries_path = tmp_path / "queries.jsonl"
    queries_path.write_text(json.dumps({"query": "variance zoning"}) + "\n", encoding="utf-8")

    documents = list(iter_jsonl_documents(docs_path))
    index = LexicalIndex.from_documents(documents, chunk_words=20, overlap_words=0)
    hits = index.search(next(read_queries(queries_path)), top_k=1)
    output_path = tmp_path / "hits.jsonl"
    written = write_jsonl(output_path, retrieval_hits_to_jsonl_rows(hits))

    assert written == 1
    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["source_id"] == "doc-a"
    assert row["metadata"]["year"] == 1947
