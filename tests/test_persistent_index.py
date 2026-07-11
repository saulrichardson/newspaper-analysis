from __future__ import annotations

from pathlib import Path

import pytest

from newsvlm_analysis.local_retrieval import SourceDocument
from newsvlm_analysis.persistent_index import (
    PersistentLexicalIndex,
    build_persistent_index,
    fts5_available,
    inspect_persistent_index,
)


pytestmark = pytest.mark.skipif(not fts5_available(), reason="Python SQLite does not provide FTS5")


def _documents() -> list[SourceDocument]:
    return [
        SourceDocument(
            doc_id="page-a",
            text="The zoning ordinance limited apartment height near the railroad station.",
            metadata={"page_id": "page-a", "issue_id": "issue-a"},
        ),
        SourceDocument(
            doc_id="page-b",
            text="The school board approved a library budget and new classroom furniture.",
            metadata={"page_id": "page-b", "issue_id": "issue-b"},
        ),
    ]


def test_persistent_index_builds_inspects_and_ranks(tmp_path: Path) -> None:
    index_path = tmp_path / "corpus.sqlite3"

    result = build_persistent_index(
        documents=_documents(),
        index_path=index_path,
        chunk_words=30,
        overlap_words=0,
        metadata={"fixture": True},
    )

    assert result.document_count == 2
    assert result.chunk_count == 2
    assert result.byte_count > 0
    info = inspect_persistent_index(index_path)
    assert info["contract_version"] == "analysis-sqlite-fts-index-v1"
    assert info["source_count"] == 2
    assert info["fts_chunk_count"] == info["chunk_count"]
    assert info["integrity"] == "ok"
    assert info["metadata"]["fixture"] is True

    with PersistentLexicalIndex(index_path) as index:
        hits = index.search("apartment height ordinance", top_k=2)
    assert hits[0].source_id == "page-a"
    assert hits[0].score > 0
    assert hits[0].metadata["issue_id"] == "issue-a"


def test_persistent_index_query_escapes_search_syntax(tmp_path: Path) -> None:
    index_path = tmp_path / "corpus.sqlite3"
    build_persistent_index(documents=_documents(), index_path=index_path, chunk_words=30, overlap_words=0)

    with PersistentLexicalIndex(index_path) as index:
        hits = index.search('zoning: (apartment) OR "height" -railroad', top_k=2)

    assert hits
    assert hits[0].source_id == "page-a"


def test_failed_rebuild_does_not_replace_existing_index(tmp_path: Path) -> None:
    index_path = tmp_path / "corpus.sqlite3"
    build_persistent_index(documents=_documents(), index_path=index_path, chunk_words=30, overlap_words=0)
    original = index_path.read_bytes()
    duplicate_documents = [
        SourceDocument(doc_id="same", text="first searchable text", metadata={"page_id": "same"}),
        SourceDocument(doc_id="same", text="second searchable text", metadata={"page_id": "same"}),
    ]

    with pytest.raises(ValueError, match="duplicate"):
        build_persistent_index(
            documents=duplicate_documents,
            index_path=index_path,
            chunk_words=30,
            overlap_words=0,
        )

    assert index_path.read_bytes() == original
    with PersistentLexicalIndex(index_path) as index:
        assert index.search("library", top_k=1)[0].source_id == "page-b"
