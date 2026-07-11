"""Persistent, dependency-free lexical retrieval for analysis run bundles."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from newsvlm_analysis.local_retrieval import RetrievalHit, SourceDocument, chunk_document, tokenize


INDEX_CONTRACT = "analysis-sqlite-fts-index-v1"


@dataclass(frozen=True)
class IndexBuildResult:
    index_path: str
    document_count: int
    chunk_count: int
    token_count: int
    byte_count: int
    seconds: float
    chunks_per_second: float
    contract_version: str = INDEX_CONTRACT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _connect(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    else:
        connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def fts5_available() -> bool:
    """Return whether the active Python SQLite build supports FTS5."""

    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE VIRTUAL TABLE fts_probe USING fts5(text)")
    except sqlite3.OperationalError:
        return False
    finally:
        connection.close()
    return True


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE corpus_metadata (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );

        CREATE TABLE chunks (
            rowid INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL UNIQUE,
            source_id TEXT NOT NULL,
            source_page_id TEXT NOT NULL,
            text TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            content_sha256 TEXT NOT NULL
        );

        CREATE INDEX chunks_source_id_idx ON chunks(source_id);
        CREATE INDEX chunks_source_page_id_idx ON chunks(source_page_id);

        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            text,
            content='chunks',
            content_rowid='rowid',
            tokenize='porter unicode61 remove_diacritics 2'
        );
        """
    )


def _write_metadata(connection: sqlite3.Connection, metadata: dict[str, Any]) -> None:
    connection.executemany(
        "INSERT INTO corpus_metadata(key, value_json) VALUES (?, ?)",
        ((key, _json(value)) for key, value in sorted(metadata.items())),
    )


def build_persistent_index(
    *,
    documents: Iterable[SourceDocument],
    index_path: Path,
    chunk_words: int = 220,
    overlap_words: int = 40,
    metadata: dict[str, Any] | None = None,
) -> IndexBuildResult:
    """Build an atomic SQLite FTS5 index from source documents.

    The index is assembled in a temporary file and moved into place only after
    all chunks and the FTS index have committed successfully.
    """

    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be non-negative and smaller than chunk_words")
    if not fts5_available():
        raise RuntimeError("the active Python SQLite build does not provide FTS5")

    destination = index_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    document_count = 0
    chunk_count = 0
    token_count = 0

    try:
        connection = _connect(temporary)
        try:
            connection.execute("PRAGMA journal_mode = OFF")
            connection.execute("PRAGMA synchronous = OFF")
            connection.execute("PRAGMA temp_store = MEMORY")
            _create_schema(connection)

            batch: list[tuple[Any, ...]] = []
            for document in documents:
                document_count += 1
                for chunk in chunk_document(
                    document,
                    chunk_words=chunk_words,
                    overlap_words=overlap_words,
                ):
                    source_page_id = str(chunk.metadata.get("page_id") or chunk.source_id)
                    batch.append(
                        (
                            chunk.chunk_id,
                            chunk.source_id,
                            source_page_id,
                            chunk.text,
                            _json(chunk.metadata),
                            chunk.token_count,
                            hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
                        )
                    )
                    chunk_count += 1
                    token_count += chunk.token_count
                    if len(batch) >= 1_000:
                        connection.executemany(
                            """
                            INSERT INTO chunks(
                                chunk_id, source_id, source_page_id, text,
                                metadata_json, token_count, content_sha256
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            batch,
                        )
                        batch.clear()

            if batch:
                connection.executemany(
                    """
                    INSERT INTO chunks(
                        chunk_id, source_id, source_page_id, text,
                        metadata_json, token_count, content_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )

            if document_count == 0:
                raise ValueError("cannot build an index from zero documents")
            if chunk_count == 0:
                raise ValueError("source documents produced zero searchable chunks")

            index_metadata = {
                "contract_version": INDEX_CONTRACT,
                "document_count": document_count,
                "chunk_count": chunk_count,
                "token_count": token_count,
                "chunk_words": chunk_words,
                "overlap_words": overlap_words,
                "sqlite_version": sqlite3.sqlite_version,
                **(metadata or {}),
            }
            _write_metadata(connection, index_metadata)
            connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
            connection.execute("ANALYZE")
            connection.execute("PRAGMA optimize")
            connection.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"index input contains duplicate chunk or source identifiers: {exc}") from exc
        finally:
            connection.close()

        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    elapsed = time.perf_counter() - started
    byte_count = destination.stat().st_size
    return IndexBuildResult(
        index_path=str(destination),
        document_count=document_count,
        chunk_count=chunk_count,
        token_count=token_count,
        byte_count=byte_count,
        seconds=round(elapsed, 6),
        chunks_per_second=round(chunk_count / elapsed, 3) if elapsed > 0 else 0.0,
    )


def _fts_query(query: str) -> str:
    unique_terms = list(dict.fromkeys(tokenize(query)))
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in unique_terms)


class PersistentLexicalIndex:
    """Read-only search handle for an analysis SQLite FTS5 index."""

    def __init__(self, index_path: Path) -> None:
        self.path = index_path.expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"analysis index does not exist: {self.path}")
        self.connection = _connect(self.path, read_only=True)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "PersistentLexicalIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def metadata(self) -> dict[str, Any]:
        rows = self.connection.execute(
            "SELECT key, value_json FROM corpus_metadata ORDER BY key"
        ).fetchall()
        return {str(row["key"]): json.loads(str(row["value_json"])) for row in rows}

    def inspect(self) -> dict[str, Any]:
        metadata = self.metadata()
        chunk_count = int(self.connection.execute("SELECT count(*) FROM chunks").fetchone()[0])
        fts_chunk_count = int(self.connection.execute("SELECT count(*) FROM chunks_fts").fetchone()[0])
        source_count = int(
            self.connection.execute("SELECT count(DISTINCT source_id) FROM chunks").fetchone()[0]
        )
        integrity = str(self.connection.execute("PRAGMA quick_check").fetchone()[0])
        return {
            "contract_version": metadata.get("contract_version", ""),
            "index_path": str(self.path),
            "byte_count": self.path.stat().st_size,
            "chunk_count": chunk_count,
            "fts_chunk_count": fts_chunk_count,
            "source_count": source_count,
            "integrity": integrity,
            "metadata": metadata,
        }

    def search(self, query: str, *, top_k: int = 10) -> list[RetrievalHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        match_query = _fts_query(query)
        if not match_query:
            return []
        rows = self.connection.execute(
            """
            SELECT
                c.chunk_id,
                c.source_id,
                c.source_page_id,
                c.text,
                c.metadata_json,
                bm25(chunks_fts, 1.0) AS rank_score
            FROM chunks_fts
            JOIN chunks AS c ON c.rowid = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY rank_score ASC, c.chunk_id ASC
            LIMIT ?
            """,
            (match_query, top_k),
        ).fetchall()
        hits: list[RetrievalHit] = []
        for rank, row in enumerate(rows, start=1):
            metadata = json.loads(str(row["metadata_json"]))
            metadata.setdefault("page_id", str(row["source_page_id"]))
            hits.append(
                RetrievalHit(
                    query=query,
                    rank=rank,
                    score=round(max(0.0, -float(row["rank_score"])), 9),
                    chunk_id=str(row["chunk_id"]),
                    source_id=str(row["source_id"]),
                    text=str(row["text"]),
                    metadata=metadata,
                )
            )
        return hits


def inspect_persistent_index(index_path: Path) -> dict[str, Any]:
    with PersistentLexicalIndex(index_path) as index:
        return index.inspect()
