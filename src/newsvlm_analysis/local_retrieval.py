"""Offline lexical retrieval for API-free QA context building."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_'-]*")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    source_id: str
    text: str
    metadata: dict[str, Any]
    terms: Counter[str]
    token_count: int


@dataclass(frozen=True)
class RetrievalHit:
    query: str
    rank: int
    score: float
    chunk_id: str
    source_id: str
    text: str
    metadata: dict[str, Any]


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def iter_jsonl_documents(
    path: Path,
    *,
    id_field: str = "id",
    text_field: str = "text",
) -> Iterator[SourceDocument]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            text = str(row.get(text_field) or "")
            if not text.strip():
                continue
            raw_id = str(row.get(id_field) or "").strip()
            doc_id = raw_id or f"{path.stem}:{line_number}"
            metadata = {key: value for key, value in row.items() if key != text_field}
            metadata.setdefault("source_path", str(path))
            metadata.setdefault("line_number", line_number)
            yield SourceDocument(doc_id=doc_id, text=text, metadata=metadata)


def iter_text_documents(root: Path, suffixes: tuple[str, ...] = (".txt", ".md")) -> Iterator[SourceDocument]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue
        yield SourceDocument(
            doc_id=str(path.relative_to(root)),
            text=text,
            metadata={"source_path": str(path)},
        )


def chunk_document(
    document: SourceDocument,
    *,
    chunk_words: int = 220,
    overlap_words: int = 40,
) -> Iterator[TextChunk]:
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap_words < 0:
        raise ValueError("overlap_words must be non-negative")
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    words = document.text.split()
    if not words:
        return

    step = chunk_words - overlap_words
    chunk_index = 0
    for start in range(0, len(words), step):
        chunk_words_slice = words[start : start + chunk_words]
        if not chunk_words_slice:
            break
        text = " ".join(chunk_words_slice)
        terms = Counter(tokenize(text))
        if terms:
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "chunk_index": chunk_index,
                    "word_start": start,
                    "word_end": start + len(chunk_words_slice),
                }
            )
            yield TextChunk(
                chunk_id=f"{document.doc_id}#chunk-{chunk_index:05d}",
                source_id=document.doc_id,
                text=text,
                metadata=metadata,
                terms=terms,
                token_count=sum(terms.values()),
            )
        chunk_index += 1
        if start + chunk_words >= len(words):
            break


class LexicalIndex:
    def __init__(self, chunks: Iterable[TextChunk]) -> None:
        self.chunks = list(chunks)
        self.doc_freq: Counter[str] = Counter()
        total_tokens = 0
        for chunk in self.chunks:
            total_tokens += chunk.token_count
            self.doc_freq.update(chunk.terms.keys())
        self.avg_len = total_tokens / len(self.chunks) if self.chunks else 0.0

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[SourceDocument],
        *,
        chunk_words: int = 220,
        overlap_words: int = 40,
    ) -> "LexicalIndex":
        return cls(
            chunk
            for document in documents
            for chunk in chunk_document(
                document,
                chunk_words=chunk_words,
                overlap_words=overlap_words,
            )
        )

    def search(self, query: str, *, top_k: int = 10) -> list[RetrievalHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_terms = Counter(tokenize(query))
        if not query_terms or not self.chunks:
            return []

        scored: list[tuple[float, TextChunk]] = []
        total_chunks = len(self.chunks)
        k1 = 1.5
        b = 0.75
        avg_len = self.avg_len or 1.0

        for chunk in self.chunks:
            score = 0.0
            length_norm = k1 * (1 - b + b * (chunk.token_count / avg_len))
            for term, query_weight in query_terms.items():
                freq = chunk.terms.get(term, 0)
                if freq <= 0:
                    continue
                df = self.doc_freq.get(term, 0)
                idf = math.log(1 + (total_chunks - df + 0.5) / (df + 0.5))
                score += query_weight * idf * ((freq * (k1 + 1)) / (freq + length_norm))
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            RetrievalHit(
                query=query,
                rank=rank,
                score=round(score, 6),
                chunk_id=chunk.chunk_id,
                source_id=chunk.source_id,
                text=chunk.text,
                metadata=chunk.metadata,
            )
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]


def read_queries(path: Path, *, query_field: str = "query") -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("{"):
                row = json.loads(stripped)
                query = str(row.get(query_field) or "").strip()
            else:
                query = stripped
            if query:
                yield query


def retrieval_hits_to_jsonl_rows(hits: Iterable[RetrievalHit]) -> Iterator[dict[str, Any]]:
    for hit in hits:
        yield {
            "query": hit.query,
            "rank": hit.rank,
            "score": hit.score,
            "chunk_id": hit.chunk_id,
            "source_id": hit.source_id,
            "text": hit.text,
            "metadata": hit.metadata,
        }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count

