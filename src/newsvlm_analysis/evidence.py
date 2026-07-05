"""Evidence-first offline analysis contracts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from newsvlm_analysis.local_retrieval import (
    LexicalIndex,
    RetrievalHit,
    SourceDocument,
)


@dataclass(frozen=True)
class EvidenceItem:
    rank: int
    score: float
    chunk_id: str
    source_id: str
    source_page_id: str
    snippet: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceContext:
    query: str
    evidence: list[EvidenceItem]
    contract_version: str = "analysis-evidence-context-v1"
    task: str = "retrieval_context"
    provenance: dict[str, Any] = field(default_factory=dict)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def iter_fused_page_documents(path: Path) -> Iterator[SourceDocument]:
    """Yield retrieval documents from parser fused-page JSON contracts."""

    targets = [path] if path.is_file() else sorted(path.glob("*.json"))
    for target in targets:
        payload = _load_json(target)
        page_id = str(payload.get("page_id") or target.stem)
        transcript = str(payload.get("transcript") or "").strip()
        if not transcript:
            regions = payload.get("regions") or []
            if isinstance(regions, list):
                transcript = "\n".join(
                    str(region.get("text") or "").strip()
                    for region in regions
                    if isinstance(region, dict) and str(region.get("text") or "").strip()
                )
        if not transcript:
            continue
        provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
        quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
        yield SourceDocument(
            doc_id=page_id,
            text=transcript,
            metadata={
                "source_path": str(target),
                "page_id": page_id,
                "model_ids": list(payload.get("model_ids") or []),
                "quality": quality,
                "parser_provenance": provenance,
                "contract_source": "parser_fused_page",
            },
        )


def evidence_item_from_hit(hit: RetrievalHit) -> EvidenceItem:
    metadata = dict(hit.metadata)
    page_id = str(metadata.get("page_id") or hit.source_id)
    return EvidenceItem(
        rank=hit.rank,
        score=hit.score,
        chunk_id=hit.chunk_id,
        source_id=hit.source_id,
        source_page_id=page_id,
        snippet=hit.text,
        metadata=metadata,
    )


def build_evidence_contexts(
    *,
    documents: Iterable[SourceDocument],
    queries: Iterable[str],
    top_k: int = 10,
    chunk_words: int = 220,
    overlap_words: int = 40,
    provenance: dict[str, Any] | None = None,
) -> list[EvidenceContext]:
    doc_list = list(documents)
    query_list = [query for query in queries if str(query).strip()]
    index = LexicalIndex.from_documents(
        doc_list,
        chunk_words=chunk_words,
        overlap_words=overlap_words,
    )
    base_provenance = {
        "retriever": "bm25_lexical",
        "document_count": len(doc_list),
        "chunk_count": len(index.chunks),
        "chunk_words": chunk_words,
        "overlap_words": overlap_words,
        "top_k": top_k,
        "uses_external_llm_api": False,
    }
    if provenance:
        base_provenance.update(provenance)
    return [
        EvidenceContext(
            query=query,
            evidence=[evidence_item_from_hit(hit) for hit in index.search(query, top_k=top_k)],
            provenance=base_provenance,
        )
        for query in query_list
    ]


def evidence_context_to_row(context: EvidenceContext) -> dict[str, Any]:
    row = asdict(context)
    row["evidence_count"] = len(context.evidence)
    return row


def write_evidence_contexts_jsonl(path: Path, contexts: Iterable[EvidenceContext]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for context in contexts:
            handle.write(json.dumps(evidence_context_to_row(context), sort_keys=True) + "\n")
            count += 1
    return count
