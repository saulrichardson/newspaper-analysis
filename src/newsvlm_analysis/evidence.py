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
from newsvlm_analysis.validation import validate_parser_run_bundle


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


def parser_run_provenance(
    run_dir: Path,
    *,
    validation_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    summary_path = run_dir / "summary.json"
    provenance_path = run_dir / "provenance.json"
    summary = _load_json(summary_path) if summary_path.is_file() else {}
    provenance = _load_json(provenance_path) if provenance_path.is_file() else {}
    out = {
        "parser_run_dir": str(run_dir),
        "parser_run_id": str(summary.get("run_id") or run_dir.name),
        "parser_profile": str(summary.get("profile") or ""),
        "parser_model_ids": list(summary.get("model_ids") or []),
        "parser_page_count": int(summary.get("page_count") or 0),
        "parser_performance": dict(summary.get("performance") or {}),
        "parser_provenance": provenance,
    }
    if validation_report is not None:
        out["parser_validation"] = {
            "status": validation_report.get("status"),
            "counts": dict(validation_report.get("counts") or {}),
            "issues": list(validation_report.get("issues") or []),
            "contract": validation_report.get("contract"),
        }
    return out


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


def iter_parser_run_documents(
    run_dir: Path,
    *,
    validation_report: dict[str, Any] | None = None,
) -> Iterator[SourceDocument]:
    run_dir = run_dir.expanduser().resolve()
    fused_pages = run_dir / "outputs" / "fused_pages"
    if not fused_pages.is_dir():
        raise FileNotFoundError(f"parser run does not contain outputs/fused_pages: {run_dir}")
    run_metadata = parser_run_provenance(run_dir, validation_report=validation_report)
    for document in iter_fused_page_documents(fused_pages):
        metadata = dict(document.metadata)
        metadata.update(run_metadata)
        metadata["contract_source"] = "parser_run_bundle"
        yield SourceDocument(
            doc_id=document.doc_id,
            text=document.text,
            metadata=metadata,
        )


def validate_parser_run_for_analysis(
    run_dir: Path,
    *,
    require_validation_report: bool = False,
    warnings_are_errors: bool = False,
) -> dict[str, Any]:
    return validate_parser_run_bundle(
        run_dir,
        require_validation_report=require_validation_report,
        warnings_are_errors=warnings_are_errors,
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
