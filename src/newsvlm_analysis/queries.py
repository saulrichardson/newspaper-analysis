"""Query manifests and retrieval evaluation for offline analysis."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from newsvlm_analysis.local_retrieval import RetrievalHit


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    query: str
    task: str = "retrieval_context"
    relevant_page_ids: tuple[str, ...] = ()
    relevant_source_ids: tuple[str, ...] = ()
    relevant_chunk_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["relevant_page_ids"] = list(self.relevant_page_ids)
        payload["relevant_source_ids"] = list(self.relevant_source_ids)
        payload["relevant_chunk_ids"] = list(self.relevant_chunk_ids)
        return payload


def query_specs_from_strings(queries: Iterable[str], *, start: int = 1) -> list[QuerySpec]:
    specs: list[QuerySpec] = []
    for offset, raw_query in enumerate(queries, start=start):
        query = str(raw_query).strip()
        if query:
            specs.append(QuerySpec(query_id=f"q{offset:05d}", query=query))
    return specs


def _string_tuple(payload: dict[str, Any], key: str, *, line_number: int) -> tuple[str, ...]:
    value = payload.get(key) or []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"query manifest row {line_number} field {key} must be a list")
    return tuple(str(item).strip() for item in value if str(item).strip())


def iter_query_specs(path: Path) -> Iterator[QuerySpec]:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        position = 0
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            position += 1
            if stripped.startswith("{"):
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError(f"query manifest row {line_number} must be a JSON object")
                query = str(payload.get("query") or "").strip()
                query_id = str(payload.get("query_id") or f"q{position:05d}").strip()
                if not query:
                    raise ValueError(f"query manifest row {line_number} is missing query")
                if not query_id:
                    raise ValueError(f"query manifest row {line_number} is missing query_id")
                known = {
                    "query_id",
                    "query",
                    "task",
                    "relevant_page_ids",
                    "relevant_source_ids",
                    "relevant_chunk_ids",
                    "metadata",
                }
                raw_metadata = payload.get("metadata") or {}
                if not isinstance(raw_metadata, dict):
                    raise ValueError(f"query manifest row {line_number} field metadata must be an object")
                metadata = dict(raw_metadata)
                metadata.update({key: value for key, value in payload.items() if key not in known})
                yield QuerySpec(
                    query_id=query_id,
                    query=query,
                    task=str(payload.get("task") or "retrieval_context"),
                    relevant_page_ids=_string_tuple(payload, "relevant_page_ids", line_number=line_number),
                    relevant_source_ids=_string_tuple(payload, "relevant_source_ids", line_number=line_number),
                    relevant_chunk_ids=_string_tuple(payload, "relevant_chunk_ids", line_number=line_number),
                    metadata=metadata,
                )
            else:
                yield QuerySpec(query_id=f"q{position:05d}", query=stripped)


def validate_query_specs(specs: Iterable[QuerySpec]) -> list[QuerySpec]:
    values = list(specs)
    if not values:
        raise ValueError("at least one non-empty query is required")
    seen: set[str] = set()
    for spec in values:
        if not spec.query_id.strip():
            raise ValueError("query_id values must be non-empty")
        if not spec.query.strip():
            raise ValueError(f"query {spec.query_id!r} has empty text")
        if spec.query_id in seen:
            raise ValueError(f"duplicate query_id: {spec.query_id}")
        seen.add(spec.query_id)
    return values


def write_query_specs(path: Path, specs: Iterable[QuerySpec]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for spec in specs:
            handle.write(json.dumps(spec.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _targets(spec: QuerySpec) -> set[str]:
    return {
        *(f"page:{value}" for value in spec.relevant_page_ids),
        *(f"source:{value}" for value in spec.relevant_source_ids),
        *(f"chunk:{value}" for value in spec.relevant_chunk_ids),
    }


def _matched_targets(spec: QuerySpec, hit: RetrievalHit) -> set[str]:
    matched: set[str] = set()
    page_id = str(hit.metadata.get("page_id") or hit.source_id)
    if page_id in spec.relevant_page_ids:
        matched.add(f"page:{page_id}")
    if hit.source_id in spec.relevant_source_ids:
        matched.add(f"source:{hit.source_id}")
    if hit.chunk_id in spec.relevant_chunk_ids:
        matched.add(f"chunk:{hit.chunk_id}")
    return matched


def evaluate_retrieval(
    *,
    specs: Iterable[QuerySpec],
    hits_by_query_id: dict[str, list[RetrievalHit]],
    top_k: int,
) -> dict[str, Any]:
    """Evaluate ranked evidence using page, source, or chunk relevance labels."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    rows: list[dict[str, Any]] = []
    for spec in specs:
        relevant = _targets(spec)
        if not relevant:
            continue
        matched: set[str] = set()
        first_rank: int | None = None
        dcg = 0.0
        for rank, hit in enumerate(hits_by_query_id.get(spec.query_id, [])[:top_k], start=1):
            hit_targets = _matched_targets(spec, hit)
            new_targets = hit_targets - matched
            if hit_targets and first_rank is None:
                first_rank = rank
            if new_targets:
                dcg += 1.0 / math.log2(rank + 1)
                matched.update(new_targets)
        ideal_hits = min(len(relevant), top_k)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        rows.append(
            {
                "query_id": spec.query_id,
                "query": spec.query,
                "relevant_target_count": len(relevant),
                "matched_target_count": len(matched),
                "matched_targets": sorted(matched),
                "first_relevant_rank": first_rank,
                "hit_at_k": first_rank is not None,
                "reciprocal_rank": round(1.0 / first_rank, 6) if first_rank else 0.0,
                "recall_at_k": round(len(matched) / len(relevant), 6),
                "ndcg_at_k": round(dcg / ideal_dcg, 6) if ideal_dcg else 0.0,
            }
        )

    evaluated = len(rows)
    if not evaluated:
        return {
            "contract_version": "analysis-retrieval-evaluation-v1",
            "status": "not_requested",
            "top_k": top_k,
            "evaluated_queries": 0,
            "metrics": {},
            "queries": [],
        }

    return {
        "contract_version": "analysis-retrieval-evaluation-v1",
        "status": "ok",
        "top_k": top_k,
        "evaluated_queries": evaluated,
        "metrics": {
            "hit_rate_at_k": round(sum(bool(row["hit_at_k"]) for row in rows) / evaluated, 6),
            "mrr": round(sum(float(row["reciprocal_rank"]) for row in rows) / evaluated, 6),
            "mean_recall_at_k": round(sum(float(row["recall_at_k"]) for row in rows) / evaluated, 6),
            "mean_ndcg_at_k": round(sum(float(row["ndcg_at_k"]) for row in rows) / evaluated, 6),
        },
        "queries": rows,
    }
