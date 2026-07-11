from __future__ import annotations

import json
from pathlib import Path

import pytest

from newsvlm_analysis.local_retrieval import RetrievalHit
from newsvlm_analysis.queries import QuerySpec, evaluate_retrieval, iter_query_specs, validate_query_specs


def _hit(rank: int, page_id: str) -> RetrievalHit:
    return RetrievalHit(
        query="zoning height",
        rank=rank,
        score=1.0 / rank,
        chunk_id=f"{page_id}#chunk-00000",
        source_id=page_id,
        text="evidence",
        metadata={"page_id": page_id},
    )


def test_query_manifest_supports_gold_relevance_labels(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "zoning-height",
                "query": "zoning apartment height",
                "task": "policy_retrieval",
                "relevant_page_ids": ["page-a"],
                "city": "Example City",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    specs = list(iter_query_specs(path))

    assert specs[0].query_id == "zoning-height"
    assert specs[0].relevant_page_ids == ("page-a",)
    assert specs[0].metadata["city"] == "Example City"


def test_retrieval_evaluation_reports_ranked_metrics() -> None:
    spec = QuerySpec(
        query_id="q1",
        query="zoning height",
        relevant_page_ids=("page-a", "page-c"),
    )

    report = evaluate_retrieval(
        specs=[spec],
        hits_by_query_id={"q1": [_hit(1, "page-b"), _hit(2, "page-a"), _hit(3, "page-c")]},
        top_k=2,
    )

    assert report["status"] == "ok"
    assert report["metrics"]["hit_rate_at_k"] == 1.0
    assert report["metrics"]["mrr"] == 0.5
    assert report["metrics"]["mean_recall_at_k"] == 0.5
    assert 0 < report["metrics"]["mean_ndcg_at_k"] < 1


def test_query_validation_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate query_id"):
        validate_query_specs(
            [
                QuerySpec(query_id="same", query="first"),
                QuerySpec(query_id="same", query="second"),
            ]
        )


def test_query_manifest_rejects_scalar_relevance_fields(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text(
        json.dumps({"query": "zoning height", "relevant_page_ids": "page-a"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relevant_page_ids must be a list"):
        list(iter_query_specs(path))


def test_ndcg_stays_bounded_for_multi_level_labels() -> None:
    spec = QuerySpec(
        query_id="q1",
        query="zoning height",
        relevant_page_ids=("page-a",),
        relevant_source_ids=("page-a",),
    )

    report = evaluate_retrieval(
        specs=[spec],
        hits_by_query_id={"q1": [_hit(1, "page-a")]},
        top_k=2,
    )

    assert report["metrics"]["mean_ndcg_at_k"] <= 1.0
