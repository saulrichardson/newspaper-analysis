from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from newsvlm_analysis.analysis_run import run_offline_analysis
from newsvlm_analysis.persistent_index import PersistentLexicalIndex, fts5_available
from newsvlm_analysis.queries import QuerySpec
from newsvlm_analysis.validation import validate_analysis_run_bundle


pytestmark = pytest.mark.skipif(not fts5_available(), reason="Python SQLite does not provide FTS5")


PAGES = {
    "page-zoning": "The city adopted a zoning ordinance limiting apartment height to six stories.",
    "page-parking": "The ordinance required one parking space for each apartment dwelling unit.",
    "page-school": "The school board discussed classroom repairs and the annual library budget.",
}


def _write_parser_run(run_dir: Path) -> None:
    fused_dir = run_dir / "outputs" / "fused_pages"
    transcript_dir = run_dir / "outputs" / "transcripts"
    reports_dir = run_dir / "reports"
    manifests_dir = run_dir / "manifests"
    for path in (fused_dir, transcript_dir, reports_dir, manifests_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for page_number, (page_id, text) in enumerate(PAGES.items(), start=1):
        fused = {
            "page_id": page_id,
            "model_ids": ["layout-a", "ocr-a"],
            "regions": [
                {
                    "region_id": f"{page_id}-r1",
                    "bbox_xyxy": [10, 20, 600, 180],
                    "label": "text",
                    "confidence": 0.95,
                    "source_model": "ocr-a",
                    "text": text,
                    "reading_order": 1,
                }
            ],
            "transcript": text,
            "disagreement_score": 0.05,
            "quality": {"region_count": 1},
            "provenance": {"model_ids": ["layout-a", "ocr-a"]},
        }
        (fused_dir / f"{page_id}.json").write_text(json.dumps(fused) + "\n", encoding="utf-8")
        (transcript_dir / f"{page_id}.txt").write_text(text + "\n", encoding="utf-8")
        manifest_rows.append(
            {
                "page_id": page_id,
                "image_path": f"/fixture/{page_id}.png",
                "issue_id": "issue-1901-01-02",
                "page_number": page_number,
                "checksum_sha256": "a" * 64,
                "source": {
                    "source_system": "fixture",
                    "source_id": f"source-{page_id}",
                },
            }
        )
    (manifests_dir / "parse_input.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "parser-fixture",
                "profile": "full",
                "page_count": len(PAGES),
                "model_ids": ["layout-a", "ocr-a"],
                "performance": {"pages_completed": len(PAGES), "errors": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "provenance.json").write_text(
        json.dumps({"contract_version": "parser-bagging-v1", "repo_commit": "parser123"}) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "validation.json").write_text(
        json.dumps({"status": "ok", "counts": {"errors": 0, "warnings": 0}, "issues": []}) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "input_manifest_validation.json").write_text(
        json.dumps(
            {
                "contract": "parse-input-manifest-validation-v1",
                "status": "ok",
                "manifest_path": str(manifests_dir / "parse_input.jsonl"),
                "counts": {"rows": len(PAGES), "errors": 0, "warnings": 0},
                "issues": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _query_specs() -> list[QuerySpec]:
    return [
        QuerySpec(
            query_id="height-policy",
            query="zoning apartment height",
            task="policy_retrieval",
            relevant_page_ids=("page-zoning",),
        ),
        QuerySpec(
            query_id="parking-policy",
            query="parking requirement dwelling unit",
            task="policy_retrieval",
            relevant_page_ids=("page-parking",),
        ),
    ]


def test_offline_analysis_run_is_searchable_evaluated_and_validated(tmp_path: Path) -> None:
    parser_run = tmp_path / "parser-run"
    analysis_run = tmp_path / "analysis-run"
    _write_parser_run(parser_run)

    summary = run_offline_analysis(
        parser_run_dir=parser_run,
        run_dir=analysis_run,
        query_specs=_query_specs(),
        top_k=2,
        chunk_words=40,
        overlap_words=0,
    )

    assert summary["status"] == "ok"
    assert summary["counts"]["documents"] == 3
    assert summary["evaluation"]["metrics"]["hit_rate_at_k"] == 1.0
    assert summary["uses_external_llm_api"] is False
    validation = validate_analysis_run_bundle(analysis_run)
    assert validation["status"] == "ok"
    assert validation["counts"]["queries"] == 2
    assert validation["counts"]["index_chunks"] == 3

    with PersistentLexicalIndex(analysis_run / "index" / "corpus.sqlite3") as index:
        hit = index.search("apartment height", top_k=1)[0]
    assert hit.metadata["issue_id"] == "issue-1901-01-02"
    assert hit.metadata["source"]["source_system"] == "fixture"
    assert hit.metadata["parser_run_id"] == "parser-fixture"


def test_offline_analysis_failure_leaves_structured_error(tmp_path: Path) -> None:
    parser_run = tmp_path / "invalid-parser-run"
    parser_run.mkdir()
    analysis_run = tmp_path / "failed-analysis-run"

    with pytest.raises(ValueError, match="failed analysis-side validation"):
        run_offline_analysis(
            parser_run_dir=parser_run,
            run_dir=analysis_run,
            query_specs=_query_specs(),
        )

    summary = json.loads((analysis_run / "summary.json").read_text(encoding="utf-8"))
    error = json.loads((analysis_run / "errors.jsonl").read_text(encoding="utf-8"))
    assert summary["status"] == "error"
    assert error["error_type"] == "ValueError"
    assert summary["uses_external_llm_api"] is False


def test_run_validator_reports_corrupt_index_without_raising(tmp_path: Path) -> None:
    parser_run = tmp_path / "parser-run"
    analysis_run = tmp_path / "analysis-run"
    _write_parser_run(parser_run)
    run_offline_analysis(
        parser_run_dir=parser_run,
        run_dir=analysis_run,
        query_specs=_query_specs(),
        top_k=2,
        chunk_words=40,
        overlap_words=0,
    )
    (analysis_run / "index" / "corpus.sqlite3").write_bytes(b"not a sqlite database")

    report = validate_analysis_run_bundle(analysis_run)

    assert report["status"] == "error"
    assert any(issue["code"] == "invalid_index" for issue in report["issues"])


def test_analysis_cli_runs_and_searches_bundle(tmp_path: Path) -> None:
    parser_run = tmp_path / "parser-run"
    analysis_run = tmp_path / "analysis-run"
    queries = tmp_path / "queries.jsonl"
    _write_parser_run(parser_run)
    queries.write_text(
        "".join(json.dumps(spec.to_dict()) + "\n" for spec in _query_specs()),
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "newsvlm_analysis",
            "run",
            "--parser-run-dir",
            str(parser_run),
            "--run-dir",
            str(analysis_run),
            "--queries-jsonl",
            str(queries),
            "--top-k",
            "2",
            "--chunk-words",
            "40",
            "--overlap-words",
            "0",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "ok"
    search = subprocess.run(
        [
            sys.executable,
            "-m",
            "newsvlm_analysis",
            "search",
            "--index",
            str(analysis_run / "index" / "corpus.sqlite3"),
            "--query",
            "parking dwelling unit",
            "--top-k",
            "1",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert search.returncode == 0, search.stderr
    assert json.loads(search.stdout)["hits"][0]["source_id"] == "page-parking"
