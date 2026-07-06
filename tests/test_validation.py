from __future__ import annotations

import json
from pathlib import Path

from newsvlm_analysis.validation import (
    validate_evidence_contexts_jsonl,
    validate_parser_run_bundle,
)


def _write_parser_run(run_dir: Path) -> None:
    fused = run_dir / "outputs" / "fused_pages" / "page-001.json"
    fused.parent.mkdir(parents=True, exist_ok=True)
    fused.write_text(
        json.dumps(
            {
                "page_id": "page-001",
                "model_ids": ["baseline_geometry_v1"],
                "transcript": "A zoning ordinance limited apartment height.",
                "quality": {"region_count": 1},
                "provenance": {"contract_source": "fixture"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "parser-fixture",
                "profile": "baseline",
                "page_count": 1,
                "model_ids": ["baseline_geometry_v1"],
                "performance": {"pages_completed": 1, "errors": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "provenance.json").write_text(
        json.dumps({"contract_version": "parser-bagging-v1", "repo_commit": "abc123"}) + "\n",
        encoding="utf-8",
    )
    validation = run_dir / "reports" / "validation.json"
    validation.parent.mkdir(parents=True, exist_ok=True)
    validation.write_text(
        json.dumps(
            {
                "contract": "parser-run-bundle-validation-v1",
                "status": "ok",
                "counts": {"errors": 0, "warnings": 0},
                "issues": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "reports" / "input_manifest_validation.json").write_text(
        json.dumps(
            {
                "contract": "parse-input-manifest-validation-v1",
                "status": "ok",
                "counts": {"rows": 1, "errors": 0, "warnings": 0},
                "issues": [],
                "manifest_path": "/tmp/source_artifacts.jsonl",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_validate_parser_run_bundle_accepts_analysis_surface(tmp_path: Path) -> None:
    run_dir = tmp_path / "parser-run"
    _write_parser_run(run_dir)

    report = validate_parser_run_bundle(run_dir, require_validation_report=True)

    assert report["status"] == "warning"
    assert report["counts"]["fused_pages"] == 1
    assert report["counts"]["nonempty_fused_pages"] == 1
    assert any(issue["code"] == "missing_transcripts_dir" for issue in report["issues"])


def test_validate_parser_run_bundle_rejects_bad_parser_validation(tmp_path: Path) -> None:
    run_dir = tmp_path / "parser-run"
    _write_parser_run(run_dir)
    (run_dir / "reports" / "validation.json").write_text(
        json.dumps({"status": "error", "issues": [{"code": "bad"}]}) + "\n",
        encoding="utf-8",
    )

    report = validate_parser_run_bundle(run_dir, require_validation_report=True)

    assert report["status"] == "error"
    assert any(issue["code"] == "parser_validation_not_ok" for issue in report["issues"])


def test_validate_parser_run_bundle_rejects_bad_input_manifest_validation(tmp_path: Path) -> None:
    run_dir = tmp_path / "parser-run"
    _write_parser_run(run_dir)
    (run_dir / "reports" / "input_manifest_validation.json").write_text(
        json.dumps({"status": "error", "counts": {"rows": 1, "errors": 1}, "issues": [{"code": "bad"}]}) + "\n",
        encoding="utf-8",
    )

    report = validate_parser_run_bundle(run_dir, require_validation_report=True)

    assert report["status"] == "error"
    assert any(issue["code"] == "input_manifest_validation_not_ok" for issue in report["issues"])


def test_validate_evidence_contexts_jsonl_accepts_contexts(tmp_path: Path) -> None:
    path = tmp_path / "contexts.jsonl"
    path.write_text(
        json.dumps(
            {
                "contract_version": "analysis-evidence-context-v1",
                "query": "zoning apartment",
                "evidence_count": 1,
                "provenance": {"uses_external_llm_api": False},
                "evidence": [
                    {
                        "rank": 1,
                        "score": 1.2,
                        "chunk_id": "page-001#chunk-00000",
                        "source_id": "page-001",
                        "source_page_id": "page-001",
                        "snippet": "zoning ordinance text",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_evidence_contexts_jsonl(path)

    assert report["status"] == "ok"
    assert report["counts"]["rows"] == 1
    assert report["counts"]["evidence_items"] == 1


def test_validate_evidence_contexts_jsonl_rejects_external_llm_flag(tmp_path: Path) -> None:
    path = tmp_path / "contexts.jsonl"
    path.write_text(
        json.dumps(
            {
                "contract_version": "analysis-evidence-context-v1",
                "query": "zoning apartment",
                "evidence_count": 0,
                "provenance": {"uses_external_llm_api": True},
                "evidence": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_evidence_contexts_jsonl(path)

    assert report["status"] == "error"
    assert any(issue["code"] == "external_llm_api_not_false" for issue in report["issues"])
    assert any(issue["code"] == "empty_evidence" for issue in report["issues"])
