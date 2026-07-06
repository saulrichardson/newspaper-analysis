from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from newsvlm_analysis.evidence import (
    build_evidence_contexts,
    iter_fused_page_documents,
    iter_parser_run_documents,
    write_evidence_contexts_jsonl,
)


def _write_fused_page(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "page_id": "issue-a__p0001__img-001",
                "model_ids": ["baseline_geometry_v1", "column_detector_v1"],
                "transcript": (
                    "The city council debated a zoning ordinance for apartment "
                    "height limits near the railroad station."
                ),
                "quality": {"region_count": 2},
                "provenance": {"parser_commit": "abc123"},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_parser_run(run_dir: Path) -> None:
    _write_fused_page(run_dir / "outputs" / "fused_pages" / "page.json")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "parser-run-001",
                "profile": "full",
                "page_count": 1,
                "model_ids": ["baseline_geometry_v1", "column_detector_v1"],
                "performance": {"pages_completed": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "provenance.json").write_text(
        json.dumps({"repo_commit": "abc123", "contract_version": "parser-bagging-v1"}) + "\n",
        encoding="utf-8",
    )
    validation_path = run_dir / "reports" / "validation.json"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(
        json.dumps({"status": "ok", "counts": {"errors": 0, "warnings": 0}, "issues": []}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "reports" / "input_manifest_validation.json").write_text(
        json.dumps(
            {
                "contract": "parse-input-manifest-validation-v1",
                "status": "ok",
                "counts": {"rows": 1, "errors": 0, "warnings": 0},
                "manifest_path": "/tmp/source_artifacts.jsonl",
                "issues": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_fused_page_documents_build_evidence_context(tmp_path: Path) -> None:
    fused_dir = tmp_path / "fused"
    _write_fused_page(fused_dir / "page.json")
    documents = list(iter_fused_page_documents(fused_dir))

    contexts = build_evidence_contexts(
        documents=documents,
        queries=["zoning apartment height"],
        top_k=1,
        chunk_words=40,
        overlap_words=0,
    )

    assert len(contexts) == 1
    assert contexts[0].contract_version == "analysis-evidence-context-v1"
    assert contexts[0].provenance["uses_external_llm_api"] is False
    assert contexts[0].evidence[0].source_page_id == "issue-a__p0001__img-001"
    assert contexts[0].evidence[0].metadata["model_ids"] == [
        "baseline_geometry_v1",
        "column_detector_v1",
    ]


def test_write_evidence_contexts_jsonl(tmp_path: Path) -> None:
    fused_dir = tmp_path / "fused"
    _write_fused_page(fused_dir / "page.json")
    contexts = build_evidence_contexts(
        documents=iter_fused_page_documents(fused_dir),
        queries=["railroad zoning"],
        top_k=1,
        chunk_words=40,
        overlap_words=0,
    )
    output = tmp_path / "contexts.jsonl"

    written = write_evidence_contexts_jsonl(output, contexts)

    row = json.loads(output.read_text(encoding="utf-8"))
    assert written == 1
    assert row["evidence_count"] == 1
    assert row["evidence"][0]["score"] > 0


def test_parser_run_documents_include_run_bundle_provenance(tmp_path: Path) -> None:
    run_dir = tmp_path / "parser-run"
    _write_parser_run(run_dir)

    documents = list(iter_parser_run_documents(run_dir))

    assert len(documents) == 1
    assert documents[0].metadata["contract_source"] == "parser_run_bundle"
    assert documents[0].metadata["parser_run_id"] == "parser-run-001"
    assert documents[0].metadata["parser_provenance"]["repo_commit"] == "abc123"
    assert documents[0].metadata["parser_input_manifest_validation"]["status"] == "ok"


def test_local_retrieval_script_accepts_fused_pages(tmp_path: Path) -> None:
    fused_dir = tmp_path / "fused"
    _write_fused_page(fused_dir / "page.json")
    output = tmp_path / "contexts.jsonl"
    script = Path("scripts/pipelines/build_local_retrieval_context.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--fused-pages",
            str(fused_dir),
            "--query",
            "zoning ordinance apartment",
            "--output-jsonl",
            str(output),
            "--output-format",
            "contexts",
            "--top-k",
            "1",
            "--chunk-words",
            "40",
            "--overlap-words",
            "0",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["output_format"] == "contexts"
    assert summary["rows_written"] == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["evidence"][0]["source_page_id"] == "issue-a__p0001__img-001"


def test_local_retrieval_script_accepts_parser_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "parser-run"
    _write_parser_run(run_dir)
    output = tmp_path / "contexts.jsonl"
    script = Path("scripts/pipelines/build_local_retrieval_context.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--parser-run-dir",
            str(run_dir),
            "--query",
            "zoning ordinance apartment",
            "--output-jsonl",
            str(output),
            "--output-format",
            "contexts",
            "--top-k",
            "1",
            "--chunk-words",
            "40",
            "--overlap-words",
            "0",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["rows_written"] == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["provenance"]["input_mode"] == "parser_run"
    assert row["provenance"]["parser_run_id"] == "parser-run-001"
    assert row["provenance"]["parser_validation"]["status"] == "warning"
    assert row["provenance"]["parser_input_manifest_validation"]["status"] == "ok"


def test_local_retrieval_script_writes_validation_sidecar(tmp_path: Path) -> None:
    run_dir = tmp_path / "parser-run"
    _write_parser_run(run_dir)
    output = tmp_path / "contexts.jsonl"
    validation = tmp_path / "validation.json"
    script = Path("scripts/pipelines/build_local_retrieval_context.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--parser-run-dir",
            str(run_dir),
            "--query",
            "zoning ordinance apartment",
            "--output-jsonl",
            str(output),
            "--output-format",
            "contexts",
            "--validation-json",
            str(validation),
            "--top-k",
            "1",
            "--chunk-words",
            "40",
            "--overlap-words",
            "0",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(validation.read_text(encoding="utf-8"))
    assert report["parser_run"]["status"] == "warning"
    assert report["parser_run"]["counts"]["input_manifest_rows"] == 1
    assert report["evidence_contexts"]["status"] == "ok"
