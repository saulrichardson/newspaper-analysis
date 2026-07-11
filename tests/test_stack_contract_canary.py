from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_stack_contract_canary_runs_against_sibling_parser(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser_repo = repo_root.parent / "newspaper-parsing"
    if not (parser_repo / "src" / "newsbag").is_dir():
        pytest.skip("sibling newspaper-parsing checkout is not available")
    script = repo_root / "scripts" / "pipelines" / "run_stack_contract_canary.py"
    run_root = tmp_path / "stack-canary"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--parser-repo",
            str(parser_repo),
            "--run-root",
            str(run_root),
            "--profile",
            "baseline",
            "--query",
            "zoning apartment height",
            "--top-k",
            "1",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["status"] == "ok"
    assert summary["uses_external_llm_api"] is False
    assert summary["parser"]["manifest_validation"]["status"] == "ok"
    assert summary["parser"]["run_validation"]["status"] == "ok"
    assert summary["contract"] == "newspaper-stack-contract-canary-v2"
    parser_analysis = json.loads(
        (Path(summary["analysis"]["run_dir"]) / "reports" / "parser_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert parser_analysis["status"] == "ok"
    assert parser_analysis["counts"]["input_manifest_rows"] == 1
    assert summary["analysis"]["validation"]["status"] == "ok"
    assert summary["analysis"]["evaluation"]["metrics"]["hit_rate_at_k"] == 1.0
    evidence_contexts = Path(summary["analysis"]["evidence_contexts_jsonl"])
    row = json.loads(evidence_contexts.read_text(encoding="utf-8"))
    assert row["provenance"]["parser_run_id"] == "parser_run"
    assert row["evidence_count"] == 1
    assert (run_root / "stack_summary.json").is_file()
