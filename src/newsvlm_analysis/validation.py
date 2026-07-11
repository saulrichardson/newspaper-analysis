"""Validation helpers for offline analysis contracts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def _issue(
    issues: list[dict[str, Any]],
    *,
    level: str,
    code: str,
    message: str,
    path: Path | str | None = None,
    line: int | None = None,
) -> None:
    row: dict[str, Any] = {"level": level, "code": code, "message": message}
    if path is not None:
        row["path"] = str(path)
    if line is not None:
        row["line"] = line
    issues.append(row)


def _read_json(path: Path, issues: list[dict[str, Any]], *, required: bool = True) -> Any:
    if not path.exists():
        if required:
            _issue(issues, level="error", code="missing_file", message="required file is missing", path=path)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _issue(
            issues,
            level="error",
            code="invalid_json",
            message=f"invalid JSON at line {exc.lineno}: {exc.msg}",
            path=path,
        )
    return None


def _status(issues: list[dict[str, Any]], *, warnings_are_errors: bool = False) -> str:
    error_count = sum(1 for issue in issues if issue.get("level") == "error")
    warning_count = sum(1 for issue in issues if issue.get("level") == "warning")
    if error_count or (warnings_are_errors and warning_count):
        return "error"
    if warning_count:
        return "warning"
    return "ok"


def _counts(issues: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "errors": sum(1 for issue in issues if issue.get("level") == "error"),
        "warnings": sum(1 for issue in issues if issue.get("level") == "warning"),
    }


def _int_count(payload: dict[str, Any], key: str) -> int:
    try:
        return int(payload.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def validate_parser_run_bundle(
    run_dir: Path,
    *,
    require_validation_report: bool = False,
    warnings_are_errors: bool = False,
) -> dict[str, Any]:
    """Validate the parser run fields needed for evidence-first analysis.

    The parser repository owns the full run-bundle validator. This analysis-side
    check intentionally validates only the stable downstream contract: summary,
    provenance, fused pages, transcripts when present, and optional parser
    validation report status.
    """

    root = run_dir.expanduser().resolve()
    issues: list[dict[str, Any]] = []
    summary = _read_json(root / "summary.json", issues)
    provenance = _read_json(root / "provenance.json", issues, required=False)
    parser_validation = _read_json(root / "reports" / "validation.json", issues, required=False)
    input_manifest_validation = _read_json(
        root / "reports" / "input_manifest_validation.json",
        issues,
        required=False,
    )

    if not isinstance(summary, dict):
        summary = {}
    if provenance is not None and not isinstance(provenance, dict):
        _issue(
            issues,
            level="error",
            code="invalid_provenance",
            message="provenance.json must contain a JSON object",
            path=root / "provenance.json",
        )
        provenance = {}

    if parser_validation is None:
        if require_validation_report:
            _issue(
                issues,
                level="error",
                code="missing_parser_validation_report",
                message="parser validation report is required but missing",
                path=root / "reports" / "validation.json",
            )
    elif not isinstance(parser_validation, dict):
        _issue(
            issues,
            level="error",
            code="invalid_parser_validation_report",
            message="parser validation report must contain a JSON object",
            path=root / "reports" / "validation.json",
        )
    else:
        parser_status = str(parser_validation.get("status") or "")
        if parser_status != "ok":
            _issue(
                issues,
                level="error",
                code="parser_validation_not_ok",
                message=f"parser validation status is {parser_status or 'missing'}",
                path=root / "reports" / "validation.json",
            )

    if input_manifest_validation is None:
        if require_validation_report:
            _issue(
                issues,
                level="warning",
                code="missing_input_manifest_validation_report",
                message="parser input-manifest validation report is missing",
                path=root / "reports" / "input_manifest_validation.json",
            )
    elif not isinstance(input_manifest_validation, dict):
        _issue(
            issues,
            level="error",
            code="invalid_input_manifest_validation_report",
            message="parser input-manifest validation report must contain a JSON object",
            path=root / "reports" / "input_manifest_validation.json",
        )
        input_manifest_validation = {}
    else:
        input_manifest_status = str(input_manifest_validation.get("status") or "")
        if input_manifest_status not in ("ok", "warning"):
            _issue(
                issues,
                level="error",
                code="input_manifest_validation_not_ok",
                message=f"parser input-manifest validation status is {input_manifest_status or 'missing'}",
                path=root / "reports" / "input_manifest_validation.json",
            )

    fused_dir = root / "outputs" / "fused_pages"
    if not fused_dir.is_dir():
        _issue(
            issues,
            level="error",
            code="missing_fused_pages",
            message="parser run must contain outputs/fused_pages",
            path=fused_dir,
        )
        fused_paths: list[Path] = []
    else:
        fused_paths = sorted(fused_dir.glob("*.json"))
        if not fused_paths:
            _issue(
                issues,
                level="error",
                code="empty_fused_pages",
                message="outputs/fused_pages contains no JSON files",
                path=fused_dir,
            )

    summary_page_count = summary.get("page_count")
    if summary_page_count not in (None, ""):
        try:
            expected_pages = int(summary_page_count)
        except (TypeError, ValueError):
            expected_pages = -1
        if expected_pages >= 0 and expected_pages != len(fused_paths):
            _issue(
                issues,
                level="warning",
                code="page_count_differs_from_fused_pages",
                message=f"summary page_count={expected_pages} but fused_pages has {len(fused_paths)} files",
                path=root / "summary.json",
            )

    transcripts_dir = root / "outputs" / "transcripts"
    transcript_files = sorted(transcripts_dir.glob("*.txt")) if transcripts_dir.is_dir() else []
    if not transcripts_dir.exists():
        _issue(
            issues,
            level="warning",
            code="missing_transcripts_dir",
            message="outputs/transcripts is missing; fused page transcript fields will be used",
            path=transcripts_dir,
        )

    nonempty_fused_pages = 0
    model_ids: set[str] = set()
    for path in fused_paths:
        payload = _read_json(path, issues)
        if not isinstance(payload, dict):
            continue
        page_id = str(payload.get("page_id") or path.stem)
        transcript = str(payload.get("transcript") or "").strip()
        if not transcript:
            regions = payload.get("regions") if isinstance(payload.get("regions"), list) else []
            transcript = "\n".join(
                str(region.get("text") or "").strip()
                for region in regions
                if isinstance(region, dict) and str(region.get("text") or "").strip()
            )
        if transcript:
            nonempty_fused_pages += 1
        else:
            _issue(
                issues,
                level="error",
                code="empty_fused_transcript",
                message=f"fused page {page_id} has no transcript text or region text",
                path=path,
            )
        for model_id in payload.get("model_ids") or []:
            model_ids.add(str(model_id))

    summary_model_ids = {str(item) for item in summary.get("model_ids") or []}
    if summary_model_ids and model_ids and not model_ids.issubset(summary_model_ids):
        _issue(
            issues,
            level="warning",
            code="fused_model_ids_not_in_summary",
            message="some fused page model_ids are not listed in summary.json",
            path=fused_dir,
        )

    input_manifest_counts = (
        input_manifest_validation.get("counts")
        if isinstance(input_manifest_validation, dict) and isinstance(input_manifest_validation.get("counts"), dict)
        else {}
    )
    counts = {
        **_counts(issues),
        "fused_pages": len(fused_paths),
        "nonempty_fused_pages": nonempty_fused_pages,
        "transcript_files": len(transcript_files),
        "parser_models": len(summary_model_ids or model_ids),
        "input_manifest_rows": _int_count(input_manifest_counts, "rows"),
        "input_manifest_errors": _int_count(input_manifest_counts, "errors"),
        "input_manifest_warnings": _int_count(input_manifest_counts, "warnings"),
    }
    return {
        "contract": "analysis-parser-run-validation-v1",
        "status": _status(issues, warnings_are_errors=warnings_are_errors),
        "run_dir": str(root),
        "counts": counts,
        "parser_validation_report": str(root / "reports" / "validation.json") if parser_validation else "",
        "input_manifest_validation_report": (
            str(root / "reports" / "input_manifest_validation.json") if input_manifest_validation else ""
        ),
        "issues": issues,
    }


def validate_evidence_contexts_jsonl(
    path: Path,
    *,
    min_rows: int = 1,
    require_evidence: bool = True,
    warnings_are_errors: bool = False,
) -> dict[str, Any]:
    output = path.expanduser().resolve()
    issues: list[dict[str, Any]] = []
    rows = 0
    evidence_items = 0
    queries_with_evidence = 0
    contract_versions: set[str] = set()

    if not output.exists():
        _issue(issues, level="error", code="missing_output", message="evidence context JSONL is missing", path=output)
    else:
        for line_number, line in enumerate(output.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                _issue(
                    issues,
                    level="error",
                    code="invalid_jsonl",
                    message=f"invalid JSONL row: {exc.msg}",
                    path=output,
                    line=line_number,
                )
                continue
            if not isinstance(row, dict):
                _issue(
                    issues,
                    level="error",
                    code="invalid_context_row",
                    message="context row must be a JSON object",
                    path=output,
                    line=line_number,
                )
                continue
            rows += 1
            contract_versions.add(str(row.get("contract_version") or ""))
            if str(row.get("query") or "").strip() == "":
                _issue(
                    issues,
                    level="error",
                    code="missing_query",
                    message="context row is missing query text",
                    path=output,
                    line=line_number,
                )
            provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
            if provenance.get("uses_external_llm_api") is not False:
                _issue(
                    issues,
                    level="error",
                    code="external_llm_api_not_false",
                    message="context provenance must explicitly set uses_external_llm_api=false",
                    path=output,
                    line=line_number,
                )
            evidence = row.get("evidence")
            if not isinstance(evidence, list):
                _issue(
                    issues,
                    level="error",
                    code="invalid_evidence",
                    message="context evidence must be a list",
                    path=output,
                    line=line_number,
                )
                continue
            if evidence:
                queries_with_evidence += 1
            elif require_evidence:
                _issue(
                    issues,
                    level="error",
                    code="empty_evidence",
                    message="context row has no evidence items",
                    path=output,
                    line=line_number,
                )
            try:
                evidence_count = int(row.get("evidence_count") if row.get("evidence_count") is not None else len(evidence))
            except (TypeError, ValueError):
                evidence_count = -1
            if evidence_count != len(evidence):
                _issue(
                    issues,
                    level="error",
                    code="evidence_count_mismatch",
                    message="evidence_count does not match evidence list length",
                    path=output,
                    line=line_number,
                )
            for item_index, item in enumerate(evidence, start=1):
                if not isinstance(item, dict):
                    _issue(
                        issues,
                        level="error",
                        code="invalid_evidence_item",
                        message=f"evidence item {item_index} must be an object",
                        path=output,
                        line=line_number,
                    )
                    continue
                evidence_items += 1
                for key in ("rank", "score", "chunk_id", "source_id", "source_page_id", "snippet"):
                    if item.get(key) in (None, ""):
                        _issue(
                            issues,
                            level="error",
                            code="missing_evidence_field",
                            message=f"evidence item {item_index} is missing {key}",
                            path=output,
                            line=line_number,
                        )

    if rows < min_rows:
        _issue(
            issues,
            level="error",
            code="too_few_context_rows",
            message=f"expected at least {min_rows} context rows, found {rows}",
            path=output,
        )
    counts = {
        **_counts(issues),
        "rows": rows,
        "queries_with_evidence": queries_with_evidence,
        "evidence_items": evidence_items,
        "contract_versions": len({version for version in contract_versions if version}),
    }
    return {
        "contract": "analysis-evidence-context-validation-v1",
        "status": _status(issues, warnings_are_errors=warnings_are_errors),
        "output_jsonl": str(output),
        "counts": counts,
        "contract_versions": sorted(version for version in contract_versions if version),
        "issues": issues,
    }


def _count_jsonl_rows(path: Path, issues: list[dict[str, Any]]) -> int:
    if not path.is_file():
        _issue(issues, level="error", code="missing_file", message="required JSONL file is missing", path=path)
        return 0
    rows = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            _issue(
                issues,
                level="error",
                code="invalid_jsonl",
                message=f"invalid JSONL row: {exc.msg}",
                path=path,
                line=line_number,
            )
            continue
        if not isinstance(payload, dict):
            _issue(
                issues,
                level="error",
                code="invalid_jsonl_row",
                message="JSONL row must be an object",
                path=path,
                line=line_number,
            )
            continue
        rows += 1
    return rows


def validate_analysis_run_bundle(
    run_dir: Path,
    *,
    warnings_are_errors: bool = False,
) -> dict[str, Any]:
    """Validate the canonical persistent offline analysis run contract."""

    from newsvlm_analysis.persistent_index import inspect_persistent_index

    root = run_dir.expanduser().resolve()
    issues: list[dict[str, Any]] = []
    summary = _read_json(root / "summary.json", issues)
    config = _read_json(root / "config.json", issues)
    provenance = _read_json(root / "provenance.json", issues)
    parser_validation = _read_json(root / "reports" / "parser_validation.json", issues)
    performance = _read_json(root / "reports" / "performance.json", issues)
    evaluation = _read_json(root / "reports" / "retrieval_evaluation.json", issues)

    objects = {
        "summary": summary,
        "config": config,
        "provenance": provenance,
        "parser_validation": parser_validation,
        "performance": performance,
        "evaluation": evaluation,
    }
    for name, payload in objects.items():
        if payload is not None and not isinstance(payload, dict):
            _issue(
                issues,
                level="error",
                code=f"invalid_{name}",
                message=f"{name} must contain a JSON object",
            )
    summary = summary if isinstance(summary, dict) else {}
    config = config if isinstance(config, dict) else {}
    provenance = provenance if isinstance(provenance, dict) else {}
    parser_validation = parser_validation if isinstance(parser_validation, dict) else {}
    performance = performance if isinstance(performance, dict) else {}

    if summary.get("contract_version") != "offline-analysis-run-v1":
        _issue(
            issues,
            level="error",
            code="invalid_run_contract",
            message="summary contract_version must be offline-analysis-run-v1",
            path=root / "summary.json",
        )
    if summary.get("status") not in ("ok", None):
        _issue(
            issues,
            level="error",
            code="run_status_not_ok",
            message=f"analysis run status is {summary.get('status')}",
            path=root / "summary.json",
        )
    for name, payload, path in (
        ("config", config, root / "config.json"),
        ("provenance", provenance, root / "provenance.json"),
        ("summary", summary, root / "summary.json"),
    ):
        if payload.get("uses_external_llm_api") is not False:
            _issue(
                issues,
                level="error",
                code="external_llm_api_not_false",
                message=f"{name} must explicitly set uses_external_llm_api=false",
                path=path,
            )
    if parser_validation.get("status") == "error":
        _issue(
            issues,
            level="error",
            code="parser_validation_not_usable",
            message="analysis run records a failed parser bundle",
            path=root / "reports" / "parser_validation.json",
        )

    query_count = _count_jsonl_rows(root / "inputs" / "queries.jsonl", issues)
    evidence_report = validate_evidence_contexts_jsonl(
        root / "outputs" / "evidence_contexts.jsonl",
        min_rows=1,
        require_evidence=False,
    )
    if evidence_report["status"] == "error":
        issues.extend(evidence_report["issues"])

    index_info: dict[str, Any] = {}
    index_path = root / "index" / "corpus.sqlite3"
    try:
        index_info = inspect_persistent_index(index_path)
        if index_info.get("contract_version") != "analysis-sqlite-fts-index-v1":
            _issue(
                issues,
                level="error",
                code="invalid_index_contract",
                message="persistent index contract is missing or unsupported",
                path=index_path,
            )
        if int(index_info.get("chunk_count") or 0) <= 0:
            _issue(
                issues,
                level="error",
                code="empty_index",
                message="persistent analysis index contains no chunks",
                path=index_path,
            )
        if index_info.get("integrity") != "ok":
            _issue(
                issues,
                level="error",
                code="index_integrity_failed",
                message=f"SQLite quick_check returned {index_info.get('integrity')}",
                path=index_path,
            )
        if int(index_info.get("fts_chunk_count") or -1) != int(index_info.get("chunk_count") or 0):
            _issue(
                issues,
                level="error",
                code="fts_chunk_count_mismatch",
                message="FTS and relational chunk counts do not match",
                path=index_path,
            )
    except (FileNotFoundError, OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        _issue(
            issues,
            level="error",
            code="invalid_index",
            message=str(exc),
            path=index_path,
        )

    evidence_rows = int(evidence_report.get("counts", {}).get("rows") or 0)
    if query_count != evidence_rows:
        _issue(
            issues,
            level="error",
            code="query_context_count_mismatch",
            message=f"queries.jsonl has {query_count} rows but evidence output has {evidence_rows}",
        )
    summary_counts = summary.get("counts") if isinstance(summary.get("counts"), dict) else {}
    if summary_counts and _int_count(summary_counts, "queries") != query_count:
        _issue(
            issues,
            level="error",
            code="summary_query_count_mismatch",
            message="summary query count does not match queries.jsonl",
            path=root / "summary.json",
        )
    stage_seconds = performance.get("stage_seconds") if isinstance(performance.get("stage_seconds"), dict) else {}
    for required_stage in ("validate_parser_bundle", "load_parser_documents", "build_index", "query_index"):
        if required_stage not in stage_seconds:
            _issue(
                issues,
                level="error",
                code="missing_performance_stage",
                message=f"performance report is missing stage {required_stage}",
                path=root / "reports" / "performance.json",
            )

    errors_path = root / "errors.jsonl"
    errors_rows = _count_jsonl_rows(errors_path, issues)
    if errors_rows:
        _issue(
            issues,
            level="error",
            code="run_errors_present",
            message=f"errors.jsonl contains {errors_rows} rows",
            path=errors_path,
        )

    counts = {
        **_counts(issues),
        "queries": query_count,
        "evidence_contexts": evidence_rows,
        "evidence_items": int(evidence_report.get("counts", {}).get("evidence_items") or 0),
        "index_chunks": int(index_info.get("chunk_count") or 0),
        "index_sources": int(index_info.get("source_count") or 0),
        "run_errors": errors_rows,
    }
    return {
        "contract": "offline-analysis-run-validation-v1",
        "status": _status(issues, warnings_are_errors=warnings_are_errors),
        "run_dir": str(root),
        "counts": counts,
        "index": index_info,
        "evidence_contexts": evidence_report,
        "issues": issues,
    }
