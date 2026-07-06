"""Validation helpers for offline analysis contracts."""

from __future__ import annotations

import json
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

    counts = {
        **_counts(issues),
        "fused_pages": len(fused_paths),
        "nonempty_fused_pages": nonempty_fused_pages,
        "transcript_files": len(transcript_files),
        "parser_models": len(summary_model_ids or model_ids),
    }
    return {
        "contract": "analysis-parser-run-validation-v1",
        "status": _status(issues, warnings_are_errors=warnings_are_errors),
        "run_dir": str(root),
        "counts": counts,
        "parser_validation_report": str(root / "reports" / "validation.json") if parser_validation else "",
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
