"""Canonical offline analysis run orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

from newsvlm_analysis.evidence import (
    evidence_context_from_hits,
    iter_parser_run_documents,
    parser_run_provenance,
    write_evidence_contexts_jsonl,
)
from newsvlm_analysis.persistent_index import PersistentLexicalIndex, build_persistent_index
from newsvlm_analysis.queries import QuerySpec, evaluate_retrieval, validate_query_specs, write_query_specs
from newsvlm_analysis.validation import validate_analysis_run_bundle, validate_parser_run_bundle


ANALYSIS_RUN_CONTRACT = "offline-analysis-run-v1"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def _write_errors(path: Path, errors: Iterable[dict[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "".join(json.dumps(error, ensure_ascii=False, sort_keys=True) + "\n" for error in errors),
    )


def _git_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_fingerprints(parser_run_dir: Path) -> dict[str, str]:
    paths = {
        "summary_json": parser_run_dir / "summary.json",
        "provenance_json": parser_run_dir / "provenance.json",
        "parse_input_jsonl": parser_run_dir / "manifests" / "parse_input.jsonl",
    }
    return {name: _sha256_file(path) for name, path in paths.items() if path.is_file()}


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _prepare_run_dir(run_dir: Path) -> Path:
    root = run_dir.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"analysis run directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    for relative in ("inputs", "index", "outputs", "reports"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root


def run_offline_analysis(
    *,
    parser_run_dir: Path,
    run_dir: Path,
    query_specs: Iterable[QuerySpec],
    top_k: int = 10,
    chunk_words: int = 220,
    overlap_words: int = 40,
) -> dict[str, Any]:
    """Build, query, evaluate, and validate one self-contained analysis run."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    specs = validate_query_specs(query_specs)
    parser_root = parser_run_dir.expanduser().resolve()
    root = _prepare_run_dir(run_dir)
    started = time.perf_counter()
    stage_seconds: dict[str, float] = {}
    errors_path = root / "errors.jsonl"
    _write_errors(errors_path, [])

    try:
        stage_started = time.perf_counter()
        parser_validation = validate_parser_run_bundle(
            parser_root,
            require_validation_report=True,
        )
        stage_seconds["validate_parser_bundle"] = time.perf_counter() - stage_started
        _write_json(root / "reports" / "parser_validation.json", parser_validation)
        if parser_validation["status"] == "error":
            raise ValueError("parser run bundle failed analysis-side validation")

        stage_started = time.perf_counter()
        documents = list(iter_parser_run_documents(parser_root, validation_report=parser_validation))
        stage_seconds["load_parser_documents"] = time.perf_counter() - stage_started
        if not documents:
            raise ValueError("parser run contains no non-empty fused-page documents")

        write_query_specs(root / "inputs" / "queries.jsonl", specs)
        config = {
            "contract_version": "offline-analysis-config-v1",
            "parser_run_dir": str(parser_root),
            "top_k": top_k,
            "chunk_words": chunk_words,
            "overlap_words": overlap_words,
            "retriever": "sqlite_fts5_bm25",
            "uses_external_llm_api": False,
        }
        _write_json(root / "config.json", config)

        parser_provenance = parser_run_provenance(parser_root, validation_report=parser_validation)
        index_path = root / "index" / "corpus.sqlite3"
        index_result = build_persistent_index(
            documents=documents,
            index_path=index_path,
            chunk_words=chunk_words,
            overlap_words=overlap_words,
            metadata={
                "source_contract": "parser-run-bundle",
                "parser_run_id": parser_provenance.get("parser_run_id", ""),
                "parser_profile": parser_provenance.get("parser_profile", ""),
                "parser_model_ids": parser_provenance.get("parser_model_ids", []),
            },
        )
        stage_seconds["build_index"] = index_result.seconds

        contexts = []
        hits_by_query_id = {}
        query_latencies: list[float] = []
        with PersistentLexicalIndex(index_path) as index:
            index_info = index.inspect()
            for spec in specs:
                query_started = time.perf_counter()
                hits = index.search(spec.query, top_k=top_k)
                query_latencies.append(time.perf_counter() - query_started)
                hits_by_query_id[spec.query_id] = hits
                contexts.append(
                    evidence_context_from_hits(
                        query=spec.query,
                        query_id=spec.query_id,
                        task=spec.task,
                        hits=hits,
                        provenance={
                            "analysis_run_id": root.name,
                            "retriever": "sqlite_fts5_bm25",
                            "index_contract": index_info["contract_version"],
                            "document_count": index_result.document_count,
                            "chunk_count": index_result.chunk_count,
                            "chunk_words": chunk_words,
                            "overlap_words": overlap_words,
                            "top_k": top_k,
                            "parser_run_id": parser_provenance.get("parser_run_id", ""),
                            "parser_model_ids": parser_provenance.get("parser_model_ids", []),
                            "uses_external_llm_api": False,
                        },
                    )
                )

        evidence_path = root / "outputs" / "evidence_contexts.jsonl"
        write_evidence_contexts_jsonl(evidence_path, contexts)
        stage_seconds["query_index"] = sum(query_latencies)

        stage_started = time.perf_counter()
        evaluation = evaluate_retrieval(specs=specs, hits_by_query_id=hits_by_query_id, top_k=top_k)
        stage_seconds["evaluate_retrieval"] = time.perf_counter() - stage_started
        _write_json(root / "reports" / "retrieval_evaluation.json", evaluation)

        repo_root = Path(__file__).resolve().parents[2]
        provenance = {
            "contract_version": "offline-analysis-provenance-v1",
            "analysis_commit": _git_commit(repo_root),
            "parser": parser_provenance,
            "source_fingerprints_sha256": _source_fingerprints(parser_root),
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "sqlite": sqlite3.sqlite_version,
            },
            "uses_external_llm_api": False,
        }
        _write_json(root / "provenance.json", provenance)

        elapsed = time.perf_counter() - started
        query_count = len(specs)
        query_total = sum(query_latencies)
        performance = {
            "contract_version": "offline-analysis-performance-v1",
            "seconds_total": round(elapsed, 6),
            "stage_seconds": {key: round(value, 6) for key, value in stage_seconds.items()},
            "index": index_result.to_dict(),
            "queries": {
                "count": query_count,
                "seconds_total": round(query_total, 6),
                "queries_per_second": round(query_count / query_total, 3) if query_total > 0 else 0.0,
                "latency_ms_mean": round((query_total / query_count) * 1_000, 3),
                "latency_ms_p50": round(_percentile(query_latencies, 0.50) * 1_000, 3),
                "latency_ms_p95": round(_percentile(query_latencies, 0.95) * 1_000, 3),
            },
        }
        _write_json(root / "reports" / "performance.json", performance)

        evidence_items = sum(len(context.evidence) for context in contexts)
        summary = {
            "contract_version": ANALYSIS_RUN_CONTRACT,
            "status": "ok",
            "run_id": root.name,
            "run_dir": str(root),
            "parser_run_dir": str(parser_root),
            "counts": {
                "documents": index_result.document_count,
                "chunks": index_result.chunk_count,
                "queries": query_count,
                "queries_with_evidence": sum(bool(context.evidence) for context in contexts),
                "evidence_items": evidence_items,
                "errors": 0,
            },
            "paths": {
                "config_json": str(root / "config.json"),
                "queries_jsonl": str(root / "inputs" / "queries.jsonl"),
                "index_sqlite3": str(index_path),
                "evidence_contexts_jsonl": str(evidence_path),
                "parser_validation_json": str(root / "reports" / "parser_validation.json"),
                "retrieval_evaluation_json": str(root / "reports" / "retrieval_evaluation.json"),
                "performance_json": str(root / "reports" / "performance.json"),
                "validation_json": str(root / "reports" / "validation.json"),
                "provenance_json": str(root / "provenance.json"),
                "errors_jsonl": str(errors_path),
            },
            "performance": {
                "seconds_total": performance["seconds_total"],
                "chunks_per_second": index_result.chunks_per_second,
                "queries_per_second": performance["queries"]["queries_per_second"],
            },
            "evaluation": {
                "status": evaluation["status"],
                "metrics": evaluation["metrics"],
            },
            "uses_external_llm_api": False,
        }
        _write_json(root / "summary.json", summary)

        validation_started = time.perf_counter()
        run_validation = validate_analysis_run_bundle(root)
        performance["stage_seconds"]["validate_analysis_bundle"] = round(
            time.perf_counter() - validation_started,
            6,
        )
        performance["seconds_total"] = round(time.perf_counter() - started, 6)
        _write_json(root / "reports" / "performance.json", performance)
        _write_json(root / "reports" / "validation.json", run_validation)
        summary["status"] = "ok" if run_validation["status"] != "error" else "error"
        summary["validation"] = {
            "status": run_validation["status"],
            "counts": run_validation["counts"],
        }
        summary["performance"]["seconds_total"] = performance["seconds_total"]
        _write_json(root / "summary.json", summary)
        return summary
    except Exception as exc:
        error = {
            "stage": "offline_analysis_run",
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_errors(errors_path, [error])
        failure_summary = {
            "contract_version": ANALYSIS_RUN_CONTRACT,
            "status": "error",
            "run_id": root.name,
            "run_dir": str(root),
            "parser_run_dir": str(parser_root),
            "counts": {"errors": 1},
            "error": error,
            "uses_external_llm_api": False,
        }
        _write_json(root / "summary.json", failure_summary)
        raise
