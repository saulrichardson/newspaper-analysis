#!/usr/bin/env python3
"""Run an API-free acquisition -> parser -> analysis contract canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
DEFAULT_PARSER_REPO = REPO_ROOT.parent / "newspaper-parsing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parser-repo",
        type=Path,
        default=DEFAULT_PARSER_REPO,
        help="Path to a newspaper-parsing checkout. Default: sibling ../newspaper-parsing.",
    )
    parser.add_argument(
        "--parser-python",
        default=None,
        help="Python executable for parser commands. Default: first candidate that imports parser dependencies.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Output run root. Default: artifacts/scratch/stack_contract_canary_<UTC timestamp>.",
    )
    parser.add_argument("--profile", default="stack_contract", help="Parser bagging profile.")
    parser.add_argument("--query", default="zoning ordinance apartment height", help="Analysis retrieval query.")
    parser.add_argument("--top-k", type=int, default=3, help="Evidence items to keep per query.")
    parser.add_argument("--chunk-words", type=int, default=80)
    parser.add_argument("--overlap-words", type=int, default=0)
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="Treat warnings as errors in parser/analysis validation.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_fixture_page(path: Path, *, width: int = 320, height: int = 480) -> None:
    """Write a small valid PGM page image without requiring image libraries."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = ["P2", f"{width} {height}", "255"]
    for y in range(height):
        values: list[str] = []
        for x in range(width):
            in_text_line = 35 <= x <= 285 and any(start <= y < start + 8 for start in range(45, 420, 32))
            in_rule = 28 <= x <= 292 and y in (30, 432)
            values.append("20" if in_text_line or in_rule else "255")
        rows.append(" ".join(values))
    path.write_text("\n".join(rows) + "\n", encoding="ascii")


def write_source_artifact_manifest(manifest: Path, image_path: Path) -> dict[str, Any]:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "page_id": "stack-canary-issue__p0001__fixture-page",
        "image_path": str(image_path.resolve()),
        "issue_id": "stack-canary-issue",
        "page_number": 1,
        "checksum_sha256": sha256_file(image_path),
        "source": {
            "source_system": "stack_contract_canary",
            "source_id": "fixture-page",
            "source_url": "",
            "metadata": {
                "contract_version": "source-artifact-v1",
                "artifact_kind": "page_image",
                "image_exists": True,
            },
        },
        "metadata": {
            "contract_version": "source-artifact-v1",
            "artifact_kind": "page_image",
            "fixture": True,
        },
    }
    manifest.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    return row


def write_command_adapter(adapter_script: Path, config_path: Path, *, profile: str) -> None:
    adapter_script.parent.mkdir(parents=True, exist_ok=True)
    adapter_script.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import json",
                "import sys",
                "from pathlib import Path",
                "page_id = sys.argv[1]",
                "model_id = sys.argv[2]",
                "output_path = Path(sys.argv[3])",
                "width = int(float(sys.argv[4]))",
                "height = int(float(sys.argv[5]))",
                (
                    "text = 'The stack contract canary contains a zoning ordinance "
                    "about apartment height near the railroad station.'"
                ),
                "payload = {",
                "  'page_id': page_id,",
                "  'model_id': model_id,",
                "  'regions': [",
                "    {",
                "      'bbox_xyxy': [max(5, width * 0.08), max(5, height * 0.10), width * 0.92, height * 0.42],",
                "      'label': 'text',",
                "      'confidence': 0.98,",
                "      'text': text,",
                "      'reading_order': 1,",
                "      'metadata': {'fixture': 'stack_contract_canary'},",
                "    }",
                "  ],",
                "  'metadata': {'contract_fixture': True},",
                "}",
                "output_path.parent.mkdir(parents=True, exist_ok=True)",
                "output_path.write_text(json.dumps(payload) + '\\n', encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "include_builtin_adapters": False,
        "command_adapters": [
            {
                "model_id": "stack_contract_text_v1",
                "family": "fixture_ocr",
                "resource_class": "cpu",
                "profiles": [profile],
                "command": [
                    "{python}",
                    str(adapter_script.resolve()),
                    "{page_id}",
                    "{model_id}",
                    "{output_path}",
                    "{width}",
                    "{height}",
                ],
                "timeout_seconds": 30,
            }
        ],
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command_env(parser_repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = [str(SRC_DIR), str(parser_repo / "src")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env.pop("OPENAI_API_KEY", None)
    env.pop("OPENAI_KEY", None)
    env.pop("CODEX_API_KEY", None)
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("GOOGLE_API_KEY", None)
    return env


def select_parser_python(explicit: str | None, parser_repo: Path) -> str:
    candidates = [
        explicit,
        str(parser_repo / ".venv" / "bin" / "python"),
        sys.executable,
        shutil.which("python"),
        shutil.which("python3"),
    ]
    checked: list[str] = []
    for candidate in candidates:
        if not candidate or candidate in checked:
            continue
        checked.append(candidate)
        try:
            completed = subprocess.run(
                [candidate, "-c", "import PIL, numpy"],
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError:
            if explicit:
                raise SystemExit(f"ERROR: --parser-python is not executable: {candidate}")
            continue
        if completed.returncode == 0:
            return candidate
        if explicit:
            raise SystemExit(
                f"ERROR: --parser-python cannot import Pillow and NumPy: {candidate}\n{completed.stderr}"
            )
    raise SystemExit(
        "ERROR: no parser-capable Python found; pass --parser-python with Pillow and NumPy installed. "
        f"Checked: {checked}"
    )


def run_json_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    label: str,
) -> dict[str, Any]:
    result = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise SystemExit(
            json.dumps(
                {
                    "error": f"{label}_failed",
                    "returncode": result.returncode,
                    "cmd": cmd,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
                indent=2,
            )
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            json.dumps(
                {
                    "error": f"{label}_invalid_json_stdout",
                    "message": str(exc),
                    "cmd": cmd,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
                indent=2,
            )
        ) from exc
    if not isinstance(payload, dict):
        raise SystemExit(json.dumps({"error": f"{label}_non_object_stdout", "cmd": cmd}, indent=2))
    return payload


def default_run_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "artifacts" / "scratch" / f"stack_contract_canary_{stamp}"


def main() -> int:
    args = parse_args()
    parser_repo = args.parser_repo.expanduser().resolve()
    if not (parser_repo / "src" / "newsbag").is_dir():
        raise SystemExit(f"ERROR: --parser-repo does not look like newspaper-parsing: {parser_repo}")
    parser_python = select_parser_python(args.parser_python, parser_repo)

    run_root = (args.run_root or default_run_root()).expanduser().resolve()
    inputs_dir = run_root / "inputs"
    parser_run_dir = run_root / "parser_run"
    analysis_dir = run_root / "analysis"
    manifest = inputs_dir / "source_artifacts.jsonl"
    image_path = inputs_dir / "stack-canary-page.pgm"
    adapter_script = inputs_dir / "stack_contract_adapter.py"
    bagging_config = inputs_dir / "bagging_config.json"
    queries_jsonl = inputs_dir / "queries.jsonl"
    parser_manifest_validation = parser_run_dir / "reports" / "source_artifacts.validation.json"
    parser_bundle_validation = parser_run_dir / "reports" / "validation.json"
    evidence_jsonl = analysis_dir / "outputs" / "evidence_contexts.jsonl"
    analysis_validation = analysis_dir / "reports" / "validation.json"
    stack_analysis_validation = analysis_dir / "reports" / "stack_validation.json"

    write_fixture_page(image_path)
    manifest_row = write_source_artifact_manifest(manifest, image_path)
    write_command_adapter(adapter_script, bagging_config, profile=str(args.profile))
    queries_jsonl.write_text(
        json.dumps(
            {
                "query_id": "stack-contract-query",
                "query": str(args.query),
                "task": "stack_contract_retrieval",
                "relevant_page_ids": [manifest_row["page_id"]],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    env = command_env(parser_repo)

    parser_manifest_report = run_json_command(
        [
            parser_python,
            "-m",
            "newsbag",
            "validate-parse-input-manifest",
            "--manifest",
            str(manifest),
            "--require-files",
            "--require-checksums",
            "--verify-checksums",
            "--output-json",
            str(parser_manifest_validation),
            "--json",
        ]
        + (["--strict"] if args.strict_validation else []),
        cwd=parser_repo,
        env=env,
        label="parser_manifest_validation",
    )

    parser_summary = run_json_command(
        [
            parser_python,
            "-m",
            "newsbag",
            "bagging-canary",
            "--manifest",
            str(manifest),
            "--run-dir",
            str(parser_run_dir),
            "--profile",
            str(args.profile),
            "--config",
            str(bagging_config),
        ],
        cwd=parser_repo,
        env=env,
        label="parser_bagging_canary",
    )

    parser_bundle_report = run_json_command(
        [
            parser_python,
            "-m",
            "newsbag",
            "validate-run",
            "--run-dir",
            str(parser_run_dir),
            "--output-json",
            str(parser_bundle_validation),
            "--json",
        ]
        + (["--strict"] if args.strict_validation else []),
        cwd=parser_repo,
        env=env,
        label="parser_run_validation",
    )

    analysis_summary = run_json_command(
        [
            sys.executable,
            "-m",
            "newsvlm_analysis",
            "run",
            "--parser-run-dir",
            str(parser_run_dir),
            "--run-dir",
            str(analysis_dir),
            "--queries-jsonl",
            str(queries_jsonl),
            "--top-k",
            str(args.top_k),
            "--chunk-words",
            str(args.chunk_words),
            "--overlap-words",
            str(args.overlap_words),
        ],
        cwd=REPO_ROOT,
        env=env,
        label="offline_analysis_run",
    )

    analysis_report = run_json_command(
        [
            sys.executable,
            "-m",
            "newsvlm_analysis",
            "validate-run",
            "--run-dir",
            str(analysis_dir),
            "--output-json",
            str(stack_analysis_validation),
        ]
        + (["--warnings-as-errors"] if args.strict_validation else []),
        cwd=REPO_ROOT,
        env=env,
        label="offline_analysis_validation",
    )
    evaluation = json.loads(
        (analysis_dir / "reports" / "retrieval_evaluation.json").read_text(encoding="utf-8")
    )
    status = (
        "ok"
        if parser_manifest_report.get("status") == "ok"
        and parser_bundle_report.get("status") == "ok"
        and analysis_summary.get("status") == "ok"
        and analysis_report.get("status") == "ok"
        and int((analysis_summary.get("counts") or {}).get("evidence_items") or 0) > 0
        and (evaluation.get("metrics") or {}).get("hit_rate_at_k") == 1.0
        else "error"
    )
    stack_summary = {
        "contract": "newspaper-stack-contract-canary-v2",
        "status": status,
        "run_root": str(run_root),
        "uses_external_llm_api": False,
        "inputs": {
            "manifest": str(manifest),
            "page_image": str(image_path),
            "bagging_config": str(bagging_config),
            "command_adapter": str(adapter_script),
            "queries_jsonl": str(queries_jsonl),
            "manifest_page_id": manifest_row["page_id"],
        },
        "parser": {
            "repo": str(parser_repo),
            "python": parser_python,
            "profile": str(args.profile),
            "run_dir": str(parser_run_dir),
            "manifest_validation": parser_manifest_report,
            "summary": parser_summary,
            "run_validation": parser_bundle_report,
        },
        "analysis": {
            "query": str(args.query),
            "run_dir": str(analysis_dir),
            "evidence_contexts_jsonl": str(evidence_jsonl),
            "validation_json": str(analysis_validation),
            "summary": analysis_summary,
            "validation": analysis_report,
            "evaluation": evaluation,
        },
    }
    summary_path = run_root / "stack_summary.json"
    summary_path.write_text(json.dumps(stack_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stack_summary, indent=2, sort_keys=True))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
