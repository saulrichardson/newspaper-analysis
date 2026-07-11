#!/bin/bash
# Short CPU canary for the persistent, API-free offline analysis run bundle.

#SBATCH -A torch_pr_609_general
#SBATCH -p cpu_short
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --time=00:10:00
#SBATCH -J newspaper_offline_analysis
#SBATCH -o /scratch/%u/codex_hpc/newspaper_analysis/logs/%x-%j.out
#SBATCH -e /scratch/%u/codex_hpc/newspaper_analysis/logs/%x-%j.err

set -euo pipefail

BASE="${BASE:-/scratch/$USER/codex_hpc/newspaper_analysis}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE/newspaper-analysis}"
RUN_DIR="${RUN_DIR:-$BASE/runs/offline_analysis_${SLURM_JOB_ID:-manual}}"
VENV_DIR="${VENV_DIR:-$BASE/envs/offline_analysis_py312}"
BOOTSTRAP_PY="${BOOTSTRAP_PY:-}"
ANALYSIS_PY="${ANALYSIS_PY:-$VENV_DIR/bin/python}"
JOB_ID_VALUE="${SLURM_JOB_ID:-manual}"

unset OPENAI_API_KEY OPENAI_KEY CODEX_API_KEY ANTHROPIC_API_KEY GOOGLE_API_KEY

mkdir -p "$BASE/logs" "$BASE/runs" "$RUN_DIR"

echo "[offline-analysis] host=$(hostname)"
echo "[offline-analysis] date=$(date -Is)"
echo "[offline-analysis] project_root=$PROJECT_ROOT"
echo "[offline-analysis] run_dir=$RUN_DIR"

if [ ! -d "$PROJECT_ROOT/src/newsvlm_analysis" ]; then
  echo "[offline-analysis] ERROR: invalid PROJECT_ROOT: $PROJECT_ROOT" >&2
  exit 2
fi

select_bootstrap_python() {
  if [ -n "$BOOTSTRAP_PY" ]; then
    printf "%s\n" "$BOOTSTRAP_PY"
    return 0
  fi
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
      then
        command -v "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

if [ ! -x "$ANALYSIS_PY" ]; then
  BOOTSTRAP_PY="$(select_bootstrap_python)" || {
    echo "[offline-analysis] ERROR: Python >=3.11 was not found" >&2
    exit 2
  }
  "$BOOTSTRAP_PY" -m venv "$VENV_DIR"
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$ANALYSIS_PY" -c "import newsvlm_analysis" >/dev/null

if ! "$ANALYSIS_PY" - <<'PY' >/dev/null 2>&1
import sqlite3
connection = sqlite3.connect(":memory:")
connection.execute("CREATE VIRTUAL TABLE probe USING fts5(text)")
PY
then
  echo "[offline-analysis] ERROR: Python SQLite does not provide FTS5" >&2
  exit 2
fi

PARSER_RUN_DIR="$RUN_DIR/fixture_parser_run"
ANALYSIS_RUN_DIR="$RUN_DIR/analysis"
QUERIES_JSONL="$RUN_DIR/queries.jsonl"
mkdir -p \
  "$PARSER_RUN_DIR/outputs/fused_pages" \
  "$PARSER_RUN_DIR/outputs/transcripts" \
  "$PARSER_RUN_DIR/manifests" \
  "$PARSER_RUN_DIR/reports"

cat > "$PARSER_RUN_DIR/outputs/fused_pages/page-zoning.json" <<'JSON'
{"page_id":"page-zoning","model_ids":["layout-fixture","ocr-fixture"],"regions":[{"region_id":"zoning-r1","bbox_xyxy":[10,20,600,180],"label":"text","confidence":0.96,"source_model":"ocr-fixture","text":"The city adopted a zoning ordinance limiting apartment height to six stories.","reading_order":1}],"transcript":"The city adopted a zoning ordinance limiting apartment height to six stories.","disagreement_score":0.02,"quality":{"region_count":1},"provenance":{"contract_source":"torch_fixture"}}
JSON
cat > "$PARSER_RUN_DIR/outputs/fused_pages/page-parking.json" <<'JSON'
{"page_id":"page-parking","model_ids":["layout-fixture","ocr-fixture"],"regions":[{"region_id":"parking-r1","bbox_xyxy":[10,20,600,180],"label":"text","confidence":0.95,"source_model":"ocr-fixture","text":"The ordinance required one parking space for each apartment dwelling unit.","reading_order":1}],"transcript":"The ordinance required one parking space for each apartment dwelling unit.","disagreement_score":0.03,"quality":{"region_count":1},"provenance":{"contract_source":"torch_fixture"}}
JSON
cat > "$PARSER_RUN_DIR/outputs/fused_pages/page-school.json" <<'JSON'
{"page_id":"page-school","model_ids":["layout-fixture","ocr-fixture"],"regions":[{"region_id":"school-r1","bbox_xyxy":[10,20,600,180],"label":"text","confidence":0.94,"source_model":"ocr-fixture","text":"The school board approved classroom repairs and the annual library budget.","reading_order":1}],"transcript":"The school board approved classroom repairs and the annual library budget.","disagreement_score":0.01,"quality":{"region_count":1},"provenance":{"contract_source":"torch_fixture"}}
JSON

printf '%s\n' \
  'The city adopted a zoning ordinance limiting apartment height to six stories.' \
  > "$PARSER_RUN_DIR/outputs/transcripts/page-zoning.txt"
printf '%s\n' \
  'The ordinance required one parking space for each apartment dwelling unit.' \
  > "$PARSER_RUN_DIR/outputs/transcripts/page-parking.txt"
printf '%s\n' \
  'The school board approved classroom repairs and the annual library budget.' \
  > "$PARSER_RUN_DIR/outputs/transcripts/page-school.txt"

cat > "$PARSER_RUN_DIR/manifests/parse_input.jsonl" <<'JSONL'
{"page_id":"page-zoning","image_path":"/fixture/page-zoning.png","issue_id":"issue-1901-01-02","page_number":1,"checksum_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","source":{"source_system":"torch_fixture","source_id":"source-page-zoning"}}
{"page_id":"page-parking","image_path":"/fixture/page-parking.png","issue_id":"issue-1901-01-02","page_number":2,"checksum_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","source":{"source_system":"torch_fixture","source_id":"source-page-parking"}}
{"page_id":"page-school","image_path":"/fixture/page-school.png","issue_id":"issue-1901-01-02","page_number":3,"checksum_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","source":{"source_system":"torch_fixture","source_id":"source-page-school"}}
JSONL
cat > "$PARSER_RUN_DIR/summary.json" <<'JSON'
{"run_id":"torch-parser-fixture","profile":"full","page_count":3,"model_ids":["layout-fixture","ocr-fixture"],"performance":{"pages_completed":3,"errors":0}}
JSON
cat > "$PARSER_RUN_DIR/provenance.json" <<'JSON'
{"contract_version":"parser-bagging-v1","repo_commit":"torch-fixture"}
JSON
cat > "$PARSER_RUN_DIR/reports/validation.json" <<'JSON'
{"contract":"parser-run-bundle-validation-v1","status":"ok","counts":{"errors":0,"warnings":0},"issues":[]}
JSON
cat > "$PARSER_RUN_DIR/reports/input_manifest_validation.json" <<'JSON'
{"contract":"parse-input-manifest-validation-v1","status":"ok","manifest_path":"manifests/parse_input.jsonl","counts":{"rows":3,"errors":0,"warnings":0},"issues":[]}
JSON
cat > "$QUERIES_JSONL" <<'JSONL'
{"query_id":"height-policy","query":"zoning apartment height","task":"policy_retrieval","relevant_page_ids":["page-zoning"]}
{"query_id":"parking-policy","query":"parking requirement dwelling unit","task":"policy_retrieval","relevant_page_ids":["page-parking"]}
JSONL

cd "$PROJECT_ROOT"
set +e
"$ANALYSIS_PY" -m newsvlm_analysis run \
  --parser-run-dir "$PARSER_RUN_DIR" \
  --run-dir "$ANALYSIS_RUN_DIR" \
  --queries-jsonl "$QUERIES_JSONL" \
  --top-k 2 \
  --chunk-words 40 \
  --overlap-words 0 \
  > "$RUN_DIR/run_stdout.json" \
  2> "$RUN_DIR/run_stderr.log"
RUN_STATUS=$?
set -e

"$ANALYSIS_PY" - <<PY
from __future__ import annotations

import json
from pathlib import Path

run_root = Path("$RUN_DIR")
analysis_run = Path("$ANALYSIS_RUN_DIR")

def load(path: Path) -> dict:
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

summary = load(analysis_run / "summary.json")
validation = load(analysis_run / "reports" / "validation.json")
evaluation = load(analysis_run / "reports" / "retrieval_evaluation.json")
performance = load(analysis_run / "reports" / "performance.json")
metrics = evaluation.get("metrics", {})
ok = all(
    [
        int("$RUN_STATUS") == 0,
        summary.get("status") == "ok",
        validation.get("status") == "ok",
        metrics.get("hit_rate_at_k") == 1.0,
        metrics.get("mrr") == 1.0,
        summary.get("uses_external_llm_api") is False,
        validation.get("counts", {}).get("index_chunks", 0) >= 3,
    ]
)
status = {
    "status": "ok" if ok else "error",
    "job_id": "$JOB_ID_VALUE",
    "run_dir": str(run_root),
    "analysis_run_dir": str(analysis_run),
    "summary": summary,
    "validation": validation,
    "evaluation": evaluation,
    "performance": performance,
    "run_stdout_json": str(run_root / "run_stdout.json"),
    "run_stderr_log": str(run_root / "run_stderr.log"),
    "uses_external_llm_api": False,
}
(run_root / "slurm_status.json").write_text(
    json.dumps(status, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(status, indent=2, sort_keys=True))
raise SystemExit(0 if ok else 1)
PY
