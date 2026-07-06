#!/bin/bash
# CPU Slurm smoke for API-free evidence-context construction.

#SBATCH -A torch_pr_609_general
#SBATCH -p cs
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH -J newspaper_analysis_evidence
#SBATCH -o /scratch/%u/codex_hpc/newspaper_analysis/logs/%x-%j.out
#SBATCH -e /scratch/%u/codex_hpc/newspaper_analysis/logs/%x-%j.err

set -euo pipefail

BASE="${BASE:-/scratch/$USER/codex_hpc/newspaper_analysis}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE/newspaper-analysis}"
RUN_DIR="${RUN_DIR:-$BASE/runs/evidence_context_${SLURM_JOB_ID:-manual}}"
PARSER_RUN_DIR="${PARSER_RUN_DIR:-$BASE/fixtures/parser_run}"
QUERY="${QUERY:-zoning ordinance apartment height}"
VENV_DIR="${VENV_DIR:-$BASE/envs/analysis_py312}"
BOOTSTRAP_PY="${BOOTSTRAP_PY:-}"
ANALYSIS_PY="${ANALYSIS_PY:-$VENV_DIR/bin/python}"

unset OPENAI_API_KEY OPENAI_KEY CODEX_API_KEY

mkdir -p "$BASE/logs" "$BASE/fixtures" "$BASE/runs" "$RUN_DIR"

echo "[analysis-evidence] host=$(hostname)"
echo "[analysis-evidence] date=$(date -Is)"
echo "[analysis-evidence] project_root=$PROJECT_ROOT"
echo "[analysis-evidence] run_dir=$RUN_DIR"
echo "[analysis-evidence] parser_run_dir=$PARSER_RUN_DIR"

if [ ! -d "$PROJECT_ROOT/src/newsvlm_analysis" ]; then
  echo "[analysis-evidence] ERROR: PROJECT_ROOT does not look like newspaper-analysis: $PROJECT_ROOT" >&2
  exit 2
fi

select_bootstrap_python() {
  if [ -n "$BOOTSTRAP_PY" ]; then
    printf "%s\n" "$BOOTSTRAP_PY"
    return 0
  fi
  for candidate in python3.12 python3.11 python3 /scratch/$USER/paddleocr_vl15/envs/mineru25_py310/bin/python; do
    if command -v "$candidate" >/dev/null 2>&1 || [ -x "$candidate" ]; then
      if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
      then
        command -v "$candidate" 2>/dev/null || printf "%s\n" "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

if [ ! -x "$ANALYSIS_PY" ]; then
  BOOTSTRAP_PY="$(select_bootstrap_python)" || {
    echo "[analysis-evidence] ERROR: could not find Python >=3.11 on this node" >&2
    exit 2
  }
  echo "[analysis-evidence] bootstrap_python=$BOOTSTRAP_PY"
  "$BOOTSTRAP_PY" -m venv "$VENV_DIR"
fi

if ! "$ANALYSIS_PY" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
then
  echo "[analysis-evidence] ERROR: ANALYSIS_PY must be Python >=3.11: $ANALYSIS_PY" >&2
  "$ANALYSIS_PY" --version >&2 || true
  exit 2
fi

"$ANALYSIS_PY" -m pip install --upgrade pip >/dev/null
"$ANALYSIS_PY" -m pip install -e "$PROJECT_ROOT" >/dev/null

if ! find "$PARSER_RUN_DIR/outputs/fused_pages" -maxdepth 1 -name '*.json' -type f 2>/dev/null | grep -q .; then
  echo "[analysis-evidence] creating fixture parser run bundle"
  mkdir -p "$PARSER_RUN_DIR/outputs/fused_pages"
  cat > "$PARSER_RUN_DIR/outputs/fused_pages/torch-page-001.json" <<'JSON'
{
  "page_id": "torch-page-001",
  "model_ids": ["baseline_geometry_v1", "column_detector_v1"],
  "transcript": "The city council debated a zoning ordinance that limited apartment height near the railroad station.",
  "quality": {"region_count": 2},
  "provenance": {"contract_source": "torch_fixture"}
}
JSON
  cat > "$PARSER_RUN_DIR/summary.json" <<'JSON'
{
  "run_id": "torch-parser-fixture",
  "profile": "full",
  "page_count": 1,
  "model_ids": ["baseline_geometry_v1", "column_detector_v1"],
  "performance": {"pages_completed": 1, "errors": 0}
}
JSON
  cat > "$PARSER_RUN_DIR/provenance.json" <<'JSON'
{
  "contract_version": "parser-bagging-v1",
  "repo_commit": "torch-fixture"
}
JSON
fi

cd "$PROJECT_ROOT"
OUTPUT_JSONL="$RUN_DIR/evidence_contexts.jsonl"
"$ANALYSIS_PY" scripts/pipelines/build_local_retrieval_context.py \
  --parser-run-dir "$PARSER_RUN_DIR" \
  --query "$QUERY" \
  --output-jsonl "$OUTPUT_JSONL" \
  --output-format contexts \
  --top-k 3 \
  --chunk-words 80 \
  --overlap-words 0 \
  > "$RUN_DIR/summary.json"

"$ANALYSIS_PY" - <<PY
from __future__ import annotations

import json
from pathlib import Path

run_dir = Path("$RUN_DIR")
summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
rows = [
    json.loads(line)
    for line in (run_dir / "evidence_contexts.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
status = {
    "status": "ok" if rows and rows[0].get("evidence_count", 0) > 0 else "error",
    "run_dir": str(run_dir),
    "summary": summary,
    "evidence_contexts_jsonl": str(run_dir / "evidence_contexts.jsonl"),
    "parser_run_dir": "$PARSER_RUN_DIR",
    "uses_external_llm_api": False,
}
(run_dir / "slurm_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps(status, indent=2, sort_keys=True))
raise SystemExit(0 if status["status"] == "ok" else 1)
PY
