#!/bin/bash
# CPU Slurm smoke for the acquisition-style manifest -> parser -> analysis stack contract.

#SBATCH -A torch_pr_609_general
#SBATCH -p cs
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --time=00:12:00
#SBATCH -J newspaper_stack_contract
#SBATCH -o /scratch/%u/codex_hpc/newspaper_stack/logs/%x-%j.out
#SBATCH -e /scratch/%u/codex_hpc/newspaper_stack/logs/%x-%j.err

set -euo pipefail

BASE="${BASE:-/scratch/$USER/codex_hpc/newspaper_stack}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE/newspaper-analysis}"
PARSER_PROJECT_ROOT="${PARSER_PROJECT_ROOT:-$BASE/newspaper-parsing}"
RUN_DIR="${RUN_DIR:-$BASE/runs/stack_contract_${SLURM_JOB_ID:-manual}}"
QUERY="${QUERY:-zoning ordinance apartment height}"
VENV_DIR="${VENV_DIR:-$BASE/envs/stack_py312}"
BOOTSTRAP_PY="${BOOTSTRAP_PY:-}"
ANALYSIS_PY="${ANALYSIS_PY:-$VENV_DIR/bin/python}"

unset OPENAI_API_KEY OPENAI_KEY CODEX_API_KEY

mkdir -p "$BASE/logs" "$BASE/runs" "$RUN_DIR"

echo "[stack-contract] host=$(hostname)"
echo "[stack-contract] date=$(date -Is)"
echo "[stack-contract] project_root=$PROJECT_ROOT"
echo "[stack-contract] parser_project_root=$PARSER_PROJECT_ROOT"
echo "[stack-contract] run_dir=$RUN_DIR"

if [ ! -d "$PROJECT_ROOT/src/newsvlm_analysis" ]; then
  echo "[stack-contract] ERROR: PROJECT_ROOT does not look like newspaper-analysis: $PROJECT_ROOT" >&2
  exit 2
fi
if [ ! -d "$PARSER_PROJECT_ROOT/src/newsbag" ]; then
  echo "[stack-contract] ERROR: PARSER_PROJECT_ROOT does not look like newspaper-parsing: $PARSER_PROJECT_ROOT" >&2
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
    echo "[stack-contract] ERROR: could not find Python >=3.11 on this node" >&2
    exit 2
  }
  echo "[stack-contract] bootstrap_python=$BOOTSTRAP_PY"
  "$BOOTSTRAP_PY" -m venv "$VENV_DIR"
fi

if ! "$ANALYSIS_PY" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
then
  echo "[stack-contract] ERROR: ANALYSIS_PY must be Python >=3.11: $ANALYSIS_PY" >&2
  "$ANALYSIS_PY" --version >&2 || true
  exit 2
fi

"$ANALYSIS_PY" -m pip install --upgrade pip >/dev/null
"$ANALYSIS_PY" -m pip install -e "$PARSER_PROJECT_ROOT" >/dev/null
"$ANALYSIS_PY" -m pip install -e "$PROJECT_ROOT" >/dev/null

cd "$PROJECT_ROOT"
set +e
"$ANALYSIS_PY" scripts/pipelines/run_stack_contract_canary.py \
  --parser-repo "$PARSER_PROJECT_ROOT" \
  --parser-python "$ANALYSIS_PY" \
  --run-root "$RUN_DIR" \
  --query "$QUERY" \
  --top-k 3 \
  > "$RUN_DIR/stack_stdout.json" \
  2> "$RUN_DIR/stack_stderr.log"
CANARY_STATUS=$?
set -e

"$ANALYSIS_PY" - <<PY
from __future__ import annotations

import json
from pathlib import Path

run_dir = Path("$RUN_DIR")
summary_path = run_dir / "stack_summary.json"
stdout_path = run_dir / "stack_stdout.json"
stderr_path = run_dir / "stack_stderr.log"
summary = {}
if summary_path.is_file():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
elif stdout_path.is_file() and stdout_path.read_text(encoding="utf-8").strip():
    summary = json.loads(stdout_path.read_text(encoding="utf-8"))
status = {
    "status": "ok" if int("$CANARY_STATUS") == 0 and summary.get("status") == "ok" else "error",
    "run_dir": str(run_dir),
    "stack_summary_json": str(summary_path),
    "stack_stdout_json": str(stdout_path),
    "stack_stderr_log": str(stderr_path),
    "summary": summary,
    "uses_external_llm_api": False,
}
(run_dir / "slurm_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps(status, indent=2, sort_keys=True))
raise SystemExit(0 if status["status"] == "ok" else 1)
PY
