#!/bin/bash
# Sync the public analysis package to Torch and run its scheduler-backed canary.

set -euo pipefail

REMOTE="${REMOTE:-torch}"
ACCOUNT="${ACCOUNT:-torch_pr_609_general}"
PARTITION="${PARTITION:-cpu_short}"
REMOTE_BASE="${REMOTE_BASE:-}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
POLL_SECONDS="${POLL_SECONDS:-10}"
WAIT=1
PLAN_ONLY=0
SKIP_SYNC=0

usage() {
  sed -n '1,80p' "$0"
  cat <<'TXT'

Flags:
  --remote HOST          SSH host, default: torch
  --remote-base PATH     Scratch root, default: /scratch/$REMOTE_USER/codex_hpc/newspaper_analysis
  --account ACCOUNT      Slurm account, default: torch_pr_609_general
  --partition PARTITION  Slurm partition, default: cpu_short
  --timeout SECONDS      Poll timeout, default: 900
  --poll SECONDS         Poll interval, default: 10
  --no-wait              Submit without polling
  --skip-sync            Reuse the existing remote repository copy
  --plan-only            Print the plan without syncing or submitting
TXT
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote) REMOTE="${2:-}"; shift 2 ;;
    --remote-base) REMOTE_BASE="${2:-}"; shift 2 ;;
    --account) ACCOUNT="${2:-}"; shift 2 ;;
    --partition) PARTITION="${2:-}"; shift 2 ;;
    --timeout) TIMEOUT_SECONDS="${2:-900}"; shift 2 ;;
    --poll) POLL_SECONDS="${2:-10}"; shift 2 ;;
    --no-wait) WAIT=0; shift ;;
    --skip-sync) SKIP_SYNC=1; shift ;;
    --plan-only|--dry-run) PLAN_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

REMOTE_USER="$(ssh "$REMOTE" 'printf %s "$USER"')"
if [[ -z "$REMOTE_USER" ]]; then
  echo "ERROR: could not determine the user for SSH host $REMOTE" >&2
  exit 2
fi
if [[ -z "$REMOTE_BASE" ]]; then
  REMOTE_BASE="/scratch/$REMOTE_USER/codex_hpc/newspaper_analysis"
fi

PROJECT_ROOT="$REMOTE_BASE/newspaper-analysis"
RUN_DIR="$REMOTE_BASE/runs/offline_analysis_$(date -u +%Y%m%d_%H%M%S)"
SCRIPT="slurm/pipelines/offline_analysis_cpu_short.sbatch"

echo "[plan] remote=$REMOTE"
echo "[plan] remote_base=$REMOTE_BASE"
echo "[plan] project_root=$PROJECT_ROOT"
echo "[plan] run_dir=$RUN_DIR"
echo "[plan] account=$ACCOUNT"
echo "[plan] partition=$PARTITION"

if [[ "$PLAN_ONLY" -eq 1 ]]; then
  exit 0
fi

ssh "$REMOTE" "mkdir -p '$REMOTE_BASE/logs' '$REMOTE_BASE/runs' '$PROJECT_ROOT'"
if [[ "$SKIP_SYNC" -eq 0 ]]; then
  rsync -az --delete --delete-excluded \
    --exclude '.git/' \
    --exclude '.pytest_cache/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.env' \
    --exclude '.env.*' \
    --exclude '.venv/' \
    --exclude 'venv/' \
    --exclude 'artifacts/' \
    --exclude 'archive/' \
    --exclude 'reports/' \
    --exclude 'newspaper-analysis-local/' \
    ./ "$REMOTE:$PROJECT_ROOT/"
fi

ssh "$REMOTE" "cd '$PROJECT_ROOT' && sbatch --test-only -A '$ACCOUNT' -p '$PARTITION' --cpus-per-task=2 --mem=3G --time=00:10:00 --wrap hostname >/dev/null"
JOB_ID="$(
  ssh "$REMOTE" "cd '$PROJECT_ROOT' && sbatch --parsable -A '$ACCOUNT' -p '$PARTITION' \
    --export=ALL,BASE='$REMOTE_BASE',PROJECT_ROOT='$PROJECT_ROOT',RUN_DIR='$RUN_DIR' \
    '$SCRIPT'"
)"

LOG_OUT="$REMOTE_BASE/logs/newspaper_offline_analysis-$JOB_ID.out"
LOG_ERR="$REMOTE_BASE/logs/newspaper_offline_analysis-$JOB_ID.err"
echo "[submit] job_id=$JOB_ID"
echo "[submit] run_dir=$RUN_DIR"
echo "[submit] stdout=$LOG_OUT"
echo "[submit] stderr=$LOG_ERR"

if [[ "$WAIT" -eq 0 ]]; then
  exit 0
fi

deadline=$((SECONDS + TIMEOUT_SECONDS))
while [[ "$SECONDS" -lt "$deadline" ]]; do
  queued="$(ssh "$REMOTE" "squeue -h -j '$JOB_ID' -o %T 2>/dev/null || true")"
  if [[ -z "$queued" ]]; then
    break
  fi
  echo "[poll] job_id=$JOB_ID state=$queued"
  sleep "$POLL_SECONDS"
done

if [[ "$SECONDS" -ge "$deadline" ]]; then
  echo "ERROR: timed out waiting for job $JOB_ID after $TIMEOUT_SECONDS seconds" >&2
  exit 3
fi

if ! ssh "$REMOTE" "test -f '$RUN_DIR/slurm_status.json'"; then
  echo "ERROR: Slurm job did not produce $RUN_DIR/slurm_status.json" >&2
  ssh "$REMOTE" "sacct -j '$JOB_ID' --format=JobID,State,ExitCode,Elapsed -n -P 2>/dev/null || true" >&2
  ssh "$REMOTE" "tail -80 '$LOG_OUT' '$LOG_ERR' 2>/dev/null || true" >&2
  exit 4
fi

ssh "$REMOTE" "cat '$RUN_DIR/slurm_status.json'"
ssh "$REMOTE" "python3 - '$RUN_DIR/slurm_status.json'" <<'PY'
import json
import sys
from pathlib import Path

status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if status.get("status") == "ok" else 1)
PY
