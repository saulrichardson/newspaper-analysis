#!/bin/bash
# Sync this repo to Torch, run the API-free evidence-context Slurm smoke, and
# print the structured status.

set -euo pipefail

REMOTE="${REMOTE:-torch}"
ACCOUNT="${ACCOUNT:-torch_pr_609_general}"
PARTITION="${PARTITION:-cs}"
REMOTE_BASE="${REMOTE_BASE:-}"
QUERY="${QUERY:-zoning ordinance apartment height}"
WAIT=1
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
POLL_SECONDS="${POLL_SECONDS:-10}"
PLAN_ONLY=0
SKIP_SYNC=0

usage() {
  sed -n '1,60p' "$0"
  cat <<'TXT'

Flags:
  --remote HOST          SSH host, default: torch
  --remote-base PATH     Scratch root, default: /scratch/$REMOTE_USER/codex_hpc/newspaper_analysis
  --account ACCOUNT      Slurm account, default: torch_pr_609_general
  --partition PARTITION  Slurm partition, default: cs
  --query TEXT           Retrieval query for the smoke
  --timeout SECONDS      Poll timeout, default: 900
  --poll SECONDS         Poll interval, default: 10
  --no-wait              Submit and print job/run paths without polling
  --skip-sync            Reuse the existing remote repo copy
  --plan-only            Print planned remote commands without executing
TXT
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote)
      REMOTE="${2:-}"
      shift 2
      ;;
    --remote-base)
      REMOTE_BASE="${2:-}"
      shift 2
      ;;
    --account)
      ACCOUNT="${2:-}"
      shift 2
      ;;
    --partition)
      PARTITION="${2:-}"
      shift 2
      ;;
    --query)
      QUERY="${2:-}"
      shift 2
      ;;
    --timeout)
      TIMEOUT_SECONDS="${2:-900}"
      shift 2
      ;;
    --poll)
      POLL_SECONDS="${2:-10}"
      shift 2
      ;;
    --no-wait)
      WAIT=0
      shift 1
      ;;
    --skip-sync)
      SKIP_SYNC=1
      shift 1
      ;;
    --plan-only|--dry-run)
      PLAN_ONLY=1
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

REMOTE_USER="$(ssh "$REMOTE" 'printf %s "$USER"')"
if [[ -z "$REMOTE_USER" ]]; then
  echo "ERROR: could not determine remote user for $REMOTE" >&2
  exit 2
fi

if [[ -z "$REMOTE_BASE" ]]; then
  REMOTE_BASE="/scratch/$REMOTE_USER/codex_hpc/newspaper_analysis"
fi

PROJECT_ROOT="$REMOTE_BASE/newspaper-analysis"
RUN_DIR="$REMOTE_BASE/runs/evidence_context_$(date -u +%Y%m%d_%H%M%S)"
PARSER_RUN_DIR="$REMOTE_BASE/fixtures/parser_run"
SCRIPT="slurm/pipelines/local_evidence_context_cs.sbatch"

echo "[plan] remote=$REMOTE"
echo "[plan] remote_user=$REMOTE_USER"
echo "[plan] remote_base=$REMOTE_BASE"
echo "[plan] project_root=$PROJECT_ROOT"
echo "[plan] run_dir=$RUN_DIR"
echo "[plan] parser_run_dir=$PARSER_RUN_DIR"
echo "[plan] query=$QUERY"

if [[ "$PLAN_ONLY" -eq 1 ]]; then
  echo "[plan] would sync repo and submit $SCRIPT"
  exit 0
fi

ssh "$REMOTE" "mkdir -p '$REMOTE_BASE/logs' '$REMOTE_BASE/runs' '$REMOTE_BASE/fixtures' '$PROJECT_ROOT'"

if [[ "$SKIP_SYNC" -eq 0 ]]; then
  rsync -az --delete --delete-excluded \
    --exclude '.git/' \
    --exclude '.pytest_cache/' \
    --exclude '.playwright-cli/' \
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

ssh "$REMOTE" "cd '$PROJECT_ROOT' && sbatch --test-only -A '$ACCOUNT' -p '$PARTITION' --cpus-per-task=2 --mem=2G --time=00:10:00 --wrap hostname >/dev/null"

JOB_ID="$(
  ssh "$REMOTE" "cd '$PROJECT_ROOT' && sbatch --parsable -A '$ACCOUNT' -p '$PARTITION' \
    --export=ALL,BASE='$REMOTE_BASE',PROJECT_ROOT='$PROJECT_ROOT',RUN_DIR='$RUN_DIR',PARSER_RUN_DIR='$PARSER_RUN_DIR',QUERY='$QUERY' \
    '$SCRIPT'"
)"

echo "[submit] job_id=$JOB_ID"
echo "[submit] run_dir=$RUN_DIR"
echo "[submit] logs=$REMOTE_BASE/logs/newspaper_analysis_evidence-$JOB_ID.out"

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

ssh "$REMOTE" "cat '$RUN_DIR/slurm_status.json'"
