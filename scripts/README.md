# Scripts

The active script surface is grouped by responsibility:

- `scripts/frontier/`: frontier corpus, semantics, clustering, and writeup entrypoints
- `scripts/pipelines/`: issue-classifier, transcription, and recovery workflows
- `scripts/platform/`: gateway runners, provider batch helpers, and execution utilities

Legacy flat scripts live under `archive/legacy/scripts/`.

Useful no-API canaries:

```bash
python -m newsvlm_analysis run \
  --parser-run-dir /path/to/parser/run \
  --run-dir artifacts/runs/offline-analysis-canary \
  --query "zoning ordinance apartment height"

bash scripts/pipelines/submit_torch_offline_analysis.sh

python scripts/pipelines/build_local_retrieval_context.py \
  --parser-run-dir /path/to/parser/run \
  --query "zoning ordinance apartment height" \
  --output-jsonl artifacts/scratch/evidence_contexts.jsonl \
  --output-format contexts \
  --validation-json artifacts/scratch/evidence_contexts.validation.json

python scripts/pipelines/run_stack_contract_canary.py \
  --parser-repo ../newspaper-parsing \
  --run-root artifacts/scratch/stack_contract_canary

bash scripts/pipelines/submit_torch_stack_contract_canary.sh \
  --parser-repo ../newspaper-parsing
```

`python -m newsvlm_analysis run` is the canonical production workflow. The
standalone retrieval-context and stack scripts are narrower contract canaries.
