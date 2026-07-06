# Scripts

The active script surface is grouped by responsibility:

- `scripts/frontier/`: frontier corpus, semantics, clustering, and writeup entrypoints
- `scripts/pipelines/`: issue-classifier, transcription, and recovery workflows
- `scripts/platform/`: gateway runners, provider batch helpers, and execution utilities

Legacy flat scripts live under `archive/legacy/scripts/`.

Useful no-API canaries:

```bash
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
