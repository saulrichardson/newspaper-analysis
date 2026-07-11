# `newspaper-analysis`

[![CI](https://github.com/saulrichardson/newspaper-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/saulrichardson/newspaper-analysis/actions/workflows/ci.yml)

`newspaper-analysis` turns validated `newspaper-parsing` run bundles into
persistent, evidence-first research corpora. The canonical workflow is fully
offline: it indexes fused transcripts, executes repeatable question manifests,
evaluates ranked evidence when gold labels are available, and emits a validated
run bundle with parser provenance and performance measurements.

No external LLM API or API key is required. Higher-level local inference can be
added after retrieval through an explicit adapter without changing the corpus,
evidence, or evaluation contracts.

## Canonical workflow

Install the dependency-light core package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run one or more ad hoc questions against a parser run:

```bash
newspaper-analysis run \
  --parser-run-dir /path/to/parser-run \
  --run-dir artifacts/runs/zoning-research-001 \
  --query "Which height limits were proposed near rail stations?" \
  --query "What parking requirements applied to apartment buildings?"
```

For reproducible evaluation, use a JSONL query manifest:

```json
{"query_id":"height-policy","query":"zoning apartment height","task":"policy_retrieval","relevant_page_ids":["issue-a__p0003"]}
{"query_id":"parking-policy","query":"parking requirement dwelling unit","task":"policy_retrieval","relevant_page_ids":["issue-a__p0004"]}
```

```bash
newspaper-analysis run \
  --parser-run-dir /path/to/parser-run \
  --run-dir artifacts/runs/zoning-eval-001 \
  --queries-jsonl queries.jsonl \
  --top-k 10
```

Each run contains:

- `index/corpus.sqlite3`: persistent SQLite FTS5 corpus index
- `inputs/queries.jsonl`: normalized question and relevance manifest
- `outputs/evidence_contexts.jsonl`: ranked, source-linked evidence packets
- `reports/retrieval_evaluation.json`: hit rate, MRR, recall, and nDCG
- `reports/performance.json`: index throughput and query latency
- `reports/validation.json`: complete run-contract validation
- `config.json`, `provenance.json`, `summary.json`, and `errors.jsonl`

The index remains independently queryable:

```bash
newspaper-analysis search \
  --index artifacts/runs/zoning-eval-001/index/corpus.sqlite3 \
  --query "minimum lot area multifamily district" \
  --top-k 5

newspaper-analysis validate-run \
  --run-dir artifacts/runs/zoning-eval-001
```

The older one-shot evidence builder remains useful for inspecting a single
contract, but production work should use `newspaper-analysis run`:

```bash
python scripts/pipelines/build_local_retrieval_context.py \
  --parser-run-dir ../newspaper-parsing/<run> \
  --query "zoning ordinance apartment height" \
  --output-jsonl artifacts/scratch/evidence_contexts.jsonl \
  --output-format contexts \
  --validation-json artifacts/scratch/evidence_contexts.validation.json
```

This emits `analysis-evidence-context-v1` JSONL packets without building the
persistent corpus or complete run bundle.

## Torch canary

The production run surface has a scheduler-backed CPU canary on NYU Torch:

```bash
bash scripts/pipelines/submit_torch_offline_analysis.sh
```

It syncs only public source files, creates a three-page parser fixture inside a
fresh scratch run, verifies FTS5 support, builds and searches the index,
evaluates gold page labels, validates the run bundle, and returns
`slurm_status.json`. No provider credentials are loaded.

## Cross-repo canary

Cross-repo contract canary:

```bash
python scripts/pipelines/run_stack_contract_canary.py \
  --parser-repo ../newspaper-parsing \
  --run-root artifacts/scratch/stack_contract_canary
```

The canary creates a tiny source-artifact manifest, runs `newsbag
bagging-canary`, validates the parser bundle, builds analysis evidence
contexts, and writes `stack_summary.json`. It is intended to prove the
acquisition-style manifest -> parser bundle -> analysis evidence handoff, not
to run a production corpus.

Torch acquisition-style manifest -> parser -> persistent analysis contract canary:

```bash
bash scripts/pipelines/submit_torch_stack_contract_canary.sh \
  --parser-repo ../newspaper-parsing
```

## What lives here

- `src/newsvlm_analysis/analysis_run.py`: canonical run orchestrator
- `src/newsvlm_analysis/persistent_index.py`: atomic SQLite FTS5 index
- `src/newsvlm_analysis/queries.py`: query manifests and ranked evaluation
- `src/newsvlm_analysis/evidence.py`: evidence and parser-provenance contracts
- `src/newsvlm_analysis/validation.py`: parser, evidence, and run validators
- `src/newsvlm_analysis/frontier/`: active modular analysis code
- `scripts/frontier/`: frontier entrypoints and report builders
- `scripts/pipelines/`: active issue-classifier, transcription, and recovery workflows
- `scripts/platform/`: gateway and batch execution utilities
- `prompts/frontier/`: frontier prompt bundles
- `prompts/pipelines/`: active workflow prompts
- `docs/workflows/`: current workflow documentation
- `reports/curated/`: commit-worthy report bundles
- `artifacts/`: local run outputs, scratch work, and generated reports
- `archive/legacy/`: quarantined legacy workflows, docs, prompts, and reports
- `vendor/agent-gateway/`: optional gateway submodule

## Repository organization framework

The repo now uses a strict split between active code, curated outputs, and local artifacts:

- `reports/curated/`: curated report bundles kept in git.
- `artifacts/runs/`: local run roots. Ignored by git except `artifacts/runs/README.md`.
- `artifacts/scratch/`: local one-off experiments and temporary files. Ignored by git except `artifacts/scratch/README.md`.
- `artifacts/reports/`: generated or exploratory report material not meant for version control.
- `archive/legacy/`: older flat workflows kept out of the active surface area.

Rule of thumb:
- Commit active code, workflow docs, prompts, and curated reports.
- Do not commit raw batch outputs, temporary run roots, ad-hoc scratch experiments, or generated report dumps.

## Scientific workflows

The dependency-light core intentionally uses only the Python standard library.
Install `requirements.txt` for the retained pandas, clustering, embedding,
visualization, and report workflows under `src/newsvlm_analysis/frontier/` and
`scripts/frontier/`. Provider-specific historical workflows are not part of
the canonical offline run and are not exercised by its tests or Torch canary;
their compatibility dependencies are isolated in `requirements-providers.txt`.

## Notes on reproducibility

The canonical run automatically records parser commit and validation metadata,
source-manifest fingerprints, parser model IDs, analysis commit, runtime
versions, configuration, stage timings, and the normalized query manifest.
Curated reports should cite the corresponding analysis run ID and retain its
`summary.json` and `provenance.json`.
