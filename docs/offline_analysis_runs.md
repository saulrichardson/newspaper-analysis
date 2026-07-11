# Offline analysis run contract

The production analysis unit is a run directory, not a notebook session or a
provider request log. A run consumes one validated parser bundle and one query
manifest, then materializes a persistent corpus, ranked evidence, evaluation,
performance, and provenance.

## Data flow

```text
parser summary + provenance + parse_input.jsonl + fused_pages/*.json
                              |
                              v
                    analysis-side validation
                              |
                              v
                deterministic word-span chunking
                              |
                              v
                  atomic SQLite FTS5 index
                              |
                  +-----------+-----------+
                  |                       |
                  v                       v
          evidence contexts       relevance evaluation
                  |                       |
                  +-----------+-----------+
                              |
                              v
                  validated analysis run bundle
```

## Query manifest

Each JSONL row requires `query`. `query_id` is strongly recommended and is
generated deterministically when omitted. Optional labels support evaluation at
three provenance levels:

- `relevant_page_ids`: parser fused-page IDs
- `relevant_source_ids`: indexed source document IDs
- `relevant_chunk_ids`: exact deterministic chunk IDs

`task` and `metadata` are preserved for downstream local inference and grouped
evaluation. Unknown top-level fields are moved into `metadata` during loading.

## Index behavior

`analysis-sqlite-fts-index-v1` stores chunk text, stable source/page IDs,
word-span metadata, parser/source provenance, token counts, and content hashes.
The index is built in a temporary file and atomically moved into place after the
relational and FTS tables commit. A failed rebuild therefore cannot replace a
known-good index.

Search text is tokenized before it reaches SQLite's query grammar. This keeps
punctuation or user-supplied operators from changing query semantics. Ranking
uses FTS5 BM25 with deterministic chunk-ID tie breaking.

## Evaluation

When a query has relevance labels, the run reports:

- hit rate at K
- mean reciprocal rank
- mean recall at K
- mean normalized discounted cumulative gain at K

Unlabeled questions remain valid production queries and produce evidence
contexts; evaluation reports `not_requested` when no labeled query is present.
A query with no retrieved evidence is also a valid result and remains visible
as an empty evidence list rather than becoming a hidden fallback.

## Failure and validation

The orchestrator fails before indexing if the parser contract is unusable. Any
runtime failure leaves `errors.jsonl` and an error `summary.json` in the run
directory. Successful runs are checked by
`offline-analysis-run-validation-v1`, including:

- required JSON and JSONL contracts
- explicit `uses_external_llm_api=false` provenance
- query/evidence row-count agreement
- readable non-empty persistent index
- parser validation status
- required performance stages
- absence of structured run errors

The CLI returns a nonzero exit code for a failed run or validation report.
