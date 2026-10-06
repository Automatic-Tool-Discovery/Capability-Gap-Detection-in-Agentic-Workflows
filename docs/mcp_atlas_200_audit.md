# Original MCP-Atlas 200-record audit

Date: 2026-09-07. No tool-removal experiments or model calls were performed.

## Source and reproducibility

Downloaded the public 500-row Parquet and dataset card from `ScaleAI/MCP-Atlas`, pinned to revision `8c563b55d7c967755f474299848049834d624617`. The dataset is CC-BY-4.0; retain attribution to MCP-Atlas and its authors when distributing derivatives. The copied upstream README remains unchanged.

Selected 200 distinct task IDs by ascending SHA256 of `2026-09-07:<task_id>`. Selection does not depend on failure cues, labels, or classifier predictions. This is a reproducible sample, not a stratified or representative-failure guarantee. All five original fields and their values are preserved, including complete serialized trajectories, tool lists/metadata, and reference claims. Reference claims belong to evaluation evidence, not classifier input.

Files under `data/benchmarks/mcp_atlas_original/`:

- `MCP-Atlas.parquet`: original full source snapshot.
- `README.md`: upstream source card.
- `sample_200/records.jsonl`: 200 unchanged source records, serialized to JSONL.
- `sample_200/manifest.json`: revision, source/sample/script checksums, seed, and selected IDs.
- `sample_200/audit.json`: structural inventory and limitations.
- `sample_200/review_queue.jsonl`: per-record structural flags and blank annotation fields. Keep human annotations separate from regenerated audit files.

Rebuild into a NEW output directory, to avoid overwriting annotations:

```bash
.venv/bin/python scripts/prepare_mcp_atlas_original.py --output /tmp/atlas-original-reproduction
```

The script refuses existing output directories. It supports both upstream enabled-tool encodings: string names and objects with `name`/parameter metadata. Source values are never normalized in `records.jsonl`.

## Findings

| Check | Result |
|---|---:|
| Source records | 500 |
| Selected unique tasks | 200 |
| Recorded tool calls | 963 |
| Distinct enabled tool names | 208 |
| Trajectories ending with a tool message | 198 |
| Trajectories ending with assistant text and no calls | 2 |
| Records with explicit `isError=true` content blocks | 0 |
| Records with a called tool outside their declared inventory | 1 |
| Missing linked call/results detected | 0 |
| Gold labels assigned | 0 |

**This is an external reference-trajectory sample, not an established natural-failure dataset.** The [upstream dataset card](https://huggingface.co/datasets/ScaleAI/MCP-Atlas) describes TRAJECTORY as the sequence resolving the task and states that diagnostic results/processes are not included in the public release. It provides neither F6 labels nor a general collection of observed failed attempts.

Absence of a terminal answer is missing evidence, not a failure label. Absence of explicit error blocks is not proof of successful task completion; errors may be embedded in text and calls can succeed without satisfying the task. Enabled tool names alone do not establish semantic coverage. We have not established the number of naturally occurring F6 cases.

Inspected the three structural exceptions:

- `689cd6f8522029b7ad7b2015`: calls `filesystem_read_file`, absent from its declared inventory. A recorded call outside the list is an inventory/provenance discrepancy, not automatic evidence of a missing capability. It needs source reconciliation.
- `689a373e148a49f91233dfca`: ends with assistant planning text about extracting artwork dimensions, not a completed user-facing answer. This illustrates why terminal assistant text cannot be automatically labeled success.
- `689bd255c0422b257e7dfcde`: ends with a statement about a $54.75 meal expenditure. This is a candidate answer, not a validated success label; full claims and task constraints still require review.

## Consequence for the project

Preserve these 200 original records as reference evidence. Do not assign F0/F6 labels from tool-call errors, missing final answers, or the last recorded message. This sample is not part of the controlled live pilot and is not a scored evaluation set.

The audit did not establish natural F6 examples in MCP-Atlas. The current project plan uses the source only for feasibility/reference analysis and keeps it separate from the constructed tool-withholding pilot. New unmodified harness runs might produce observed errors but do not guarantee missing-capability cases; ablated runs are controlled interventions and must be labeled as such.

The existing `--benchmark mcp-atlas` route points to an older synthetic adapter. **Do not use that route to evaluate this original sample.** Use the source-preserving preparation script and its audit artifacts for ingestion checks only.

## Validation

Tests cover stable selection despite source reordering, preservation of source objects, duplicate/undersized population rejection, both tool-name formats, explicit errors without invented failure labels, missing final answers, and malformed trajectories remaining unreviewed. Full source-to-sample equality and manifest checksums are checked after generation. Existing core tests are also run; their passing does not validate the old synthetic evaluation methodology.
