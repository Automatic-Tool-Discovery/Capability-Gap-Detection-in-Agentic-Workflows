# Day 1–2: trace validity foundations

Implemented 2026-10-05 against the existing working tree. Scope follows the implementation plan in the project review: separate experiment intent from evidence, remove metadata leakage, and enforce enabled tools. No live generation or paid model evaluation was performed.

## Implemented

- New live records use schema version 2. `experiment` records task ID, shared pair ID, condition, actual withheld names, intended capabilities, and expected label. None of these establishes gold truth.
- New records start with `annotation.status=unreviewed`, unknown outcome, and empty gold fields. They include model, UTC recording time, generator provenance, and a complete enabled inventory from the MCP session. Completeness describes the inventory, not the correctness of tool descriptions or implementations.
- Unreviewed/unresolved records cannot be parsed with gold annotations. Verified records require a label, known outcome, reviewer, method, and evidence. These are provenance requirements, not an automated proof of correctness. Outcome validators and annotation import are subsequent work.
- Existing records remain readable as version 1 with legacy annotation status. Explicit archived labels are preserved; missing gold capabilities are no longer inferred from prose about withheld tools. Archives are not rewritten or silently certified as verified. Legacy scoring remains possible and must be identified as historical.
- `--label-clean-controls` fails before loading data or calling models. The old Python helper also rejects automatic success labeling. An entirely unlabeled evaluation stops before classifier calls.
- Both fair classifiers now use one allowlisted evidence builder: task, available tools and capabilities, enabled-tool schemas, plan, complete recorded calls/results, and final response. IDs, server identifiers, experiment metadata, annotations, provenance, gold fields, and hidden-tool schemas are excluded. Oracle mode explicitly adds only the failure explanation.
- Calls to tools outside the offered inventory are rejected before MCP dispatch and recorded as errors. Enabled calls still execute. Unknown withholding configuration names fail before the model call.
- README and KNOWLEDGE identify the migration and point readers to the current experiment decision. No metric or historical result was regenerated.

## Compatibility

New live generation commands still work, but their output is an unreviewed collection rather than an automatically labeled evaluation set. Do not restore the retired flag to score it. Review task fulfillment and alternative capabilities using recorded evidence, then attach independently supported annotations.

The `gold_missing_tools` Python argument remains for caller compatibility; it now populates only `experiment.intended_missing_capabilities`. Existing condition-bearing trace IDs remain bookkeeping identifiers and are excluded from classifier inputs. A new pair ID is shared by the two executions of a task within a collection run.

The old cleaned-Qwen builder is a historical reconstruction utility; its call/error diagnostics are not outcome validators and should not be used to certify new runs.

## Verification

Run `.venv/bin/python -m unittest discover -s tests -v`.

Regression cases cover metadata-invariant fair inputs, equal evidence access, excluded hidden schemas, explicit oracle rationale, rejected automatic F0 labels, review provenance requirements, no model calls for wholly unlabeled evaluation, blocked hidden tools, working enabled alternatives, expected-label separation, unknown withholding configuration, and serialization round trips. Existing Atlas and core tests remain in the suite.

## Remaining work

This foundation does not create natural F6 labels or remove the limits of a constructed pilot. The `realtime_120` suite is the expanded controlled pilot; its results must be described as detection of the assigned tool-withholding condition, not as natural failure performance.

Further research work includes outcome validation and review ingestion, request-capable baseline comparison, checkpointed evaluation, AgentRx inventory reconstruction, semantic scoring, grouped partitions, and token/cost accounting. The synthetic Atlas adapter is not evidence for the natural-source result. The original Atlas sample and human review artifacts are unchanged.
