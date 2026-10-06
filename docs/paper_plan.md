# Report outline and evidence limits

This is the writing outline for the current project scope. Use generated, saved pilot outputs for numeric results; do not copy numbers from historical all-split summaries into a held-out-results table.

## 1. Problem and scope

Define F6 as a task requiring an ability unavailable through the agent's offered tools or valid substitutes. Separate literal capability absence from execution errors, bad parameters, policy restrictions, missing user information, and model knowledge failures.

## 2. Methods

Describe the shared model-visible evidence and compare `llm-fair` with `capmatch-fair`. State that both methods see the same task, offered tool inventory and schemas, recorded tool calls/results, plan, and final response, while bookkeeping IDs, gold fields, and withheld-tool metadata are excluded. Explain that the capability matcher converts task requirements into an available-versus-missing comparison and may emit a structured request.

## 3. Evaluation

Present the project-authored 120-prompt paired tool-withholding study as a **controlled pilot**. Give the number of completed pairs, six family clusters, missing runs, model/runtime details, and assigned-condition confusion matrices. Measure precision, recall, F1, failure/coverage, per-family behavior, and request quality only where a defensible reference target exists. Keep natural-source screening in a separate dataset-feasibility section.

## 4. Data feasibility and limitations

The natural-source review indexed 2,414 records and screened 124 keyword hits. It found zero verified natural F6 positives and 12 unresolved boundary cases. Explain that the review was an assistant desk screen rather than independent human adjudication, and that unresolved cases are not gold labels. MCP-Atlas records are reference task trajectories, not verified failure cases. See the detailed [natural-source report](natural_gap_collection_2026-10-06.md) and [MCP-Atlas audit](mcp_atlas_200_audit.md).

## 5. Results and interpretation

Interpret pilot results only as recognition of the assigned intervention in this constructed suite. Report sample size and clustering alongside every metric. Avoid claims of natural-failure performance, broad generalization, or superiority based on historical scores whose training and test examples overlap.

## 6. Conclusion

Summarize what the implementation demonstrates, what the controlled pilot supports, and what remains untested: natural F6 performance, independent adjudication, generalization across capability families, and semantic correctness/buildability of generated tool requests.

## Reproducibility checklist

- Keep raw traces, run manifest, predictions, and summary together.
- Record model/provider/revision, prompts, code revision, task suite, timestamps, and failures.
- Preserve unresolved annotations as unresolved; do not infer task success from control condition or failure from withholding condition.
- Generate report tables from the saved predictions and evaluator output.
- Keep historical artifacts clearly marked and do not overwrite them with new metric versions.
