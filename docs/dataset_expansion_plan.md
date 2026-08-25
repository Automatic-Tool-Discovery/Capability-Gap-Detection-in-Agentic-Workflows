# Dataset Expansion Plan

Date: 2026-08-22

## Current Dataset Position

The project currently has three dataset tracks:

- Live MCP traces: 6 controlled tasks, each run as control and gap, for 12 traces total.
- AgentRx: sample tau-retail traces are local; full Hugging Face AgentRx support exists for tau-retail and magentic-one, gated by access/authentication.
- MCP-Atlas: 50 rows are cached locally, and 5 paired ablation cases are already exported for official harness/HPC execution.

This is enough for an interim prototype claim, but not enough for a strong final evaluation claim. The key weakness is not the method; it is dataset breadth and number of validated F6 cases.

## Today Plan

1. Expand controlled live MCP tasks from 6 to at least 30 usable cases.
   Focus only on non-substitutable external capabilities: private data lookup, authenticated actions, current state, paid/API-bound services, file conversion, external database search, and side-effecting operations.

2. Run the full AgentRx public Hugging Face benchmark if access is available.
   Prioritize tau-retail first because its tool universe is already modeled. Then run magentic-one as a secondary diagnostic track, with a caveat that available-tool parsing is weaker.

3. Turn MCP-Atlas from prepared inputs into scored paired results.
   First run one baseline/ablated smoke pair. If baseline score is higher than ablated score, run the remaining four prepared pairs. Add more pairs only after the harness path is proven.

4. Re-run clean evaluations and replace stale outputs.
   The current AgentRx summary is stale and still mentions the removed heuristic baseline. Final tables should compare `llm-fair` and `capmatch-fair`.

## HPC Use

Use HPC for batch generation and scoring, not for model training.

Best HPC targets:

- MCP-Atlas official harness runs.
- Larger live-trace generation batches with low concurrency.
- Repeated LLM judge/classifier runs for stable result tables.
- Ablation sweeps over more withheld tools.

Initial job shape:

- CPU job, no GPU required.
- `concurrency=1` for smoke tests.
- Strict per-run output files for baseline and ablated conditions.
- Cache Hugging Face and package artifacts in workspace/project storage, not home.

## Dataset Acceptance Criteria

A new case should count as a validated capability-gap example only if:

- The task succeeds or plausibly progresses with the full toolset.
- The task fails when all tools for one required capability are withheld.
- The missing capability cannot be substituted by another visible tool or model-internal knowledge.
- The gold missing capability is recorded in the trace.
- The generated capability request names the missing capability and includes inputs, outputs, and rationale.

## Minimum Dataset Target

The minimum target is at least 30 validated capability-gap cases, not 15-20.
Use this split:

- 15 live MCP controlled-gap cases from our own tool server.
- 10-15 MCP-Atlas paired ablation cases scored by the official harness.
- 5+ AgentRx intent-not-supported / F6-compatible cases if available through the full Hugging Face benchmark.

If AgentRx contributes fewer usable F6 cases, fill the gap with more MCP-Atlas ablations before expanding live tasks beyond 20. This keeps the final dataset from looking entirely self-authored.

## Near-Term Target

Aim for a results table with:

- At least 30 validated capability-gap cases total.
- 15-20 controlled live F6 cases.
- 29 AgentRx tau-retail traces.
- 44 AgentRx magentic-one traces, marked as partial-support if tool-universe coverage remains incomplete.
- 10-25 MCP-Atlas paired ablation cases scored by the official harness.

That would support a much stronger thesis claim: the method works on controlled capability gaps, is compatible with real failure benchmarks, and can be scaled through paired ablations on MCP-Atlas.
