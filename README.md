# Capability Gap Detection in Agentic Workflows

> **New to the project?** Read [KNOWLEDGE.md](KNOWLEDGE.md) first — a full
> from-scratch guide (glossary, every file, datasets, setup, and what to do next).

## Project Overview

When a tool-using AI agent fails, sometimes the cause is not a bad plan or a buggy
tool call — it is that **the capability the task needs simply does not exist** in
the agent's toolset. We call this a **capability gap**.

This project detects capability gaps from agent execution traces and goes one step
further than diagnosis: it produces a **capability request** — a structured
description of the missing tool that *would* let the agent finish the task. This is
the step toward automated tool synthesis and self-extending agents.

**Baseline paper:** [AgentRx](https://github.com/microsoft/AgentRx) (Microsoft
Research, 2026). AgentRx *diagnoses* agent failures from trajectories and labels
them with a 9-category taxonomy. Our goal is to detect the **capability-gap** slice
of that problem *more efficiently and more usefully* — and to emit the missing
capability spec, which AgentRx does not.

## Research Goal

Given a **user task**, the **available tools**, and the **agent execution trace**:

1. Decide **why** the agent failed.
2. Decide whether the failure is a **capability gap** (F6).
3. If so, emit the **missing capability** that would be required to complete the task.

The headline contribution is step 3: turning a diagnosis into an actionable
*capability request*.

## Failure Taxonomy (F0–F8)

| Label | Description |
|-------|-------------|
| F0 | Success (no failure) |
| F1 | Reasoning or planning error |
| F2 | Wrong tool selected |
| F3 | Wrong tool parameters |
| F4 | Tool runtime error |
| F5 | Tool documentation or schema error |
| F6 | **Missing capability gap** (our focus) |
| F7 | Insufficient user information |
| F8 | Environment or state error |

---

## The Baseline: AgentRx (methods + limitations)

### What AgentRx does (method)

AgentRx pinpoints the **first unrecoverable ("critical") failure step** in a failed
trajectory and assigns it a taxonomy category. Its pipeline:

1. **Normalize** heterogeneous multi-agent logs into one intermediate representation.
2. **Synthesize constraints** in two stages:
   - *Global constraints* from the tool schema + domain policy (trajectory-independent).
   - *Dynamic constraints* from the task instruction + observed prefix (trajectory-dependent).
   - Each constraint has a **guard** (when it applies) and an **assertion** (SAT/VIOL),
     checked either **programmatically** (schema/equality/membership) or
     **semantically** (an LLM checker over natural-language rules).
3. **Validation log** — a step-indexed list of violated constraints + supporting evidence.
4. **LLM judge** reads the validation log + a **taxonomy checklist** (yes/no questions
   per category) and outputs the critical step `ŝ` and category `ŷ`. Default model: GPT-5.

### AgentRx's benchmark (3 domains, 115 failed trajectories)

| Domain | What it is | # failed traj | Public? |
|--------|-----------|---------------|---------|
| **τ-bench** (tau_retail) | retail tool-calling, single agent + simulated user + domain policy | 29 | ✅ on HF |
| **Magentic-One** | generalist multi-agent web/file tasks (5 agents) | 44 | ✅ on HF |
| **Flash** | incident-management workflow agent (Microsoft-internal) | 42 | ❌ not released |

### Limitations (our opening)

These are stated in the paper or evident from its results, and each is something we
can exploit:

1. **It stops at diagnosis.** AgentRx outputs *step + label*. It never proposes a
   fix or the missing capability. → *We add the capability request.*
2. **Capability gaps are a rare afterthought.** "Intent Not Supported" (= our F6) is
   just 1 of 9 categories and is rare in the data: **τ-bench 6.9%, Magentic 6.8%,
   Flash 0%** (≈5 cases total across the public domains). AgentRx never studies this
   class in depth. → *We focus the whole method on F6.*
3. **Even SOTA category accuracy is low** (τ-bench 40.2%, Magentic 44.4%). Lots of
   headroom. → *A specialized capability matcher can beat a generic judge on its slice.*
4. **Expensive & policy-dependent.** Many LLM calls; global constraints need a domain
   policy (Flash/Magentic have none). → *Capability matching is cheaper and
   policy-light: "is a tool for the needed capability present?"*.
5. **Taxonomy can be misled by noisy/false-positive violations** and inherits
   benchmark blind spots. → *Ground-truth F6 from our live generator avoids label noise.*

---

## Datasets

We use **real** traces only (synthetic traces were removed from the project).

### 1. AgentRx real traces (external, gated on Hugging Face)

| Domain | Loads | Status |
|--------|-------|--------|
| `tau_retail` | 29 traces, 15-tool universe, all tool-calls parsed | ✅ fully usable |
| `magentic_one` | 44 traces, gold labels parse | ⚠️ multi-agent log format: tool-calls/tools not yet parsed |
| `flash` | — | ❌ not in the public release |

> The capability-gap (F6) class is **rare in real data** — only ~5 cases across
> tau_retail + magentic_one. This scarcity is the main reason we generate our own
> ground-truth F6 traces (below).

One-time access setup:

```bash
# 1. Accept terms at https://huggingface.co/datasets/microsoft/AgentRx
# 2. Authenticate:
huggingface-cli login        # or: export HF_TOKEN=...
```

### 2. Live MCP traces (ours, dynamic, ground-truth F6)

`src/live_agent.py` runs a **live LLM agent** that decides which MCP tools to call,
calls them for real against `mcp_servers/research_tools/server.py`, and records what
happens. For each task in `data/live_tasks.json` we run it twice:

- **control** — full toolset (expected: success/normal failure).
- **gap** — *withhold every tool that provides the needed capability* (expected: F6
  with a **known** missing capability).

```bash
export SCADS_API_KEY="your-tud-ai-key"          # https://llm.scads.ai/docs/
export SCADS_MODEL="alias-huge-no-thinking"     # must support tool calling
python -m src.live_agent --mode both
# → data/live_traces.jsonl
```

Findings from live runs:

- **Withhold by *capability*, not by tool.** Removing only `calculator` did not create
  a gap — the agent used `run_python` instead. A genuine gap requires withholding
  *all* substitutable tools (`required_tools` + the shared capability vocabulary).
- **Live-data / side-effect tasks are the clean gaps** (weather, currency, send email):
  the agent correctly refuses. Pure reasoning gaps (mental arithmetic, translation) the
  LLM fakes from its own knowledge, so they are weaker F6 examples.

> **On Flash / Magentic-One / τ-bench access:** τ-bench and Magentic-One are already
> available through the AgentRx HF dataset (above) — no GitHub needed. Flash is not
> public anywhere. If you want to run the *original* τ²-bench or Magentic-One agents to
> generate **new** traces (rather than reuse AgentRx's annotated ones), that needs the
> upstream GitHub repos and is a separate, heavier track — say the word and we'll scope it.

---

## Baseline Method (single, LLM-based)

We keep **one** internal baseline: an LLM classifier
(`src/llm_classifier.py`) via the [TUD:AI API](https://llm.scads.ai/docs/usage/api/).
This mirrors AgentRx's own "LLM-as-judge" baseline, so the comparison is apples-to-apples.

| CLI name | Uses gold `failure_explanation`? | Role |
|----------|----------------------------------|------|
| `llm-fair` | No | Honest, deployable baseline |
| `llm-oracle` | Yes | Upper bound (sees the human rationale) |

(The earlier rule-based heuristic baseline was removed: it was blind to semantic
failures — it labeled ~20/29 real τ-bench failures as "success" because no tool threw
an error string — so it was not a meaningful comparator.)

---

## Our Method: the Capability Matcher

`src/capability_matcher.py` is the core contribution and where we beat AgentRx on the
capability-gap slice. Unlike AgentRx's black-box judge that picks 1 of 9 labels, the
gap decision here is **structured and grounded**:

1. **Extract required capabilities** — an LLM lists the capabilities the *task* needs
   (independent of what tools exist), each as a canonical slug + evidence.
2. **Deterministic match** — code computes `required − available = missing` against the
   trace's available capabilities. The gap decision is a verifiable set comparison
   ("probable tools vs. available tools"), not a single opaque guess. We also guard
   against the LLM hallucinating coverage by re-checking each claimed-available slug.
3. **Emit a capability request** — on a gap, label the trace **F6** *and* output a
   structured `CapabilityRequest` (name, capability, inputs, outputs, rationale): the
   spec of the tool that would let the agent finish. **AgentRx produces no such artifact.**
4. **Defer otherwise** — if nothing is missing, hand off to the LLM baseline for the
   fine-grained F0–F8 label. So the matcher is a *capability layer on top of* the baseline,
   which isolates its contribution to the F6 class.

| CLI name | Uses gold rationale? |
|----------|----------------------|
| `capmatch-fair` | No |
| `capmatch-oracle` | Yes |

```bash
# Our method vs the baseline on the real τ-retail failures
python -m src.evaluate --benchmark agentrx --agentrx-source hf --agentrx-only \
  --method llm-fair capmatch-fair --split all --save-predictions
```

With `--save-predictions`, detected gaps write `missing_capabilities` and
`capability_requests` into `outputs/evaluation/predictions_*.jsonl`.

---

## Project Structure

```
src/
  schemas.py              AgentTrace / ToolCall / Prediction / CapabilityRequest
  taxonomy.py             F0–F8 definitions
  llm_classifier.py       the single baseline (LLM-as-judge)
  capability_matcher.py   OUR METHOD: required-vs-available capabilities → F6 + capability request
  live_agent.py           dynamic MCP trace generator (ground-truth F6)
  evaluate.py             run the baseline over datasets + metrics
  evaluation/
    capabilities.py       shared capability vocabulary (cross-dataset alignment)
    metrics.py            accuracy, macro/weighted F1, F6 P/R/F1, binary gap F1
    splits.py             all / random / loo / cv5
    benchmarks/
      agentrx.py          AgentRx loader (tau_retail + magentic_one) + taxonomy map
      mcp_atlas.py        MCP-Atlas loader (synthetic F6 by withholding a required tool)
data/
  live_tasks.json         task defs + required_tools for ground-truth F6
  live_traces.jsonl       generated live traces
  benchmarks/agentrx_samples/   public AgentRx sample trajectories (ungated)
mcp_servers/research_tools/server.py   one MCP server, ~17 tools
fixtures/                 files the MCP tools read (csv/pdf/png/txt)
papers/                   reference papers
outputs/evaluation/       evaluation summaries + per-trace predictions
```

---

## How to Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export SCADS_API_KEY="your-tud-ai-key"
```

```bash
# Baseline on the real AgentRx τ-retail data (29 failures)
python -m src.evaluate --benchmark agentrx --agentrx-source hf --agentrx-only \
  --method llm-fair --split all --save-predictions

# Baseline on our live MCP traces (ground-truth F6)
python -m src.evaluate --method llm-fair --split all

# Quick offline sanity check (public AgentRx samples, no HF auth)
python -m src.evaluate --benchmark agentrx --agentrx-source samples --agentrx-only \
  --method llm-fair --split all
```

Summaries → `outputs/evaluation/summary_<split>_<benchmark>.json`.
Per-trace gold-vs-pred (with `--save-predictions`) → `outputs/evaluation/predictions_*.jsonl`.

### What gets measured

| Metric | Meaning |
|--------|---------|
| Accuracy | Exact match on F0–F8 |
| Macro / weighted F1 | Multi-class performance |
| F6 precision/recall/F1 | Capability-gap class only |
| Binary gap-detection F1 | F6 vs everything else (the cleanest target) |

---

## Cross-dataset alignment & taxonomy mapping

Every trace — AgentRx, MCP-Atlas, or live — is one `AgentTrace`
`{user_task, available_tools, tool_calls, gold_label, source, domain, capabilities}`.
`src/evaluation/capabilities.py` maps each benchmark's raw tool names onto canonical
capabilities (e.g. both user-lookup tools → `user_lookup`) so "missing capability"
means the same thing everywhere.

**Two flavors of F6 surfaced by alignment:**

| Source | What "capability gap" means |
|--------|-----------------------------|
| MCP-Atlas / live | A required tool is **literally absent** from the toolset |
| τ-bench ("Intent Not Supported") | The tools **exist** but the action is **disallowed by domain policy** |

So a "required-tool-missing" matcher handles the literal gaps directly; τ-bench gaps
additionally need **policy awareness** — a key design point for the capability matcher.

**AgentRx → our taxonomy (intentionally lossy; the mismatch is itself a finding):**

| AgentRx category | Our label |
|------------------|-----------|
| Intent Not Supported | **F6 missing capability gap** |
| Invalid Invocation | F3 wrong tool parameters |
| Underspecified User Intent | F7 insufficient user information |
| System Failure | F4 tool runtime error |
| Guardrails Triggered | F8 environment or state error |
| Instruction/Plan Adherence, Invention of New Info, Misinterpretation of Tool Output, Intent-Plan Misalignment, Inconclusive | F1 reasoning or planning error |

## Key finding so far

On the 29 real τ-retail failures, the LLM baseline scores low on exact F0–F8 accuracy,
but per-trace inspection shows **much of the "error" is taxonomy disagreement, not bad
reasoning**: e.g. `tau_retail_91` is gold F7, but the LLM calls it F6 with a sound
justification (*"no return/exchange/refund tools exist in available_tools"*). This means
(a) the LLM's F6 calls are semantically grounded, and (b) fair evaluation should center
on the **binary capability-gap task**, which is exactly what our method targets.
