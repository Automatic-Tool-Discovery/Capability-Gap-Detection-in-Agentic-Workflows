# Search for natural, human-annotated capability-gap data

**Current result (October 6, 2026):** no reviewed public source has yet provided a verified natural F6 example set suitable for the project's evaluation. The local collection indexes 2,414 source records; that number is not a count of failures or capability gaps. The focused review screened 124 keyword hits and assigned no F6 gold labels. See the [full collection report](natural_gap_collection_2026-10-06.md) and [current experiment plan](week_experiment_decision_2026-10-06.md).

## What qualifies

A natural F6 example needs an observed task and outcome, a sufficiently complete inventory of available tools, evidence that the task requires an ability, and review showing that no available tool or valid combination can provide it. A taxonomy label, failed call, absent tool name, or task that an agent could not solve is insufficient by itself. Tool-withholding creates a useful controlled case, but it is not a naturally observed missing-capability failure.

## Sources reviewed

| Source | Why it was considered | Current limitation |
|---|---|---|
| AgentRx | Human-reviewed agent-failure taxonomy includes “Intent Not Supported,” the closest category to F6 | Only a few compatible examples are indicated in the public domains; complete tool/inventory evidence and literal absence must be checked case by case |
| AgentFail | Expert-attributed workflow failures with trace and root-cause fields | Released fields and paper counts do not establish a sufficiently large human-labeled F6 subset |
| TRAIL | Expert-annotated agent traces | Tool errors, API failures, and environment issues do not automatically establish absent capability |
| Who&When | Human failure attribution and source explanations | Attribution of an agent or decisive step is not an F6 label; repeated tasks occur across variants |
| MAST human study | Small set with human annotations | Labels concern a broader multi-agent failure taxonomy; the retrieved examples do not establish F6 |
| AgentProcessBench | Human step-level process annotations | Step quality is a different target from missing capability |
| AgentFail / AgentErrorBench / AgentHallu / AgentAsk | Related failure or capability annotations | Their task definitions and labels do not directly establish natural literal-tool gaps at the required scale |
| MCP-Atlas | Original tasks, enabled-tool lists, reference claims, and resolving trajectories | It is a task-use benchmark, not a released observed-failure set with F6 labels; see the [200-record audit](mcp_atlas_200_audit.md) |
| ToolBeHonest | Explicit missing-tool task categories and human review | Cases are deliberately constructed by removing necessary tools, so they support a controlled comparison only if the project scope changes |

The review did not establish that no suitable dataset exists. It established that the sources and records examined here do not currently support claiming a verified natural F6 benchmark. The [October 6 collection report](natural_gap_collection_2026-10-06.md) contains source revisions, acquisition notes, screening counts, and unresolved examples.

## Reporting decision

Report natural-source search as a feasibility result. Keep unresolved cases unresolved, and keep any controlled withholding study in a separate result table. If a suitable dataset becomes available, re-check its labels, task outcomes, alternatives, tool inventory, and license before adding cases to evaluation.
