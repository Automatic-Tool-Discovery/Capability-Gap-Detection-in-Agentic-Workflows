# Research project

1. Project Overview
# Capability Gap Detection in Agentic Workflows

This project investigates how execution traces from tool-using AI agents can be analyzed to identify capability gaps.

A capability gap occurs when an agent fails because the required functionality is unavailable in the current toolset. Detecting such gaps is an important step toward automated tool synthesis and adaptive agent systems.

The project is inspired by AgentRx, which diagnoses failures from execution traces, but extends the idea toward identifying missing capabilities and generating structured capability specifications.
2. Research Goal
## Research Goal

Given:

- A user task
- Available tools
- Agent execution trace

Determine:

1. Why the agent failed
2. Whether the failure is caused by a capability gap
3. What missing capability would be required to complete the task
3. Failure Taxonomy
## Failure Taxonomy

| Label | Description |
|---------|---------|
| F0 | Success (No Failure) |
| F1 | Reasoning or Planning Error |
| F2 | Wrong Tool Selected |
| F3 | Wrong Tool Parameters |
| F4 | Tool Runtime Error |
| F5 | Tool Documentation or Schema Error |
| F6 | Missing Capability Gap |
| F7 | Insufficient User Information |
| F8 | Environment or State Error |
4. Dataset
## Dataset

The current dataset contains 20 manually constructed execution traces.

Each trace contains:

- User task
- Available tools
- Agent plan
- Tool calls
- Observations and errors
- Final response
- Gold failure label
- Failure explanation

The dataset covers all failure categories defined in the taxonomy.
5. Baseline Method
## Baseline Method

A heuristic rule-based classifier is implemented as the initial baseline.

The classifier analyzes:

- User task
- Tool call errors
- Failure explanations
- Available tools

and predicts one of the defined failure categories.

The baseline serves as a reference point for future LLM-based approaches.
6. Results

This is the most important addition today.

## Evaluation Results

Dataset Size: 20 traces

Overall Accuracy: 90%

| Metric | Value |
|----------|----------|
| Accuracy | 90% |
| Macro F1 | 0.89 |
| Weighted F1 | 0.89 |

The baseline performs well on the current manually curated dataset and establishes a benchmark for future experiments.
7. Current Pipeline
## Current Pipeline

User Task
    ↓
Agent Execution Trace
    ↓
Heuristic Classifier
    ↓
Failure Category

Example:

Task: Extract text from handwritten image

Available Tools:
- read_file
- summarizer

Prediction:
F6_missing_capability_gap
8. Future Work

This is where your thesis contribution starts.

## Future Work

The current system identifies failure categories using heuristic rules.

Future work includes:

- Expanding the trace dataset
- Developing an LLM-based classifier
- Generating structured capability specifications from F6 traces
- Supporting downstream tool synthesis workflows
- Comparing heuristic and LLM-based approaches