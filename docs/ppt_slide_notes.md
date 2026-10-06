# Presentation outline

Use this outline for the final project presentation. Add numerical results only after a current controlled-pilot run has completed and its saved predictions have been inspected.

## 1. Problem

Agents can fail because of reasoning mistakes, tool errors, missing information, or a necessary ability absent from their available tools. This project focuses on detecting the last case (F6).

## 2. Method

- `llm-fair`: direct LLM failure judge.
- `capmatch-fair`: identify task-required abilities, compare them with available tool capabilities, and produce a structured capability request when an ability appears absent.
- Both receive equivalent trace evidence; condition labels and gold fields stay hidden until scoring.

## 3. Experiment

Show the 120-prompt project-authored paired tool-withholding pilot: full-tool runs versus runs with a target capability withheld. Report completed pairs, six capability-family clusters, and condition-detection metrics. Label it a controlled pilot; withholding assignment is the target, not a natural failure label or 120 independent capability families.

## 4. Dataset feasibility

The natural-source index contains 2,414 records. A desk-screen of 124 keyword hits found zero verified natural F6 positives and 12 unresolved media/file cases. This is evidence that suitable labels were not established in the reviewed sources, not proof that such failures never occur.

## 5. Results

Populate from `outputs/controlled_gap_pilot_120/summary.json` only after the run is complete. Include failures/missing outputs and per-family results. Do not use perfect scores from archived `all`-split summaries as held-out results: those summaries list the same 60 examples for training and testing.

## 6. Limitations and next work

The controlled prompts are project-authored and clustered into six families. Natural F6 performance, independent labels, broader generalization, and the semantic quality of generated capability requests remain open questions. A larger natural-failure evaluation requires cases with complete tool inventories and verified outcomes.
