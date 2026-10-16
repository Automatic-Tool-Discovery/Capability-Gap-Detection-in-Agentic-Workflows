# Controlled pilot results — October 16, 2026

The Capella run produced 240 traces: 120 paired tasks across six families, with 20 tasks (40 traces) per family. Qwen3 8B via Ollama generated the traces and powered both detectors. Generation job `4424401` completed in 28:59; evaluation job `4424614` completed in 1:08:39. These timings and successful Slurm exit codes were supplied in the terminal transcript.

## Verified results

| Method | Accuracy | Gap precision | Gap recall | Gap F1 | Both conditions correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| llm-fair | 99.17% | 98.36% | 100% | 99.17% | 118/120 |
| capmatch-fair | 98.33% | 96.77% | 100% | 98.36% | 116/120 |

Both methods scored all 240 traces with zero classifier exceptions. Confusion matrices (rows: control, withheld; columns: predicted no gap, gap) are `[[118, 2], [0, 120]]` and `[[116, 4], [0, 120]]`, respectively. Failure-inclusive scores equal completed-case scores.

Local verification checked the input SHA-256, unique trace and prediction IDs, 120 complete pairs, matching pair IDs, withheld-tool inventory differences, family balance, and recomputed aggregate/per-family metrics and paired accuracy from saved predictions. The trace hash matches the evaluation summary: `35da4273f6e8dfa0e9d1b7db7d5ee4a853b6d2f341caba0a6b4080412beaf741`.

## Error review

All six incorrect predictions across the two methods are false positives relative to the assigned intervention, covering four distinct control traces:

- `pilot120_iss_position_12` (both methods): “the station” is ambiguous without explicit ISS context. The agent made no tool call and discussed geocoding; both detectors inferred a missing timestamp capability despite `realtime_iss_position` being available. This is a prompt/interpretation concern, not evidence of withholding.
- `pilot120_holiday_in_2027` (both): the available holiday tool returned an empty-response error for India in 2027. The agent then claimed the current year was 2023 and attributed the error to future-year coverage. That explanation is unsupported by the recorded error. Both detectors inferred missing capability; the control assignment alone does not establish task success or API coverage.
- `pilot120_quake_m41` (capmatch only): the detector demanded temporal filtering for “today’s” feed despite a successful earthquake call. Whether the response meets the time constraint requires separate task-outcome review.
- `pilot120_quake_m59` (capmatch only): the detector demanded human-readable timestamp conversion, which the user did not explicitly request. The tool returned a matching earthquake.

Generation contains four recorded tool-call errors: ISS timeouts in `pilot120_iss_position_14`, `_16`, and `_17`, plus the India holiday error. Zero classifier exceptions must not be described as zero execution errors. One control made no tool calls; all 120 gap traces made no tool calls. This strong behavioral separation may make intervention detection easier.

Both methods have 100% accuracy on book, FX, and weather; both have 97.5% on holiday and ISS. Earthquake accuracy is 100% for llm-fair and 95% for capmatch-fair. The methods disagree on only the two earthquake controls.

## Interpretation and artifacts

This single run supports detection of assigned tool withholding. It does not establish natural F6 performance, verified task outcomes, statistical superiority, or generalization across 120 independent capabilities. Prompts cluster within six families, and generation and classification share the same model. Preserve the original results when investigating prompt ambiguity and API failures; any revised suite should be a separately identified experiment.

- [Raw traces](../outputs/pilot-4424401/traces.jsonl)
- [Evaluation summary](../outputs/eval-4424614/summary.json)
- [LLM predictions](../outputs/eval-4424614/predictions_llm-fair.jsonl)
- [Capability-matching predictions](../outputs/eval-4424614/predictions_capmatch-fair.jsonl)

Each downloaded run directory also contains runtime logs and installed package versions. The original evaluation job `4424590` failed on a Python import before scoring; the successful retry explicitly set `PYTHONPATH`. The local evaluator entry point now inserts the project root, matching the smoke-test script; direct `--help` execution was verified.
