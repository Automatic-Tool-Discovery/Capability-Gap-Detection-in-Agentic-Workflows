# Follow-up protocol: controlled pilot v2

This is a post-pilot revision, not a held-out confirmation of the original scores. Keep the October 16 artifacts unchanged.

## Scope fixed before running

Use `data/live_task_suites_v2.json`: 120 tasks, six families, paired full-inventory and withheld-tool conditions. All task IDs have a new `pilot120v2` prefix. The four ISS prompts that referred only to “the station” or “the space station” now explicitly name the International Space Station. All other prompts, including the India holiday request and earthquake cases that produced false positives, remain unchanged. Do not remove API failures or tune prompts to improve detector scores.

Run three complete generation/evaluation repetitions with the cached Qwen3 8B model. Each repetition has a separate directory and starts its own model server. Temperature remains the existing zero setting: repetitions measure live-source/runtime repeatability, not independent random-seed trials. They do not triple the number of independent tasks or capabilities. Generation and classification still share a model.

The array runs one repetition at a time. Each requests a maximum of 12 GPU hours; three jobs reserve up to 36 GPU hours in total, although the original generation plus evaluation took about 98 minutes. The script fails rather than overwriting an existing run directory and requires the selected model to be cached. A failed repetition must be reported and investigated, not silently discarded.

## Analysis

Report every repetition separately: input hash, completion/pair counts, tool-call errors, classifier exceptions, confusion matrices, precision/recall/F1, per-family and paired exact accuracy. Compare the same tasks across repetitions; report ranges descriptively without treating repetitions as independent new samples. Review recurring false positives and model disagreements. Retain failed controls in intervention scoring; task outcomes require separate annotation.

## Second-model stage

The runner accepts `MODEL_NAME` at submission, but a second model has not yet been chosen or staged. First smoke-test its tool calling on an allocated GPU. For a clean detector-model comparison, evaluate both models against the same saved v2 traces in separate output directories; changing generator and detector together confounds the comparison. Do not interpret running the full array under another model as an isolated detector comparison.

## Launch

Transfer the prepared follow-up bundle from the Mac to the existing HPC project, extract it there, and submit `sbatch hpc/controlled_followup.sbatch`. The bundle contains v2 tasks, the new batch script, the fixed evaluator entry point, and this protocol. It reuses the existing installed environment and cached model. No new model download is needed for the three Qwen repetitions.
