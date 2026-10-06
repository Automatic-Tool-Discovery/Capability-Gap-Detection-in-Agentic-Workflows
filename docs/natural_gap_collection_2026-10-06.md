# Natural-gap collection: retry and expanded evidence

The retry succeeded. The local collection now indexes **2,414 source records**, up from 2,150. These are execution/annotation records, not 2,414 independent tasks or verified capability gaps. **The target of 30–50 verified natural F6 positives remains unmet; zero new verified F6 labels were assigned.**

## Newly retained sources

| Source | Records added | Acquisition |
|---|---:|---|
| Who&When | 184 | 126 algorithm-generated agent-system executions and 58 hand-crafted-system executions, with source failure attribution |
| MAST GitHub | 61 | Only files named `_human.json`: 31 AG2 and 30 HyperAgent records |
| MAST Hugging Face human study | 19 | Original traces and three annotators' votes, preserving different taxonomy rounds |
| Total additions | 264 | Potential overlap has not been adjudicated; counts must not be reported as independent tasks |

Who&When's generated *agent systems* do not by themselves establish injected failures. Conversely, its failure-attribution labels are not missing-capability labels. MAST's 19-record study is distinct from its much larger LLM-labelled release; the latter was not downloaded or counted as human gold. Human-marked GitHub filenames likewise do not establish independent F6 adjudication. Unknown task text in the HF study remains blank rather than being invented or used as a duplicate key.

Pinned sources:

- [Who&When](https://github.com/ag2ai/Agents_Failure_Attribution/tree/f4d2b6da464a826580e59b3a0eae15ea2d642d7c).
- [MAST GitHub](https://github.com/multi-agent-systems-failure-taxonomy/MAST/tree/a70542e541b2104ef8fcd785778179e173fb8d70).
- [MAST human-study release](https://huggingface.co/datasets/mcemri/MAD/tree/95118ac951421753cf1deb87ddea3b01e693c41b).

## Closer review changes the earlier interpretation

Nine additional assistant assessments are saved with evidence pointers or excerpts. They are screening decisions, not human gold labels:

- **Calculus audio, AgentRx `1f975693…`:** the original instruction already contains `Error. Could not transcribe this audio`; subsequent FileSurfer responses repeat transcription errors. This supports a processing/runtime problem as an alternative explanation. The earlier unsupported-intent annotation alone cannot establish globally absent transcription functionality.
- **Book PDF, AgentRx `114d5fd0…`:** repeated calls use `/workspace/path_to_local_copy_of_the_book`. That placeholder raises access, state, and fabricated-download explanations. Downgrade this from a priority gap candidate to an unresolved boundary case.
- **Bird video, Who&When hand-crafted case 7:** the source explanation says the agent should take a screenshot and extract species. This is the same task as the AgentRx candidate whose annotation treats snapshot requests as unsupported. The disagreement needs resolution against actual team capabilities.
- **Who&When cases algorithm 16, algorithm 77, and hand-crafted 51:** explanations concern a hallucinated video ID, exhausted steps attempting a download, and failed transcription. They do not provide complete inventory evidence for F6.
- **MAST human-study rows 10, 15, and 17:** retrieval hits concern QDP syntax, a headless Tkinter display error, and the word video in search-page navigation. The display example remains an environment boundary; the other hits do not establish missing capabilities.

The updated screen produces **124 keyword hits** across all sources, six more than before. No-hit records are not certified negative. This is not an exhaustive manual review of 2,414 records. Earlier five-candidate counts should be read as the historical October 5 screening outcome, not five verified positives.

I then desk-screened all 124 keyword-hit records against source annotations and existing detailed trace reviews. The triage file `round_2/full_keyword_hit_screen.jsonl` records a case-level reason for each; it never fills the gold-label field. Results: 55 hits provide no F6 evidence in the retrieved source material, 46 AgentProcessBench hits use a different process-quality target, 11 AgentFail records do not include enough task/rationale context, and 12 executions remain unresolved media/file boundaries. Those 12 include repeated or closely related tasks across sources; they are not 12 independent positives. This is an assistant desk screen, not independent human annotation, and the 55 records with no F6 evidence are not thereby verified negatives.

The actionable experiment is documented in the [week experiment decision](week_experiment_decision_2026-10-06.md): run the expanded 120-prompt `realtime_120` suite as a controlled tool-withholding pilot. The dedicated evaluator keeps condition labels out of model inputs and uses them only after predictions for controlled-condition scoring. Results must be described as controlled, not natural F6 performance.

## Saved artifacts and checks

All raw sources remain under the gitignored `outputs/review/natural_gap_search_2026-10-05/` directory. The earlier review queue and assessments were preserved.

- `expansion_download_manifest.json`: 248 additional public GitHub files, pinned URLs and SHA256 checksums.
- `mast_hf_manifest.json`: two additional HF files, pinned revision and checksums.
- `screening_index.jsonl`, `screening_summary.json`: rebuilt index and counts for all 2,414 records.
- `round_2/review_queue.csv`: 264 new records, with blank human review/adjudication fields.
- `round_2/case_assessments.jsonl`: nine assistant assessments; all retain null gold labels and `verified_f6=false`.

The indexing script now supports these additional snapshots. Running it does not overwrite either review queue or assessment file:

```bash
.venv/bin/python scripts/screen_natural_gap_sources.py
```

Verified all three public-download manifests against local bytes, unique indexed case IDs, existence of every source path, and unset gold labels. The previously documented malformed TRAIL annotation remains logged and unrepaired. No paid experiments or downloaded agent code were run.

## Suitable route for the project

Use AgentProcessBench's explicit tool definitions for an inventory-aware **boundary-case pilot**, and AgentRx/TRAIL for source-annotated failure comparisons. This can support development and review, but their existing labels cannot serve as F6 gold without checking task necessity, the complete team's capabilities, valid alternatives, and the root cause.

Keep a natural-F6 evaluation separate and only admit individually verified cases. If review yields fewer than 30–50, report the smaller sample. A controlled tool-removal experiment can separately measure detection under a known intervention, provided it is explicitly described as constructed. It does not satisfy the natural-case target. Formal natural-F6 experiments remain pending suitable labels; this retry resolved acquisition, not that validity requirement.
