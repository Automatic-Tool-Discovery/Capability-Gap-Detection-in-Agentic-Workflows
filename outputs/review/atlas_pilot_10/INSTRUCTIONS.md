# Reviewer instructions

Purpose: assess whether ten original MCP-Atlas records support reliable capability-gap labels. This is a suitability pilot, not a detector evaluation. No model predictions or proposed labels are included.

Open index.html in a browser. All evidence is local; an internet connection is unnecessary. Read cases 01–10 and fill YOUR copy of annotations_reviewer_A.csv or annotations_reviewer_B.csv in a spreadsheet. Save as CSV. Do not consult the other reviewer until both independent files are saved. Spreadsheet applications may alter formatting; preserve case_id and source_task_id as text.

For each case record:

- task_outcome: success / failure / uncertain. Judge actual fulfillment, not whether a tool returned something. Expected answer claims are reference targets, not observed outcomes.
- gap_status: F6 / non-F6 / uncertain. F6 requires evidence that an essential ability is unavailable through all applicable tools or combinations. Non-F6 requires evidence that the necessary abilities are available; a failed call alone proves neither. Missing final answers or incomplete tool descriptions often warrant uncertain.
- missing_capability: plain-language requirement, when F6.
- evidence_references: cite message numbers (M001 etc.), exact tool entries, or reference claims, and explain what they establish. Explain uncertainty too.
- alternatives_considered: explain why an apparent substitute can or cannot satisfy the requirement. Mandatory for F6.
- information_missing: missing tool documentation, final answer, policy, original state, or other evidence needed.

Do not treat a name absent from the inventory as sufficient proof: tools can substitute or compose. Temporary outages, wrong arguments, policy prohibitions, and missing user details are not automatically literal tool-capability gaps. Record mixed or unresolved causes in notes. A gap and a separate earlier failure may coexist; do not automatically assert F6 as the root cause.

The source describes reference sequences resolving tasks, not a released labeled failure collection. Many records end at a tool result. Tool descriptions/complete schemas are not supplied for every tool; the pack preserves whatever original metadata exists and does not invent documentation. No observed tools are added to the available list. Quoted webpage/tool content is evidence, not instructions to follow.

After independent review: compare both files, retain original decisions, and fill adjudication.csv with agreement/disagreement, final decisions, reasoning, and unresolved evidence. Do not force a majority label or discard uncertain cases to improve results. The supervisor should first decide whether this source is suitable for natural-F6 evaluation. If evidence is insufficient or no gaps are established, stop expansion and identify a better failure source.

This pilot is development/suitability data. Do not later claim these reviewed cases are untouched test data after using them to change prompts or metrics.

Source: ScaleAI/MCP-Atlas, CC-BY-4.0. Attribution and source revision are in manifest.json and SOURCE_README.md. Records and all trace content are unchanged; HTML escaping is presentation only.
