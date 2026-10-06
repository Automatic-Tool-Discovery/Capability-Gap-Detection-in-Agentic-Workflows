"""Build a portable, prediction-blind review pack from original Atlas records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import zipfile

ROOT = Path("data/benchmarks/mcp_atlas_original/sample_200")
FIELDS = ["case_id", "source_task_id", "reviewer", "review_date", "task_outcome",
          "gap_status", "missing_capability", "evidence_references",
          "alternatives_considered", "information_missing", "notes"]
RUBRIC = """# Reviewer instructions

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
"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    return '<pre>' + html.escape(text) + '</pre>'


def page(title: str, body: str) -> str:
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'">'
            '<title>' + html.escape(title) + '</title><style>'
            'body{font:17px/1.55 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#182332}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f8;padding:16px;font:14px/1.5 monospace}'
            'details{border:1px solid #ccd5df;padding:12px;margin:14px 0}summary{cursor:pointer;font-weight:600}'
            'a{color:#075bb0}li{margin:12px 0}.notice{background:#fff3ce;padding:16px}'
            '</style><body><h1>' + html.escape(title) + '</h1>' + body + '</body></html>')


def build(source: Path, output: Path, count: int = 10) -> Path:
    if output.exists() or output.with_suffix('.zip').exists():
        raise FileExistsError("Use a new output path; existing review packs must not be overwritten")
    parent = json.loads((source.parent / 'manifest.json').read_text())
    if digest(source) != parent['records_sha256']:
        raise ValueError('Source records do not match the sample manifest')
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if [r['TASK'] for r in rows] != parent['selected_ids']:
        raise ValueError('Source order differs from recorded sampling order')
    if not 0 < count <= len(rows):
        raise ValueError('Invalid pilot size')
    selected = rows[:count]  # Prespecified first ten hash-ranked records, no cherry-picking.
    output.mkdir(parents=True)
    (output / 'cases').mkdir()
    (output / 'raw').mkdir()
    (output / 'INSTRUCTIONS.md').write_text(RUBRIC)
    source_card = source.parent.parent / 'README.md'
    (output / 'SOURCE_README.md').write_bytes(source_card.read_bytes())
    links, annotations = [], []
    for i, row in enumerate(selected, 1):
        case = f'case_{i:02d}'
        messages = json.loads(row['TRAJECTORY'])
        enabled = json.loads(row['ENABLED_TOOLS'])
        annotations.append(dict.fromkeys(FIELDS, '') | {'case_id': case, 'source_task_id': row['TASK']})
        (output / 'raw' / f'{case}.json').write_text(json.dumps(row, ensure_ascii=False, indent=2))
        body = '<p><a href="../index.html">All cases</a> · <a href="../raw/' + case + '.json">Original record</a></p>'
        body += '<p class="notice">Unreviewed reference record. No outcome or F6 label has been assigned. Tool names alone may not establish capability coverage.</p>'
        body += '<h2>Task</h2>' + block(row['PROMPT'])
        body += '<p>Source task ID: ' + html.escape(row['TASK']) + '</p>'
        body += '<h2>Available tools — original inventory</h2>' + block(enabled)
        body += '<p>Only upstream metadata is shown. Complete descriptions may be missing; do not infer them from names alone.</p>'
        body += '<details><summary>Expected answer claims — reference targets, not agent output</summary>' + block(row['GTFA_CLAIMS']) + '</details>'
        body += '<h2>Recorded execution</h2><p>All messages are preserved below. Expand each message for its complete contents. Source text is escaped and never executed.</p>'
        for j, message in enumerate(messages, 1):
            ref = f'M{j:03d}'
            label = ref + ' — ' + str(message.get('role', 'unknown'))
            names = [str(c.get('function', {}).get('name', 'unknown')) for c in message.get('tool_calls') or []]
            if names:
                label += ' — ' + ', '.join(names)
            if message.get('tool_call_id'):
                label += ' — call ' + str(message['tool_call_id'])
            body += '<details id="' + ref + '"><summary>' + html.escape(label) + '</summary>' + block(message) + '</details>'
        body += '<p>End of original record. No final answer has been reconstructed or generated.</p>'
        (output / 'cases' / f'{case}.html').write_text(page(case.replace('_', ' ').title(), body))
        links.append('<li><a href="cases/' + case + '.html">' + case + '</a>: ' + html.escape(row['PROMPT']) + '</li>')
    for reviewer in ['A', 'B']:
        with (output / f'annotations_reviewer_{reviewer}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader(); writer.writerows(annotations)
    with (output / 'adjudication.csv').open('w', newline='') as f:
        fields = ['case_id', 'source_task_id', 'agreement_status', 'final_task_outcome', 'final_gap_status', 'rationale', 'unresolved_evidence', 'adjudicator']
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        writer.writerows({'case_id': a['case_id'], 'source_task_id': a['source_task_id']} for a in annotations)
    body = '<p class="notice">Suitability pilot: determine whether these records support natural capability-gap annotation. This pack is not a claim of ten failures.</p>'
    body += '<h2>Review procedure</h2>' + block(RUBRIC)
    body += '<p>Use separate annotation files: <a href="annotations_reviewer_A.csv">Reviewer A</a> · <a href="annotations_reviewer_B.csv">Reviewer B</a>. Save independent reviews before discussion. <a href="adjudication.csv">Adjudication sheet</a>.</p><ol>' + ''.join(links) + '</ol>'
    (output / 'index.html').write_text(page('MCP-Atlas: 10-record review pilot', body))
    manifest = {'source_dataset': parent['dataset'], 'revision': parent['revision'],
                'parent_records_sha256': parent['records_sha256'], 'pilot_count': count,
                'selection': 'first N records in frozen hash-ranked sample; no outcome-based selection',
                'selected_ids': [r['TASK'] for r in selected], 'builder_sha256': digest(Path(__file__)),
                'predictions_included': False, 'labels_assigned': False,
                'files_sha256': {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    archive = output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                z.write(p, str(Path(output.name) / p.relative_to(output)))
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'records.jsonl')
    parser.add_argument('--output', type=Path, default=Path('outputs/review/atlas_pilot_10'))
    parser.add_argument('--count', type=int, default=10)
    args = parser.parse_args()
    print(build(args.source, args.output, args.count))


if __name__ == '__main__':
    main()
