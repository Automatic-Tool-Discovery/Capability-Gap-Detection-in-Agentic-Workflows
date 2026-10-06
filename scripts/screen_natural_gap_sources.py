"""Index locally acquired original records for review; never assign F6 labels.

Run after acquiring pinned source snapshots listed in the search report. Keyword
hits are recall-oriented retrieval cues, not evidence that a capability is absent.
Raw records and annotation text stay in the gitignored local review directory.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

CUES = re.compile(
    r"intent not supported|capabilit|unavailab|not support|missing (?:tool|file)|"
    r"no (?:tool|function)|(?:cannot|can't|unable to) (?:access|process|read|view|execute)|"
    r"(?:don't|do not) have (?:access|a tool)|video|audio", re.I
)
DEFAULT = Path('outputs/review/natural_gap_search_2026-10-05')


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def walk(obj):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from walk(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk(value)


def index_sources(base: Path):
    records, problems = [], []

    def add(source, key, path, question, annotation, screening_text, **extra):
        record = dict(
            case_id=f'{source}:{key}', source=source, source_path=str(path),
            question=question, question_text_sha256=digest(' '.join(str(question).lower().split()).encode()),
            original_annotation=annotation,
            screening_cues=sorted(set(m.group().lower() for m in CUES.finditer(screening_text))),
            review_status='unreviewed', gold_label=None,
            **extra,
        )
        records.append(record)

    for domain, filename in [('tau_retail', 'tau_retail_dataset.jsonl'), ('magentic_one', 'magentic_dataset.jsonl')]:
        raw = base / 'raw' / 'AgentRx-HF'
        annotations = [json.loads(x) for x in (raw / f'{domain}.jsonl').read_text().splitlines() if x.strip()]
        trajectories = [json.loads(x) for x in (raw / filename).read_text().splitlines() if x.strip()]
        by_id = {str(r['trajectory_id']).removeprefix(domain + '_'): r for r in trajectories}
        for ann in annotations:
            key = str(ann['trajectory_id'])
            trace = by_id.get(key)
            if trace is None:
                problems.append(dict(source='AgentRx', case_id=key, error='unmatched annotation'))
                continue
            cats = [f.get('failure_category', '') for f in ann.get('failures', [])]
            root = next((f for f in ann.get('failures', []) if str(f.get('failure_id')) == str(ann.get('root_cause_failure_id'))), {})
            add('AgentRx', domain + '/' + key, raw / filename, trace.get('instruction', ''), ann,
                json.dumps(ann), source_trajectory_id=trace['trajectory_id'], source_categories=cats,
                root_category=root.get('failure_category'), tool_inventory_status='requires_reconstruction')

    root = base / 'raw' / 'AgentHallu' / 'AgentHallu'
    for path in sorted(root.rglob('*.json')):
        row = json.loads(path.read_text())
        annotation = {k: v for k, v in row.items() if k.startswith('hallucination') or k == 'is_hallucination'}
        text = str(row.get('hallucination_reason', '')) + ' ' + ' '.join(
            str(s.get('content', '')) for s in row.get('history', []))
        add('AgentHallu', str(path.relative_to(root)), path, row.get('question', ''), annotation, text,
            source_domain=path.parent.name, tool_inventory_status='not_established',
            origin_caveat='BFCL construction requires separate audit' if path.parent.name == 'BFCL' else None)

    root = base / 'raw' / 'trail-benchmark' / 'benchmarking'
    for path in sorted((root / 'data' / 'GAIA').glob('*.json')):
        row = json.loads(path.read_text())
        annpath = root / 'processed_annotations_gaia' / path.name
        try:
            annotation = json.loads(annpath.read_text())
        except (ValueError, OSError) as exc:
            annotation = None
            problems.append(dict(source='TRAIL-GAIA', path=str(annpath), error=str(exc)))
        examples = [v['example'] for v in walk(row) if isinstance(v.get('example'), dict) and 'question' in v['example']]
        example = examples[0] if examples else {}
        add('TRAIL-GAIA', path.stem, path, example.get('question', ''), annotation,
            json.dumps(annotation or {}), source_task_id=example.get('task_id'),
            annotation_path=str(annpath), tool_inventory_status='embedded_model_inputs_require_review',
            origin_caveat='Original benchmark executions; preserve force-answer instructions in provenance')

    root = base / 'raw' / 'AgentProcessBench' / 'data' / 'AgentProcessBench'
    for path in sorted(root.glob('*.jsonl')):
        for line, text in enumerate(path.read_text().splitlines(), 1):
            row = json.loads(text)
            assistant_text = ' '.join(str(m.get('content', '')) for m in row['messages'] if m.get('role') == 'assistant')
            add('AgentProcessBench', f'{path.stem}/{row["total_index"]}', path, row.get('question', ''),
                {k: row.get(k) for k in ['step_labels', 'final_label']}, assistant_text,
                source_line=line, source_domain=path.stem, query_index=row['query_index'],
                sample_index=row['sample_index'], tool_inventory_status='explicit_tools_field',
                tool_names=[t.get('function', t).get('name') for t in row.get('tools', [])],
                origin_caveat='Five executions per query; step labels are not F6 labels; check user-side actions')

    root = base / 'raw' / 'JaWs-AgentFail'
    for path in sorted(root.rglob('*.json')):
        for index, row in enumerate(json.loads(path.read_text())):
            add('AgentFail', f'{path.relative_to(root)}#{index}', path, '', row.get('evaluation'),
                json.dumps(row, ensure_ascii=False), source_row=index,
                tool_inventory_status='workflow_configuration_required',
                origin_caveat='Release README and human annotation provenance unresolved')

    root = base / 'raw' / 'WhoWhen' / 'Who&When'
    for path in sorted(root.rglob('*.json')):
        row = json.loads(path.read_text())
        annotation = {k: row.get(k) for k in
                      ['mistake_agent', 'mistake_step', 'mistake_reason', 'mistake_type']}
        add('WhoWhen', str(path.relative_to(root)), path, row.get('question', ''),
            annotation, str(row.get('mistake_reason', '')),
            source_task_id=row.get('question_ID'), source_domain=path.parent.name,
            tool_inventory_status='role_prompts_are_not_complete_tool_inventory',
            origin_caveat='Generated agent systems are not necessarily injected failures; task overlap with AgentRx requires deduplication')

    root = base / 'raw' / 'MAST'
    for path in sorted(root.rglob('*_human.json')):
        row = json.loads(path.read_text())
        annotation = row.get('note')
        # Screen reviewer prose separately from the full trajectory and taxonomy names.
        text = json.dumps(annotation.get('text', []) if isinstance(annotation, dict) else annotation)
        add('MAST-GitHub', str(path.relative_to(root)), path,
            row.get('problem_statement', ''), annotation, text,
            source_task_id=row.get('instance_id'), tool_inventory_status='requires_reconstruction',
            origin_caveat='Source filename marks human review; taxonomy labels are not F6 labels')

    path = base / 'raw' / 'MAST-HF' / 'MAD_human_labelled_dataset.json'
    if path.exists():
        for index, row in enumerate(json.loads(path.read_text())):
            trace = row.get('trace', '')
            add('MAST-HF-human', str(index), path, '', row.get('annotations'),
                json.dumps(trace, ensure_ascii=False), source_row=index,
                source_round=row.get('round'), source_domain=row.get('mas_name'),
                trajectory_text_sha256=digest(json.dumps(trace, sort_keys=True).encode()),
                tool_inventory_status='requires_reconstruction',
                origin_caveat='Inter-annotator study; taxonomy changes across rounds; blank task text is unknown, not a deduplication key')

    summary = {
        'records_indexed': len(records),
        'by_source': dict(collections.Counter(r['source'] for r in records)),
        'records_with_retrieval_cues': dict(collections.Counter(r['source'] for r in records if r['screening_cues'])),
        'gold_labels_assigned': 0,
        'parse_or_join_problems': problems,
        'warning': 'Counts are source records, not unique independent tasks or verified natural F6 positives. Keywords are screening only. Missing hits do not exclude F6.',
    }
    (base / 'screening_index.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    (base / 'screening_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(json.dumps(index_sources(args.source_dir), indent=2))
