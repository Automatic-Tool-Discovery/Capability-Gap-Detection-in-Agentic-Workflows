import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.build_atlas_review_pack import build, page, block


class ReviewPackTests(unittest.TestCase):
    def test_pack_preserves_evidence_and_blanks_with_portable_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / 'sample'; sample.mkdir()
            (root / 'README.md').write_text('Source attribution')
            row = {'TASK': 'original-id', 'PROMPT': '<script>alert(1)</script>',
                   'ENABLED_TOOLS': '["lookup"]', 'GTFA_CLAIMS': '["expected"]',
                   'TRAJECTORY': json.dumps([{'role': 'tool', 'content': 'full evidence', 'tool_call_id': 'id'}])}
            source = sample / 'records.jsonl'
            source.write_text(json.dumps(row) + '\n')
            (sample / 'manifest.json').write_text(json.dumps({
                'records_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'selected_ids': ['original-id'], 'dataset': 'test', 'revision': 'pinned'}))
            out = root / 'pack'
            archive = build(source, out, 1)
            self.assertEqual(json.loads((out / 'raw/case_01.json').read_text()), row)
            text = (out / 'cases/case_01.html').read_text()
            self.assertNotIn('<script>alert', text)
            self.assertIn('&lt;script&gt;', text)
            self.assertIn('M001', text)
            self.assertIn('full evidence', text)
            for reviewer in ['A', 'B']:
                with (out / f'annotations_reviewer_{reviewer}.csv').open() as f:
                    annotation = list(csv.DictReader(f))[0]
                self.assertEqual(annotation['gap_status'], '')
                self.assertEqual(annotation['task_outcome'], '')
            with zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip())
                self.assertIn('pack/index.html', z.namelist())
            with self.assertRaises(FileExistsError):
                build(source, out, 1)


if __name__ == '__main__':
    unittest.main()
