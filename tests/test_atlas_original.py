import json
import unittest

from scripts.prepare_mcp_atlas_original import audit_row, enabled_names, select_rows


class OriginalAtlasTests(unittest.TestCase):
    def test_selection_preserves_records_and_ignores_order(self):
        rows = [{"TASK": str(i), "TRAJECTORY": "untouched"} for i in range(20)]
        chosen = select_rows(rows, 10, "seed")
        self.assertEqual(chosen, select_rows(list(reversed(rows)), 10, "seed"))
        self.assertTrue(all(any(r is original for original in rows) for r in chosen))
        with self.assertRaises(ValueError):
            select_rows(rows, 21, "seed")
        with self.assertRaises(ValueError):
            select_rows(rows + [rows[0]], 10, "seed")

    def test_tool_metadata_and_terminal_tool_result_are_not_failure_labels(self):
        row = {"TASK": "a", "ENABLED_TOOLS": json.dumps([{"name": "lookup"}]),
               "TRAJECTORY": json.dumps([
                   {"role": "assistant", "content": "I will search", "tool_calls": [
                       {"id": "1", "function": {"name": "lookup", "arguments": "{}"}}]},
                   {"role": "tool", "tool_call_id": "1", "content": [
                       {"text": "Temporary error", "isError": True}]}])}
        audit = audit_row(row)
        self.assertEqual(enabled_names(row["ENABLED_TOOLS"]), ["lookup"])
        self.assertEqual(audit["explicit_error_block_count"], 1)
        self.assertFalse(audit["has_terminal_assistant_text"])
        self.assertEqual(audit["issues"], ["no_terminal_assistant_text"])
        self.assertIsNone(audit["gold_label"])
        self.assertIsNone(audit["task_outcome"])

    def test_malformed_trajectory_remains_unreviewed(self):
        audit = audit_row({"TASK": "b", "ENABLED_TOOLS": "[]", "TRAJECTORY": "bad"})
        self.assertIn("unparseable_trajectory", audit["issues"])
        self.assertEqual(audit["annotation_status"], "unreviewed")


if __name__ == "__main__":
    unittest.main()
