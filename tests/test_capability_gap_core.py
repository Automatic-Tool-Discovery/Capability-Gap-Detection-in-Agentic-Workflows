"""Regression tests for the capability-gap pipeline.

These tests cover the project pieces that should work without a live LLM call:
capability normalization, MCP-Atlas gap construction, metric calculation,
offline capability-matcher behavior with a fake model response, and export of
baseline/ablated MCP-Atlas CSV inputs. They connect the source modules together
the same way the evaluation scripts do, but keep everything deterministic.
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.capability_matcher import classify_trace
from src.evaluation.benchmarks.mcp_atlas import row_to_gap_traces
from src.evaluation.capabilities import tools_to_capabilities
from src.evaluation.metrics import evaluate_predictions
from src.schemas import AgentTrace, Prediction, ToolCall
from src.taxonomy import FailureType
from scripts.export_mcp_atlas_ablation_inputs import export_inputs


F6 = FailureType.MISSING_CAPABILITY_GAP.value
F1 = FailureType.REASONING_OR_PLANNING_ERROR.value


class CapabilityGapCoreTests(unittest.TestCase):
    def test_tools_to_capabilities_maps_known_tau_tools(self):
        capabilities = tools_to_capabilities(
            ["find_user_id_by_email", "find_user_id_by_name_zip", "think", "weather_api"]
        )

        self.assertEqual(capabilities, ["user_lookup", "weather_api"])

    def test_mcp_atlas_row_to_gap_traces_withholds_required_tool(self):
        row = {
            "TASK": "task_1",
            "PROMPT": "Use the private database to answer the question.",
            "ENABLED_TOOLS": json.dumps(["database_query", "calculator"]),
            "TRAJECTORY": json.dumps(
                [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "database_query", "arguments": "{}"},
                            }
                        ],
                    }
                ]
            ),
        }

        traces = row_to_gap_traces(row, max_per_task=1)

        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0].gold_label, F6)
        self.assertNotIn("database_query", traces[0].available_tools)
        self.assertEqual(traces[0].capabilities, ["calculator"])

    def test_evaluate_predictions_reports_binary_gap_f1(self):
        traces = [
            AgentTrace(
                trace_id="gap",
                user_task="Need weather.",
                available_tools=[],
                tool_calls=[],
                gold_label=F6,
            ),
            AgentTrace(
                trace_id="nongap",
                user_task="Bad plan.",
                available_tools=["calculator"],
                tool_calls=[],
                gold_label=F1,
            ),
        ]
        predictions = [
            Prediction(
                trace_id="gap",
                predicted_label=F6,
                confidence=1.0,
                evidence=["missing weather"],
                new_tool_needed=True,
            ),
            Prediction(
                trace_id="nongap",
                predicted_label=F1,
                confidence=1.0,
                evidence=["bad plan"],
                new_tool_needed=False,
            ),
        ]

        result = evaluate_predictions(
            traces,
            predictions,
            method="unit",
            split="all",
            n_train=2,
        )

        self.assertEqual(result.accuracy, 1.0)
        self.assertEqual(result.gap_detection_f1, 1.0)

    def test_capability_matcher_emits_request_for_missing_capability(self):
        trace = AgentTrace(
            trace_id="weather_gap",
            user_task="What is the current weather in Berlin?",
            available_tools=["calculator"],
            capabilities=["compute"],
            tool_calls=[
                ToolCall(
                    tool_name="calculator",
                    arguments={"expression": "weather"},
                    error="Calculator cannot retrieve weather information.",
                )
            ],
            final_response="I cannot complete this.",
            gold_label=F6,
        )

        def fake_complete(_messages):
            return json.dumps(
                {
                    "required_capabilities": [
                        {
                            "slug": "weather_api",
                            "description": "current weather lookup",
                            "evidence": "The task asks for current weather.",
                            "matched_available": None,
                            "capability_request": {
                                "name": "weather_lookup",
                                "capability": "weather_api",
                                "description": "Fetch current weather by city.",
                                "inputs": [{"name": "city", "type": "string"}],
                                "outputs": [{"name": "forecast", "type": "string"}],
                                "rationale": "Required to answer live weather questions.",
                            },
                        }
                    ],
                    "reasoning": "No available capability covers weather.",
                }
            )

        prediction = classify_trace(trace, complete=fake_complete)

        self.assertEqual(prediction.predicted_label, F6)
        self.assertTrue(prediction.new_tool_needed)
        self.assertEqual(prediction.missing_capabilities, ["weather_api"])
        self.assertEqual(prediction.capability_requests[0].name, "weather_lookup")

    def test_export_mcp_atlas_ablation_inputs_writes_baseline_and_ablated_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows_path = root / "rows.jsonl"
            plan_path = root / "plan.jsonl"
            out_dir = root / "runs"
            row = {
                "TASK": "task_1",
                "ENABLED_TOOLS": json.dumps(["tool_a", "tool_b"]),
                "PROMPT": "Do the task.",
                "GTFA_CLAIMS": "['done']",
                "TRAJECTORY": "[]",
            }
            plan = {
                "case_id": "case_1",
                "source_task_id": "task_1",
                "hidden_required_tools": ["tool_a"],
                "visible_tools_in_gap_run": ["tool_b"],
            }
            rows_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            plan_path.write_text(json.dumps(plan) + "\n", encoding="utf-8")

            manifest = export_inputs(
                source_rows_path=rows_path,
                plan_path=plan_path,
                output_dir=out_dir,
            )

            self.assertEqual(len(manifest), 1)
            self.assertTrue((out_dir / "baseline_all.csv").exists())
            self.assertTrue((out_dir / "ablated_all.csv").exists())
            ablated_text = (out_dir / "ablated_all.csv").read_text(encoding="utf-8")
            self.assertIn("tool_b", ablated_text)
            self.assertNotIn("tool_a\", \"tool_b", ablated_text)


if __name__ == "__main__":
    unittest.main()
