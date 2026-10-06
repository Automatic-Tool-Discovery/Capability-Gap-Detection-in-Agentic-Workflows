"""Offline validity checks independent of classifier performance."""
import json
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

from mcp import types

from src.capability_matcher import _matcher_payload
from src.llm_classifier import trace_payload
from src.live_agent import run_task
from src.evaluate import run_evaluation
from src.evaluation.splits import label_clean_controls_as_success
from src.schemas import AgentTrace, ExperimentCondition, TraceAnnotation, ToolCall


class EvidenceTests(unittest.TestCase):
    def test_metadata_changes_do_not_change_fair_evidence(self):
        trace = AgentTrace(
            trace_id="secret_gap", user_task="Read the temperature.",
            available_tools=["weather"],
            tool_schemas={"weather": {"description": "Temperature lookup"},
                          "hidden": {"description": "SECRET"}},
            tool_calls=[ToolCall(tool_name="weather", arguments={"city": "Berlin"},
                                 observation="18C")], final_response="18C",
            gold_label="F6_missing_capability_gap", failure_explanation="SECRET",
            gold_missing_capabilities=["SECRET"], mcp_servers=["SECRET"],
            provenance={"source": "SECRET"},
            experiment=ExperimentCondition(task_id="SECRET", pair_id="SECRET", condition="gap"),
            annotation=TraceAnnotation(status="verified", evidence=["SECRET"],
                                       reviewer="SECRET", method="review", outcome="failure"),
        )
        payload = trace_payload(trace, use_failure_explanation=False)
        self.assertEqual(payload, _matcher_payload(trace, use_failure_explanation=False))
        self.assertNotIn("SECRET", json.dumps(payload))
        self.assertNotIn("secret_gap", json.dumps(payload))
        self.assertEqual(payload["tool_calls"][0]["arguments"], {"city": "Berlin"})
        self.assertEqual(payload["tool_calls"][0]["observation"], "18C")
        other = trace.model_copy(update={"trace_id": "control", "gold_label": None,
                                        "experiment": None, "provenance": {}})
        self.assertEqual(payload, trace_payload(other, use_failure_explanation=False))
        oracle = trace_payload(trace, use_failure_explanation=True)
        self.assertEqual(oracle.pop("failure_explanation"), "SECRET")
        self.assertEqual(oracle, payload)

    def test_annotation_requires_review_provenance_and_unresolved_has_no_gold(self):
        base = dict(trace_id="t", user_task="Task", available_tools=[], tool_calls=[],
                    gold_label="F6_missing_capability_gap")
        for status in ["unreviewed", "unresolved", "verified"]:
            with self.subTest(status=status), self.assertRaises(ValueError):
                AgentTrace(**base, annotation=TraceAnnotation(status=status))

    def test_wrong_answer_with_successful_call_cannot_be_auto_labeled(self):
        trace = AgentTrace(trace_id="control", source="mcp-live", user_task="Temperature?",
                           available_tools=["weather"], final_response="99C",
                           tool_calls=[ToolCall(tool_name="weather", arguments={}, observation="18C")])
        with self.assertRaisesRegex(ValueError, "Automatic F0 labeling is disabled"):
            label_clean_controls_as_success([trace])
        self.assertIsNone(trace.gold_label)

    def test_unlabeled_evaluation_stops_before_model_calls(self):
        trace = AgentTrace(trace_id="new", user_task="Task", available_tools=[], tool_calls=[])
        with self.assertRaisesRegex(ValueError, "review outcomes"):
            run_evaluation([trace], method="llm-fair", split_name="all",
                           train_traces=[], test_traces=[trace])


class LiveValidityTests(unittest.IsolatedAsyncioTestCase):
    async def test_hidden_tool_is_rejected_and_valid_call_still_executes(self):
        calls = [NS(id=name, function=NS(name=name, arguments='{}'))
                 for name in ["hidden", "visible"]]
        client = NS(chat=NS(completions=NS(create=Mock(side_effect=[
            NS(choices=[NS(message=NS(tool_calls=calls, content=""))]),
            NS(choices=[NS(message=NS(tool_calls=[], content="Completed using the alternative."))]),
        ]))))
        session = NS(call_tool=AsyncMock(return_value=types.CallToolResult(
            content=[types.TextContent(type="text", text="done")], isError=False)))
        tools = [types.Tool(name=name, inputSchema={"type": "object"})
                 for name in ["hidden", "visible"]]
        trace = await run_task(session, tools, task={"task_id": "t", "user_task": "Task"},
                               withhold={"hidden"}, model="fake", client=client,
                               gold_missing_tools={"hidden"}, expected_label="F6_missing_capability_gap",
                               pair_id="pair")
        session.call_tool.assert_awaited_once_with("visible", arguments={})
        self.assertIn("not in the enabled tool inventory", trace.tool_calls[0].error)
        self.assertEqual(trace.tool_calls[1].observation, "done")
        self.assertEqual(trace.experiment.withheld_tools, ["hidden"])
        self.assertEqual(trace.experiment.pair_id, "pair")
        self.assertEqual(trace.experiment.intended_missing_capabilities, ["hidden"])
        self.assertEqual(trace.annotation.status, "unreviewed")
        self.assertTrue(trace.inventory_complete)
        self.assertIsNone(trace.gold_label)
        self.assertIsNone(trace.failure_explanation)
        self.assertEqual(trace.gold_missing_capabilities, [])
        self.assertEqual(trace.gold_capability_requests, [])
        self.assertEqual(AgentTrace.model_validate_json(trace.model_dump_json()), trace)

    async def test_control_expected_failure_is_only_scenario_intent(self):
        client = NS(chat=NS(completions=NS(create=Mock(return_value=NS(
            choices=[NS(message=NS(tool_calls=[], content="Answer"))])))))
        trace = await run_task(NS(), [], task={"task_id": "t", "user_task": "Task"},
                               withhold=set(), model="fake", client=client,
                               expected_label="F3_wrong_tool_parameters")
        self.assertIsNone(trace.gold_label)
        self.assertEqual(trace.experiment.condition, "control")
        self.assertEqual(trace.experiment.expected_label, "F3_wrong_tool_parameters")

    async def test_unknown_withheld_tool_fails_before_model_call(self):
        client = Mock()
        with self.assertRaisesRegex(ValueError, "Unknown withheld tools"):
            await run_task(NS(), [], task={"task_id": "t", "user_task": "Task"},
                           withhold={"typo"}, model="fake", client=client)
        self.assertEqual(client.mock_calls, [])


if __name__ == "__main__":
    unittest.main()
