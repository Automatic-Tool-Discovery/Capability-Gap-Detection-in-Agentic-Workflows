"""Shared allowlisted classifier evidence, separate from experiment metadata."""

from typing import Any

from src.evaluation.capabilities import NON_CAPABILITY_TOOLS, normalize_tool_name, tools_to_capabilities
from src.schemas import AgentTrace


def available_capabilities(trace: AgentTrace) -> set[str]:
    caps = trace.capabilities or tools_to_capabilities(trace.available_tools)
    return {normalize_tool_name(c) for c in caps} - {
        normalize_tool_name(c) for c in NON_CAPABILITY_TOOLS
    }


def trace_evidence(trace: AgentTrace, *, use_failure_explanation: bool = False) -> dict[str, Any]:
    """Fair methods share full evidence; only explicit oracle mode adds rationale.

    Do not serialize the entire trace: IDs, provenance, scenario conditions,
    annotations and withheld-tool definitions are evaluation-only information.
    Natural task/observation text is preserved, including agent self-reports.
    """
    payload: dict[str, Any] = {
        "user_task": trace.user_task,
        "available_tools": trace.available_tools,
        "available_capabilities": sorted(available_capabilities(trace)),
        "tool_schemas": {
            name: schema for name, schema in trace.tool_schemas.items()
            if name in trace.available_tools
        },
        "agent_plan": trace.agent_plan,
        "tool_calls": [call.model_dump() for call in trace.tool_calls],
        "final_response": trace.final_response,
    }
    if use_failure_explanation and trace.failure_explanation:
        payload["failure_explanation"] = trace.failure_explanation
    return payload
