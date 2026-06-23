"""LLM-based failure classifier using the TUD:AI OpenAI-compatible API."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

from src.schemas import AgentTrace, Prediction
from src.taxonomy import FailureType

DEFAULT_BASE_URL = "https://llm.scads.ai/v1"
DEFAULT_MODEL = "alias-ha"
VALID_LABELS = [failure.value for failure in FailureType]

SYSTEM_PROMPT = """You classify failures in tool-using AI agent execution traces.

Choose exactly one label from this taxonomy:
- F0_success_no_failure
- F1_reasoning_or_planning_error
- F2_wrong_tool_selected
- F3_wrong_tool_parameters
- F4_tool_runtime_error
- F5_tool_documentation_or_schema_error
- F6_missing_capability_gap
- F7_insufficient_user_information
- F8_environment_or_state_error

F6 means the task needs functionality that is not available in the connected toolset.
Return JSON only with keys:
- predicted_label
- confidence (0.0 to 1.0)
- evidence (array of short strings)
- new_tool_needed (boolean)
"""


def trace_payload(trace: AgentTrace, *, use_failure_explanation: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "trace_id": trace.trace_id,
        "user_task": trace.user_task,
        "available_tools": trace.available_tools,
        "agent_plan": trace.agent_plan,
        "tool_calls": [call.model_dump() for call in trace.tool_calls],
        "final_response": trace.final_response,
        "mcp_servers": trace.mcp_servers,
    }
    if use_failure_explanation and trace.failure_explanation:
        payload["failure_explanation"] = trace.failure_explanation
    if trace.tool_schemas:
        payload["tool_schemas"] = trace.tool_schemas
    return payload


def _parse_json_response(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    data = json.loads(content)
    if data["predicted_label"] not in VALID_LABELS:
        raise ValueError(f"Invalid label from LLM: {data['predicted_label']}")
    return data


def _default_client_factory() -> Callable[..., Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "Install the openai package to use the LLM baseline: pip install openai"
        ) from exc

    api_key = os.environ.get("SCADS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set SCADS_API_KEY (TUD:AI) or OPENAI_API_KEY before running the LLM baseline."
        )

    base_url = os.environ.get("SCADS_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("SCADS_MODEL", DEFAULT_MODEL)
    client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(messages: list[dict[str, str]]) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""

    return complete


def classify_trace(
    trace: AgentTrace,
    *,
    use_failure_explanation: bool = False,
    complete: Callable[[list[dict[str, str]]], str] | None = None,
) -> Prediction:
    """Classify a trace with an LLM. Defaults to fair eval (no gold explanation)."""
    complete_fn = complete or _default_client_factory()
    user_prompt = (
        "Classify this agent execution trace.\n\n"
        f"{json.dumps(trace_payload(trace, use_failure_explanation=use_failure_explanation), ensure_ascii=False, indent=2)}"
    )
    content = complete_fn(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )
    data = _parse_json_response(content)
    return Prediction(
        trace_id=trace.trace_id,
        predicted_label=data["predicted_label"],
        confidence=float(data.get("confidence", 0.5)),
        evidence=list(data.get("evidence", [])),
        new_tool_needed=bool(data.get("new_tool_needed", False)),
    )
