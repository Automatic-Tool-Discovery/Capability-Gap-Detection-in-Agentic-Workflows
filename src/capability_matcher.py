"""Capability matcher: detect capability gaps (F6) and emit capability requests.

This is the project's core contribution and the part that goes beyond AgentRx.

AgentRx (the baseline) treats "Intent Not Supported" as 1 of 9 categories chosen
by a black-box LLM judge, and stops at the label. We instead make the
capability-gap decision *structured and grounded*:

    1. An LLM extracts the capabilities the TASK requires (independent of what
       tools happen to exist), each mapped to a canonical slug.
    2. Deterministic set logic compares required vs. available capabilities
       (``required - available = missing``). The gap decision is therefore not a
       single opaque guess but a verifiable comparison ("probable tools vs.
       available tools").
    3. If something required is missing, we label the trace F6 *and emit a
       structured capability request* - the spec of the tool that would let the
       agent finish. AgentRx produces no such artifact.
    4. If nothing is missing, we defer to the LLM baseline for the fine-grained
       F0-F8 label, so this acts as a capability layer on top of the baseline.

The module is testable offline: pass a ``complete`` callable to avoid any network.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

from src import llm_classifier
from src.evaluation.capabilities import (
    NON_CAPABILITY_TOOLS,
    normalize_tool_name,
)
from src.schemas import AgentTrace, CapabilityRequest, Prediction
from src.taxonomy import FailureType

F6_LABEL = FailureType.MISSING_CAPABILITY_GAP.value

EXTRACTION_SYSTEM_PROMPT = """You analyze a tool-using AI agent's execution trace to find CAPABILITY GAPS.

A capability gap means: completing the user's task requires an ability that NONE of
the available tools/capabilities provide. It is NOT a reasoning mistake, a bad
parameter, a transient tool error, or missing information from the user.

You are given: the user task, the capabilities currently available, the agent's
tool calls, and its final response.

Do this:
1. List the capabilities the TASK requires, independent of what tools exist. For
   each: a short canonical slug (snake_case), a one-line description, and the
   concrete evidence from the task/trace.
2. For each required capability, decide whether an AVAILABLE capability covers it.
   If yes, set "matched_available" to that available slug. If no available
   capability can perform it, set "matched_available" to null.
3. For every capability with matched_available=null, also produce a
   "capability_request": the spec of the tool that WOULD close the gap, with
   name, capability slug, description, inputs[], outputs[], and a short rationale.

Be conservative: only mark a capability as missing (matched_available=null) if the
task genuinely cannot be completed with the available tools. If the agent completed
the task or had a valid tool for it, it is NOT missing.

Return JSON only:
{
  "required_capabilities": [
    {
      "slug": "string",
      "description": "string",
      "evidence": "string",
      "matched_available": "string-or-null",
      "capability_request": {
        "name": "string",
        "capability": "string",
        "description": "string",
        "inputs": [{"name": "string", "type": "string", "description": "string"}],
        "outputs": [{"name": "string", "type": "string", "description": "string"}],
        "rationale": "string"
      }
    }
  ],
  "reasoning": "string"
}
The "capability_request" field may be omitted (or null) when matched_available is set.
"""


def _available_capability_set(trace: AgentTrace) -> set[str]:
    """Normalized set of capabilities the trace actually has available."""
    caps: set[str] = set(trace.capabilities or [])
    # Fall back to deriving from tool names if capabilities weren't precomputed.
    if not caps and trace.available_tools:
        from src.evaluation.capabilities import tools_to_capabilities

        caps = set(tools_to_capabilities(trace.available_tools))
    normalized = {normalize_tool_name(c) for c in caps}
    normalized -= {normalize_tool_name(t) for t in NON_CAPABILITY_TOOLS}
    return normalized


def _matcher_payload(trace: AgentTrace, *, use_failure_explanation: bool) -> dict[str, Any]:
    available = sorted(_available_capability_set(trace))
    payload: dict[str, Any] = {
        "user_task": trace.user_task,
        "available_capabilities": available,
        "available_tools": trace.available_tools,
        "tool_calls": [
            {"tool_name": c.tool_name, "error": c.error} for c in trace.tool_calls
        ],
        "final_response": trace.final_response,
    }
    if use_failure_explanation and trace.failure_explanation:
        payload["failure_explanation"] = trace.failure_explanation
    return payload


def _parse_json_response(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    return json.loads(content)


def _is_covered(matched_available: Any, available: set[str]) -> bool:
    """True iff the LLM's claimed available capability actually exists."""
    if not matched_available or not isinstance(matched_available, str):
        return False
    return normalize_tool_name(matched_available) in available


def _to_capability_request(raw: Any, *, fallback_slug: str, task: str) -> CapabilityRequest:
    raw = raw if isinstance(raw, dict) else {}
    return CapabilityRequest(
        name=str(raw.get("name") or f"{fallback_slug}_tool"),
        capability=str(raw.get("capability") or fallback_slug),
        description=str(raw.get("description") or f"Provides the '{fallback_slug}' capability."),
        inputs=list(raw.get("inputs") or []),
        outputs=list(raw.get("outputs") or []),
        rationale=str(raw.get("rationale") or f"Required by task: {task[:160]}"),
    )


def detect_capability_gap(
    trace: AgentTrace,
    *,
    use_failure_explanation: bool = False,
    complete: Callable[[list[dict[str, str]]], str] | None = None,
) -> dict[str, Any]:
    """Run requirement extraction + deterministic matching.

    Returns a dict with: is_gap, missing (list of slugs), requests
    (list[CapabilityRequest]), evidence (list[str]).
    """
    complete_fn = complete or llm_classifier._default_client_factory()
    available = _available_capability_set(trace)

    user_prompt = (
        "Find capability gaps in this trace.\n\n"
        f"{json.dumps(_matcher_payload(trace, use_failure_explanation=use_failure_explanation), ensure_ascii=False, indent=2)}"
    )
    content = complete_fn(
        [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )
    data = _parse_json_response(content)

    missing: list[str] = []
    requests: list[CapabilityRequest] = []
    evidence: list[str] = []
    for item in data.get("required_capabilities", []) or []:
        if not isinstance(item, dict):
            continue
        slug = normalize_tool_name(str(item.get("slug", "")))
        if not slug:
            continue
        if _is_covered(item.get("matched_available"), available):
            continue
        # Double-check the slug itself isn't directly available.
        if slug in available:
            continue
        missing.append(slug)
        evidence.append(
            f"missing '{slug}': {item.get('evidence') or item.get('description') or ''}".strip()
        )
        requests.append(
            _to_capability_request(
                item.get("capability_request"),
                fallback_slug=slug,
                task=trace.user_task,
            )
        )

    return {
        "is_gap": bool(missing),
        "missing": missing,
        "requests": requests,
        "evidence": evidence,
        "reasoning": data.get("reasoning", ""),
    }


def classify_trace(
    trace: AgentTrace,
    *,
    use_failure_explanation: bool = False,
    complete: Callable[[list[dict[str, str]]], str] | None = None,
) -> Prediction:
    """Predict a label. F6 (+capability request) on a detected gap, else defer
    to the LLM baseline for the fine-grained F0-F8 label."""
    complete_fn = complete or llm_classifier._default_client_factory()
    gap = detect_capability_gap(
        trace,
        use_failure_explanation=use_failure_explanation,
        complete=complete_fn,
    )

    if gap["is_gap"]:
        return Prediction(
            trace_id=trace.trace_id,
            predicted_label=F6_LABEL,
            confidence=0.8,
            evidence=gap["evidence"][:5] or ["capability gap detected"],
            new_tool_needed=True,
            missing_capabilities=gap["missing"],
            capability_requests=gap["requests"],
        )

    # No gap: defer to the baseline classifier for the precise label.
    baseline = llm_classifier.classify_trace(
        trace,
        use_failure_explanation=use_failure_explanation,
        complete=complete_fn,
    )
    return baseline
