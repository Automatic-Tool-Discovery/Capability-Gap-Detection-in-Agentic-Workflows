from pydantic import BaseModel
from typing import Any, Optional


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    observation: Optional[str] = None
    error: Optional[str] = None


class Scenario(BaseModel):
    trace_id: str
    user_task: str
    agent_plan: Optional[str] = None
    mcp_servers: list[str]
    tool_calls: list[dict[str, Any]]
    final_response: Optional[str] = None
    gold_label: Optional[str] = None
    failure_explanation: Optional[str] = None


class AgentTrace(BaseModel):
    trace_id: str
    user_task: str
    available_tools: list[str]
    agent_plan: Optional[str] = None
    tool_calls: list[ToolCall]
    final_response: Optional[str] = None
    gold_label: Optional[str] = None
    failure_explanation: Optional[str] = None
    mcp_servers: list[str] = []
    tool_schemas: dict[str, Any] = {}
    # Alignment fields (shared across benchmarks).
    source: Optional[str] = None
    domain: Optional[str] = None
    capabilities: list[str] = []


class CapabilityRequest(BaseModel):
    """A structured spec for a missing tool that would close a capability gap."""

    name: str
    capability: str
    description: str
    inputs: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    rationale: Optional[str] = None


class Prediction(BaseModel):
    trace_id: str
    predicted_label: str
    confidence: float
    evidence: list[str]
    new_tool_needed: bool
    # Populated only when a capability gap (F6) is detected.
    missing_capabilities: list[str] = []
    capability_requests: list[CapabilityRequest] = []
