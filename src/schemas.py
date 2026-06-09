from pydantic import BaseModel
from typing import Any, Optional


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    observation: Optional[str] = None
    error: Optional[str] = None


class AgentTrace(BaseModel):
    trace_id: str
    user_task: str
    available_tools: list[str]
    agent_plan: Optional[str] = None
    tool_calls: list[ToolCall]
    final_response: Optional[str] = None
    gold_label: Optional[str] = None
    failure_explanation: Optional[str] = None


class Prediction(BaseModel):
    trace_id: str
    predicted_label: str
    confidence: float
    evidence: list[str]
    new_tool_needed: bool
