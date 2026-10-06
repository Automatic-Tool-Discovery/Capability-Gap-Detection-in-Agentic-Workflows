"""Shared data models for the capability-gap project.

This file defines the typed objects that move through the whole pipeline:
raw tool calls become ``AgentTrace`` records, classifiers return ``Prediction``
objects, and capability-gap detections can include a ``CapabilityRequest`` that
describes the missing tool. Most other modules import these models, so this file
is the contract between live trace generation, benchmark adapters, evaluation,
and the capability matcher.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ExperimentCondition(BaseModel):
    """Author's scenario intent, never ground truth or classifier evidence."""

    task_id: str
    pair_id: str
    condition: Literal["control", "gap", "custom"]
    withheld_tools: list[str] = Field(default_factory=list)
    intended_missing_capabilities: list[str] = Field(default_factory=list)
    expected_label: Optional[str] = None


class TraceAnnotation(BaseModel):
    """Review provenance; legacy means verification has not been established."""

    status: Literal["legacy", "unreviewed", "unresolved", "verified"] = "legacy"
    outcome: Literal["unknown", "success", "failure"] = "unknown"
    reviewer: Optional[str] = None
    evidence: list[str] = Field(default_factory=list)
    method: Optional[str] = None


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
    schema_version: int = 1
    experiment: Optional[ExperimentCondition] = None
    annotation: TraceAnnotation = Field(default_factory=TraceAnnotation)
    inventory_complete: Optional[bool] = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    # Evaluation-only ground truth. Classifiers must not include these fields in
    # their prompts. Withholding alone must not populate these fields.
    gold_missing_capabilities: list[str] = []
    gold_capability_requests: list["CapabilityRequest"] = []

    @model_validator(mode="after")
    def validate_annotation(self):
        if self.annotation.status in {"unreviewed", "unresolved"} and (
            self.gold_label is not None or self.gold_missing_capabilities or self.gold_capability_requests
        ):
            raise ValueError("Unreviewed or unresolved traces cannot carry gold annotations.")
        if self.annotation.status == "verified" and (
            not self.gold_label or not self.annotation.evidence or not self.annotation.reviewer
            or not self.annotation.method or self.annotation.outcome == "unknown"
        ):
            raise ValueError("Verified annotations require a label, outcome, reviewer, method and evidence.")
        return self


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
