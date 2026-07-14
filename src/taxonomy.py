"""Canonical failure taxonomy used by classifiers and benchmarks.

The project maps every local trace, live MCP run, AgentRx example, and MCP-Atlas
ablation into this single F0-F8 label space. ``F6_missing_capability_gap`` is the
research target: it means the task required a capability that no available tool
provided. The rest of the labels keep non-gap failures comparable to AgentRx-style
diagnosis.
"""

from enum import Enum


class FailureType(str, Enum):
    SUCCESS_NO_FAILURE = "F0_success_no_failure"
    REASONING_OR_PLANNING_ERROR = "F1_reasoning_or_planning_error"
    WRONG_TOOL_SELECTED = "F2_wrong_tool_selected"
    WRONG_TOOL_PARAMETERS = "F3_wrong_tool_parameters"
    TOOL_RUNTIME_ERROR = "F4_tool_runtime_error"
    TOOL_DOCUMENTATION_OR_SCHEMA_ERROR = "F5_tool_documentation_or_schema_error"
    MISSING_CAPABILITY_GAP = "F6_missing_capability_gap"
    INSUFFICIENT_USER_INFORMATION = "F7_insufficient_user_information"
    ENVIRONMENT_OR_STATE_ERROR = "F8_environment_or_state_error"
