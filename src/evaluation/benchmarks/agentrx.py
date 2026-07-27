"""AgentRx benchmark adapter.

AgentRx (Microsoft, https://github.com/microsoft/AgentRx) is the baseline paper:
it diagnoses *why* an agent failed from its execution trajectory, localizing the
critical failure step and labeling it with a 10-category taxonomy.

This adapter lets us evaluate our F0-F8 classifiers on AgentRx data, so we can
compare against the same trajectories the baseline uses.

Two data sources are supported:

1. ``samples`` - the public sample trajectories shipped in the GitHub repo
   (``trajectories/tau-retail/*.json``). These are ungated; the gold label is
   inferred from the filename. Good for development and a small demo.

2. ``hf`` - the full gated benchmark on Hugging Face
   (``microsoft/AgentRx``, 115 annotated failures). Requires accepting the
   dataset terms and authenticating (``huggingface-cli login``). Each row carries
   ``failures`` (step-level categories) and a ``root_cause`` (the critical one).

Trajectories use the standard tau-bench / OpenAI chat format:
    {
      "task_id": int, "reward": float, "trial": int,
      "info": {"task": {"instruction": str, "actions": [...], "user_id": str}},
      "traj": [ {role: system|user|assistant|tool, ...}, ... ]
    }
Tool failures surface as tool messages whose content starts with "Error:".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation.capabilities import TAU_RETAIL_TOOLS, tools_to_capabilities
from src.schemas import AgentTrace, ToolCall
from src.taxonomy import FailureType

SAMPLES_DIR = Path("data/benchmarks/agentrx_samples")

# --- Taxonomy mapping: AgentRx 10-category -> our F0-F8 ----------------------
# The taxonomies were designed independently, so this map is intentionally
# lossy. Notable points:
#   * AgentRx category 7 "Intent Not Supported" == our F6 (capability gap).
#   * AgentRx splits reasoning failures more finely (categories 1, 2, 4, 5),
#     which we collapse into F1.
#   * We have F0 (success) and F5 (doc/schema error); AgentRx has no direct
#     equivalent. AgentRx has category 10 "Inconclusive", which we treat as F1.
# This mismatch is itself a research finding worth reporting in the thesis.
AGENTRX_CATEGORY_TO_F: dict[str, str] = {
    "instruction_plan_adherence_failure": FailureType.REASONING_OR_PLANNING_ERROR.value,
    "instruction_adherence_failure": FailureType.REASONING_OR_PLANNING_ERROR.value,
    "invention_of_new_information": FailureType.REASONING_OR_PLANNING_ERROR.value,
    "invalid_invocation": FailureType.WRONG_TOOL_PARAMETERS.value,
    "misinterpretation_of_tool_output": FailureType.REASONING_OR_PLANNING_ERROR.value,
    "intent_plan_misalignment": FailureType.REASONING_OR_PLANNING_ERROR.value,
    "underspecified_user_intent": FailureType.INSUFFICIENT_USER_INFORMATION.value,
    "intent_not_supported": FailureType.MISSING_CAPABILITY_GAP.value,
    "guardrails_triggered": FailureType.ENVIRONMENT_OR_STATE_ERROR.value,
    "system_failure": FailureType.TOOL_RUNTIME_ERROR.value,
    "inconclusive": FailureType.REASONING_OR_PLANNING_ERROR.value,
}

# Per-domain Hugging Face file names: (annotations, trajectories, id_prefix).
HF_DOMAIN_FILES: dict[str, tuple[str, str, str]] = {
    "tau_retail": ("tau_retail.jsonl", "tau_retail_dataset.jsonl", "tau_retail_"),
    "magentic_one": ("magentic_one.jsonl", "magentic_dataset.jsonl", "magentic_one_"),
}

# Public GitHub sample filenames -> canonical AgentRx category key.
SAMPLE_FILENAME_TO_CATEGORY: dict[str, str] = {
    "hallucination_doubt": "invention_of_new_information",
    "instruction_adherence_failure": "instruction_plan_adherence_failure",
    "intent_plan_misalignment": "intent_plan_misalignment",
    "invalid_invocation": "invalid_invocation",
    "invention_new_info": "invention_of_new_information",
    "misinterpretation_tool_output": "misinterpretation_of_tool_output",
    "underspecified_intent": "underspecified_user_intent",
}


def normalize_category(raw: str) -> str:
    """Normalize a free-form AgentRx category string to a canonical key."""
    key = raw.strip().lower()
    for char in (" ", "-", "/", ".", ":"):
        key = key.replace(char, "_")
    while "__" in key:
        key = key.replace("__", "_")
    return key.strip("_")


def map_category_to_f(raw: str) -> str | None:
    """Map an AgentRx category to an F label, or None if unrecognized."""
    return AGENTRX_CATEGORY_TO_F.get(normalize_category(raw))


def _is_error(content: str) -> bool:
    stripped = content.strip().lower()
    return stripped.startswith("error") or "error:" in stripped[:30]


def _tool_results_by_id(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for message in messages:
        if message.get("role") == "tool" and message.get("tool_call_id"):
            results[message["tool_call_id"]] = message
    return results


def parse_trajectory(obj: dict[str, Any]) -> dict[str, Any]:
    """Extract task, tool calls, tools, and final response from a trajectory."""
    info = obj.get("info", {}) or {}
    task = info.get("task", {}) or {}
    messages: list[dict[str, Any]] = obj.get("traj", []) or []

    user_task = task.get("instruction") or ""
    results_by_id = _tool_results_by_id(messages)

    tool_calls: list[ToolCall] = []
    available: set[str] = set()
    final_response = ""

    for message in messages:
        role = message.get("role")
        if role == "assistant":
            content = (message.get("content") or "").strip()
            if content:
                final_response = content
            for call in message.get("tool_calls") or []:
                function = call.get("function", {}) or {}
                name = function.get("name", "unknown_tool")
                available.add(name)
                raw_args = function.get("arguments", "")
                try:
                    arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
                except (json.JSONDecodeError, TypeError, ValueError):
                    arguments = {"_raw": raw_args}
                result = results_by_id.get(call.get("id", ""), {})
                observation = (result.get("content") or "") or None
                error = observation if observation and _is_error(observation) else None
                tool_calls.append(
                    ToolCall(
                        tool_name=name,
                        arguments=arguments if isinstance(arguments, dict) else {"value": arguments},
                        observation=observation,
                        error=error,
                    )
                )
        elif role == "tool":
            if message.get("name"):
                available.add(message["name"])

    return {
        "user_task": user_task,
        "tool_calls": tool_calls,
        "available_tools": sorted(available),
        "final_response": final_response,
    }


def trajectory_to_trace(
    obj: dict[str, Any],
    *,
    gold_label: str | None,
    trace_id: str,
    failure_explanation: str | None = None,
) -> AgentTrace:
    parsed = parse_trajectory(obj)
    # Alignment: use the full retail toolset as the available tools, not just the
    # tools this trace happened to call, so capability-gap detection can ask
    # "was the needed tool available at all?".
    available_tools = sorted(set(TAU_RETAIL_TOOLS) | set(parsed["available_tools"]))
    return AgentTrace(
        trace_id=trace_id,
        user_task=parsed["user_task"],
        available_tools=available_tools,
        agent_plan=None,
        tool_calls=parsed["tool_calls"],
        final_response=parsed["final_response"],
        gold_label=gold_label,
        failure_explanation=failure_explanation,
        mcp_servers=["agentrx-tau-retail"],
        tool_schemas={},
        source="agentrx",
        domain="tau_retail",
        capabilities=tools_to_capabilities(available_tools),
    )


def load_sample_traces(samples_dir: Path = SAMPLES_DIR) -> list[AgentTrace]:
    """Load the public GitHub sample trajectories (label inferred from filename)."""
    tau_dir = samples_dir / "tau-retail"
    if not tau_dir.exists():
        raise FileNotFoundError(
            f"AgentRx sample trajectories not found at {tau_dir}. "
            "Download them from https://github.com/microsoft/AgentRx/tree/main/trajectories/tau-retail"
        )

    traces: list[AgentTrace] = []
    for path in sorted(tau_dir.glob("*.json")):
        category = SAMPLE_FILENAME_TO_CATEGORY.get(path.stem)
        gold_label = map_category_to_f(category) if category else None
        with path.open("r", encoding="utf-8") as handle:
            obj = json.load(handle)
        traces.append(
            trajectory_to_trace(
                obj,
                gold_label=gold_label,
                trace_id=f"agentrx_tau_{path.stem}",
            )
        )
    return traces


def _root_cause_category(annotation: dict[str, Any]) -> str | None:
    """Find the category of the root-cause failure in a gated AgentRx annotation."""
    failures = annotation.get("failures") or []
    root_id = annotation.get("root_cause_failure_id")
    if root_id is None:
        root_id = (annotation.get("root_cause") or {}).get("failure_id")
    if root_id is not None:
        for failure in failures:
            if str(failure.get("failure_id")) == str(root_id):
                return failure.get("failure_category")
    if failures:
        return failures[0].get("failure_category")
    return None


def _iter_substeps(steps: list[dict[str, Any]]):
    """Flatten the steps/substeps structure into ordered (role, content) messages."""
    for step in steps:
        for substep in step.get("substeps", []) or []:
            yield substep.get("role"), (substep.get("content") or "")


def _parse_tool_call_content(content: str) -> list[dict[str, Any]] | None:
    """Parse an assistant message body that encodes a list of tool calls."""
    stripped = content.strip()
    if not stripped.startswith("["):
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and "function" in parsed[0]:
        return parsed
    return None


def parse_steps_trajectory(instruction: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Parse the HF ``steps`` trajectory format into trace components.

    Tool results carry no call id, so calls and results are paired in FIFO order.
    """
    pending: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []
    available: set[str] = set()
    final_response = ""

    for role, content in _iter_substeps(steps):
        if role == "assistant":
            calls = _parse_tool_call_content(content)
            if calls is None:
                text = content.strip()
                if text:
                    final_response = text
                continue
            for call in calls:
                function = call.get("function", {}) or {}
                name = function.get("name", "unknown_tool")
                available.add(name)
                raw_args = function.get("arguments", "")
                try:
                    arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
                except (json.JSONDecodeError, TypeError, ValueError):
                    arguments = {"_raw": raw_args}
                if not isinstance(arguments, dict):
                    arguments = {"value": arguments}
                pending.append({"name": name, "arguments": arguments, "observation": None, "error": None})
        elif role == "tool":
            observation = content or None
            error = observation if observation and _is_error(observation) else None
            if pending:
                call = pending.pop(0)
                call["observation"] = observation
                call["error"] = error
                finished.append(call)
            else:
                finished.append(
                    {"name": "unknown_tool", "arguments": {}, "observation": observation, "error": error}
                )

    finished.extend(pending)
    tool_calls = [
        ToolCall(
            tool_name=call["name"],
            arguments=call["arguments"],
            observation=call["observation"],
            error=call["error"],
        )
        for call in finished
    ]
    return {
        "user_task": instruction,
        "tool_calls": tool_calls,
        "available_tools": sorted(available),
        "final_response": final_response,
    }


def load_hf_traces(
    *,
    domain: str = "tau_retail",
    limit: int | None = None,
    use_failure_explanation: bool = False,
) -> list[AgentTrace]:
    """Load the full gated AgentRx benchmark from Hugging Face.

    Joins the annotation file (failure categories + root cause) with the
    trajectory file (instruction + steps) by trajectory id.

    Requires accepting the dataset terms at
    https://huggingface.co/datasets/microsoft/AgentRx and authenticating via
    ``huggingface-cli login`` (or setting HF_TOKEN).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Install huggingface_hub to load the gated AgentRx benchmark: "
            "uv add huggingface_hub"
        ) from exc

    if domain not in HF_DOMAIN_FILES:
        raise ValueError(
            f"Unknown AgentRx domain '{domain}'. Options: {', '.join(HF_DOMAIN_FILES)}"
        )
    annotation_file, trajectory_file, id_prefix = HF_DOMAIN_FILES[domain]

    annotation_path = hf_hub_download("microsoft/AgentRx", annotation_file, repo_type="dataset")
    trajectory_path = hf_hub_download("microsoft/AgentRx", trajectory_file, repo_type="dataset")

    def _read_jsonl(path: str) -> list[dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    annotations = _read_jsonl(annotation_path)
    trajectories: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(trajectory_path):
        raw_id = str(row.get("trajectory_id", ""))
        key = raw_id[len(id_prefix):] if raw_id.startswith(id_prefix) else raw_id
        trajectories[key] = row

    if limit is not None:
        annotations = annotations[:limit]

    traces: list[AgentTrace] = []
    for annotation in annotations:
        traj_id = str(annotation.get("trajectory_id", ""))
        trajectory = trajectories.get(traj_id)
        if trajectory is None:
            continue
        category = _root_cause_category(annotation)
        gold_label = map_category_to_f(category) if category else None
        explanation = None
        if use_failure_explanation:
            explanation = annotation.get("root_cause_reason") or annotation.get("failure_summary")

        parsed = parse_steps_trajectory(
            trajectory.get("instruction", ""),
            trajectory.get("steps", []) or [],
        )
        # Alignment: for tau_retail use the full domain toolset; otherwise fall
        # back to the observed tools.
        if domain == "tau_retail":
            available_tools = sorted(set(TAU_RETAIL_TOOLS) | set(parsed["available_tools"]))
        else:
            available_tools = parsed["available_tools"]
        traces.append(
            AgentTrace(
                trace_id=f"agentrx_{domain}_{traj_id}",
                user_task=parsed["user_task"],
                available_tools=available_tools,
                agent_plan=None,
                tool_calls=parsed["tool_calls"],
                final_response=parsed["final_response"],
                gold_label=gold_label,
                failure_explanation=explanation,
                mcp_servers=[f"agentrx-{domain}"],
                tool_schemas={},
                source="agentrx",
                domain=domain,
                capabilities=tools_to_capabilities(available_tools),
            )
        )
    return traces


def build_agentrx_dataset(
    *,
    source: str = "samples",
    domain: str = "tau_retail",
    limit: int | None = None,
    use_failure_explanation: bool = False,
) -> list[AgentTrace]:
    if source == "samples":
        return load_sample_traces()
    if source == "hf":
        return load_hf_traces(
            domain=domain,
            limit=limit,
            use_failure_explanation=use_failure_explanation,
        )
    raise ValueError(f"Unknown AgentRx source '{source}'. Use 'samples' or 'hf'.")
