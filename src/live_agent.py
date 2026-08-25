"""Dynamic trace generation: a live LLM agent acting against real MCP tools.

A real LLM decides which tools to call, calls them on the live MCP server,
observes the real results, and we record whatever actually happens.

Capability gaps (F6) are produced *dynamically* and with ground truth: we run a
task while **withholding** the tool it needs, so the agent genuinely gets stuck.
Because we chose which tool to withhold, we know the exact missing capability.

Requires a TUD:AI model with tool/function-calling support (e.g.
``alias-huge-no-thinking``). Set SCADS_API_KEY (and optionally SCADS_MODEL).

Usage:
    python -m src.live_agent                 # control + gap runs for every task
    python -m src.live_agent --mode gap       # only the capability-gap runs
    python -m src.live_agent --mode control   # only full-toolset runs
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from src.evaluation.capabilities import tools_to_capabilities
from src.model_config import get_model_config
from src.schemas import AgentTrace, CapabilityRequest, ToolCall
from src.taxonomy import FailureType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_SCRIPT = PROJECT_ROOT / "mcp_servers" / "research_tools" / "server.py"
TASKS_PATH = PROJECT_ROOT / "data" / "live_tasks.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "live_traces.jsonl"

DEFAULT_MODEL = "alias-huge-no-thinking"
MAX_STEPS = 6
REALTIME_TOOLS = {
    "open_library_search",
    "public_holidays",
    "realtime_earthquakes",
    "realtime_exchange_rate",
    "realtime_iss_position",
    "realtime_weather",
}
GENERIC_EXTERNAL_SUBSTITUTES = {"run_python", "web_search"}
REALTIME_SUBSTITUTES = {
    "realtime_weather": {"weather_api"},
    "realtime_exchange_rate": {"currency_converter"},
    "realtime_earthquakes": set(),
    "realtime_iss_position": set(),
    "public_holidays": set(),
    "open_library_search": set(),
}

SYSTEM_PROMPT = (
    "You are an assistant that completes tasks using the provided tools. "
    "Call tools to get real results. If none of the available tools can perform "
    "the requested task, do not guess or fabricate an answer - clearly state that "
    "you cannot complete the task because the required capability is unavailable."
)


def _tool_text(result: types.CallToolResult) -> str | None:
    if not result.content:
        return None
    block = result.content[0]
    if isinstance(block, types.TextContent):
        return block.text
    return str(block)


def _mcp_tools_to_openai(
    tools: list[types.Tool],
    withhold: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Convert MCP tool definitions to OpenAI function-calling schema."""
    openai_tools: list[dict[str, Any]] = []
    offered: list[str] = []
    for tool in tools:
        if tool.name in withhold:
            continue
        offered.append(tool.name)
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                },
            }
        )
    return openai_tools, sorted(offered)


def _make_client():
    from openai import OpenAI

    config = get_model_config(default_model=DEFAULT_MODEL)
    return OpenAI(api_key=config.api_key, base_url=config.base_url)


async def run_task(
    session: ClientSession,
    mcp_tools: list[types.Tool],
    *,
    task: dict[str, Any],
    withhold: set[str],
    model: str,
    client: Any,
    gold_missing_tools: set[str] | None = None,
) -> AgentTrace:
    openai_tools, offered = _mcp_tools_to_openai(mcp_tools, withhold)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["user_task"]},
    ]
    recorded_calls: list[ToolCall] = []
    final_response = ""

    for _ in range(MAX_STEPS):
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
            temperature=0,
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if not tool_calls:
            final_response = message.content or ""
            break

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ],
            }
        )

        for call in tool_calls:
            name = call.function.name
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            try:
                result = await session.call_tool(name, arguments=arguments)
                error = _tool_text(result) if result.isError else None
                observation = None if error else _tool_text(result)
            except Exception as exc:  # tool not available / call failed
                error = f"Tool '{name}' could not be called: {exc}"
                observation = None

            recorded_calls.append(
                ToolCall(
                    tool_name=name,
                    arguments=arguments,
                    observation=observation,
                    error=error,
                )
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": name,
                    "content": error or observation or "",
                }
            )

    is_gap = bool(gold_missing_tools)
    gold_label = FailureType.MISSING_CAPABILITY_GAP.value if is_gap else None
    failure_explanation = (
        f"Required tool(s) {sorted(gold_missing_tools or set())} were withheld from the agent, "
        "so the task cannot be completed with the available tools."
        if is_gap
        else None
    )
    suffix = "gap" if is_gap else ("custom" if withhold else "control")
    schemas_by_name = {tool.name: tool for tool in mcp_tools}
    gold_requests = []
    for tool_name in sorted(gold_missing_tools or set()):
        tool = schemas_by_name.get(tool_name)
        if tool is None:
            continue
        input_schema = tool.inputSchema or {}
        output_schema = tool.outputSchema or {}
        gold_requests.append(
            CapabilityRequest(
                name=tool.name,
                capability=tools_to_capabilities([tool.name])[0],
                description=tool.description or f"Provides the {tool.name} capability.",
                inputs=[
                    {
                        "name": name,
                        "type": details.get("type", "any"),
                        "description": details.get("description", ""),
                        "required": name in input_schema.get("required", []),
                    }
                    for name, details in input_schema.get("properties", {}).items()
                ],
                outputs=[
                    {
                        "name": name,
                        "type": details.get("type", "any"),
                        "description": details.get("description", ""),
                    }
                    for name, details in output_schema.get("properties", {}).items()
                ],
                rationale=f"This withheld tool is required to complete: {task['user_task']}",
            )
        )

    return AgentTrace(
        trace_id=f"live_{task['task_id']}_{suffix}",
        user_task=task["user_task"],
        available_tools=offered,
        agent_plan=None,
        tool_calls=recorded_calls,
        final_response=final_response,
        gold_label=gold_label,
        failure_explanation=failure_explanation,
        mcp_servers=["research_tools"],
        tool_schemas={tool.name: tool.model_dump() for tool in mcp_tools if tool.name in offered},
        source="mcp-live",
        domain="mcp_research_tools",
        capabilities=tools_to_capabilities(offered),
        gold_missing_capabilities=tools_to_capabilities(
            sorted(gold_missing_tools or set())
        ),
        gold_capability_requests=gold_requests,
    )


async def run_single_question(
    question: str,
    *,
    available_tools: set[str] | None = None,
    model: str = DEFAULT_MODEL,
) -> AgentTrace:
    """Run one ad-hoc question. Gold labels stay unset because need is unknown."""
    client = _make_client()
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
        cwd=str(PROJECT_ROOT),
    )
    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        mcp_tools = (await session.list_tools()).tools
        all_tools = {tool.name for tool in mcp_tools}
        selected = all_tools if available_tools is None else available_tools
        unknown = selected - all_tools
        if unknown:
            raise ValueError(f"Unknown available tool(s): {', '.join(sorted(unknown))}")
        return await run_task(
            session,
            mcp_tools,
            task={"task_id": "single_question", "user_task": question},
            withhold=all_tools - selected,
            model=model,
            client=client,
            gold_missing_tools=None,
        )


async def collect_live_traces(
    tasks_path: Path,
    output_path: Path,
    *,
    mode: str,
    model: str,
    append: bool = False,
) -> list[AgentTrace]:
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    client = _make_client()
    traces: list[AgentTrace] = []

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
        cwd=str(PROJECT_ROOT),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        mcp_tools = (await session.list_tools()).tools

        file_mode = "a" if append else "w"
        with output_path.open(file_mode, encoding="utf-8") as output_file:
            for task in tasks:
                runs: list[set[str]] = []
                if mode in {"control", "both"}:
                    runs.append(set())
                if mode in {"gap", "both"}:
                    # A capability gap requires withholding *all* tools that can
                    # provide the needed capability, not just one - otherwise the
                    # agent substitutes (e.g. run_python instead of calculator).
                    required = task.get("required_tools")
                    if not required:
                        single = task.get("required_tool")
                        required = [single] if single else []
                    if required:
                        withhold = set(task.get("withhold_tools") or required)
                        if set(required) & REALTIME_TOOLS:
                            withhold |= GENERIC_EXTERNAL_SUBSTITUTES
                            for tool_name in required:
                                withhold |= REALTIME_SUBSTITUTES.get(tool_name, set())
                        runs.append(withhold)

                for withhold in runs:
                    trace = await run_task(
                        session,
                        mcp_tools,
                        task=task,
                        withhold=withhold,
                        model=model,
                        client=client,
                        gold_missing_tools=set(required) if withhold else None,
                    )
                    traces.append(trace)
                    output_file.write(json.dumps(trace.model_dump(), ensure_ascii=False) + "\n")
                    n_err = sum(1 for call in trace.tool_calls if call.error)
                    print(
                        f"{trace.trace_id}: {len(trace.tool_calls)} calls, "
                        f"{n_err} errors, gold={trace.gold_label}"
                    )

    return traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dynamic agent traces via live MCP tool use.")
    parser.add_argument("--tasks", type=Path, default=TASKS_PATH, help="Live task definitions.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output traces file.")
    parser.add_argument(
        "--mode",
        choices=["both", "control", "gap"],
        default="both",
        help="control = full toolset; gap = withhold required tool (ground-truth F6); both.",
    )
    parser.add_argument(
        "--model",
        default=get_model_config(default_model=DEFAULT_MODEL).model,
        help="TUD:AI model with tool-calling support.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append generated traces to the output file instead of overwriting it.",
    )
    args = parser.parse_args()
    asyncio.run(
        collect_live_traces(
            args.tasks,
            args.output,
            mode=args.mode,
            model=args.model,
            append=args.append,
        )
    )


if __name__ == "__main__":
    main()
