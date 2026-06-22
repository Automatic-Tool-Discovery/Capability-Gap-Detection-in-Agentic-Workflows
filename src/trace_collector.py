"""Collect execution traces by running MCP scenarios against real MCP servers."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from src.schemas import AgentTrace, Scenario, ToolCall

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_PATH = PROJECT_ROOT / "data" / "mcp_scenarios.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "mcp_traces.jsonl"

SERVER_SCRIPTS = {
    "research_tools": PROJECT_ROOT / "mcp_servers" / "research_tools" / "server.py",
    "extended_tools": PROJECT_ROOT / "mcp_servers" / "extended_tools" / "server.py",
}


def load_scenarios(path: Path) -> list[Scenario]:
    scenarios = json.loads(path.read_text(encoding="utf-8"))
    return [Scenario.model_validate(item) for item in scenarios]


def _tool_text(result: types.CallToolResult) -> str | None:
    if not result.content:
        return None
    block = result.content[0]
    if isinstance(block, types.TextContent):
        return block.text
    return str(block)


def _tool_error(result: types.CallToolResult) -> str | None:
    if result.isError:
        return _tool_text(result) or "Tool call failed."
    return None


async def _connect_servers(
    stack: AsyncExitStack,
    server_names: list[str],
) -> tuple[dict[str, ClientSession], dict[str, str], dict[str, Any]]:
    sessions: dict[str, ClientSession] = {}
    tool_owners: dict[str, str] = {}
    tool_schemas: dict[str, Any] = {}

    for server_name in server_names:
        script_path = SERVER_SCRIPTS.get(server_name)
        if script_path is None:
            raise ValueError(f"Unknown MCP server: {server_name}")

        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(script_path)],
            cwd=str(PROJECT_ROOT),
        )
        read, write = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        sessions[server_name] = session

        tools_response = await session.list_tools()
        for tool in tools_response.tools:
            tool_owners[tool.name] = server_name
            tool_schemas[tool.name] = tool.model_dump()

    return sessions, tool_owners, tool_schemas


async def run_scenario(scenario: Scenario) -> AgentTrace:
    async with AsyncExitStack() as stack:
        sessions, tool_owners, tool_schemas = await _connect_servers(
            stack, scenario.mcp_servers
        )
        available_tools = sorted(tool_owners.keys())
        recorded_calls: list[ToolCall] = []

        for planned_call in scenario.tool_calls:
            tool_name = planned_call["tool_name"]
            arguments = planned_call.get("arguments", {})
            owner = tool_owners.get(tool_name)
            if owner is None:
                recorded_calls.append(
                    ToolCall(
                        tool_name=tool_name,
                        arguments=arguments,
                        observation=None,
                        error=f"Tool '{tool_name}' is not exposed by connected MCP servers.",
                    )
                )
                continue

            result = await sessions[owner].call_tool(tool_name, arguments=arguments)
            error = _tool_error(result)
            observation = None if error else _tool_text(result)

            recorded_calls.append(
                ToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                    observation=observation,
                    error=error,
                )
            )

        return AgentTrace(
            trace_id=scenario.trace_id,
            user_task=scenario.user_task,
            available_tools=available_tools,
            agent_plan=scenario.agent_plan,
            tool_calls=recorded_calls,
            final_response=scenario.final_response,
            gold_label=scenario.gold_label,
            failure_explanation=scenario.failure_explanation,
            mcp_servers=scenario.mcp_servers,
            tool_schemas=tool_schemas,
        )


async def collect_traces(
    scenarios_path: Path,
    output_path: Path,
) -> list[AgentTrace]:
    scenarios = load_scenarios(scenarios_path)
    traces: list[AgentTrace] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for scenario in scenarios:
            trace = await run_scenario(scenario)
            traces.append(trace)
            output_file.write(
                json.dumps(trace.model_dump(), ensure_ascii=False) + "\n"
            )
            print(f"{trace.trace_id}: collected {len(trace.tool_calls)} MCP tool calls")

    return traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect MCP execution traces.")
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=SCENARIOS_PATH,
        help="Path to MCP scenario definitions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write collected traces.",
    )
    args = parser.parse_args()
    asyncio.run(collect_traces(args.scenarios, args.output))


if __name__ == "__main__":
    main()
