"""Smoke-test whether the configured model can call MCP tools.

Run before generating a full dataset:

    python scripts/smoke_model_tool_calling.py

Configure with MODEL_PROVIDER / MODEL_NAME plus the provider's API key.
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from openai import OpenAI

from src.live_agent import _mcp_tools_to_openai
from src.model_config import get_model_config

SERVER_SCRIPT = PROJECT_ROOT / "mcp_servers" / "research_tools" / "server.py"


async def main() -> None:
    config = get_model_config(default_model="gpt-4.1")
    client = OpenAI(api_key=config.api_key, base_url=config.base_url)
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
        openai_tools, _ = _mcp_tools_to_openai(mcp_tools, set())

        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {
                    "role": "system",
                    "content": "Use the provided tool. Do not answer from memory.",
                },
                {
                    "role": "user",
                    "content": "Get the current weather in Berlin from a live weather source.",
                },
            ],
            tools=openai_tools,
            tool_choice="auto",
            temperature=0,
        )
        message = response.choices[0].message
        calls = message.tool_calls or []
        print(f"provider={config.provider} model={config.model} tool_calls={len(calls)}")
        if not calls:
            print(f"final_response={message.content!r}")
            raise SystemExit(1)

        call = calls[0]
        arguments: dict[str, Any]
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}
        print(f"called={call.function.name} arguments={arguments}")
        result = await session.call_tool(call.function.name, arguments=arguments)
        print(f"is_error={result.isError}")
        if result.content:
            print(result.content[0])


if __name__ == "__main__":
    asyncio.run(main())
