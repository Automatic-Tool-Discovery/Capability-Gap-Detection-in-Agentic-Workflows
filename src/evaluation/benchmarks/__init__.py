"""External benchmark adapters.

Each adapter converts a benchmark-specific format into the project's shared
``AgentTrace`` schema. ``agentrx.py`` imports AgentRx-style failure trajectories,
while ``mcp_atlas.py`` derives missing-tool capability-gap examples from
MCP-Atlas rows.
"""
