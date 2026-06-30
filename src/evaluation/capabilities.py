"""Shared capability vocabulary across benchmarks (alignment, suggestion C).

Different benchmarks name their tools differently (tau-bench retail vs MCP-Atlas),
but capability-gap detection needs a *common* notion of "what ability is required"
and "what ability is available". This module maps raw tool names from any source
onto a canonical capability vocabulary, so "missing capability" means the same
thing regardless of which dataset a trace came from.

Design:
- A curated mapping for known domains (tau-bench retail) groups related tools into
  one capability (e.g. both user-lookup tools -> ``user_lookup``).
- Unknown tools fall back to a normalized version of their own name, so the
  vocabulary still works on MCP-Atlas and any future source without curation.
"""

from __future__ import annotations

import re

# Full tau-bench retail tool universe (the domain's complete toolset, not just
# the tools any single trace happened to call). Extracted from the AgentRx
# tau_retail trajectories.
TAU_RETAIL_TOOLS: list[str] = [
    "calculate",
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_order_details",
    "get_product_details",
    "get_user_details",
    "list_all_product_types",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_user_address",
    "return_delivered_order_items",
    "think",
    "transfer_to_human_agents",
]

# Curated tool -> canonical capability mapping. Tools that serve the same
# user-facing ability share a capability name.
TOOL_TO_CAPABILITY: dict[str, str] = {
    # tau-bench retail
    "find_user_id_by_email": "user_lookup",
    "find_user_id_by_name_zip": "user_lookup",
    "get_user_details": "user_info",
    "modify_user_address": "user_modify",
    "get_order_details": "order_info",
    "cancel_pending_order": "order_cancel",
    "modify_pending_order_items": "order_modify",
    "modify_pending_order_address": "order_modify",
    "exchange_delivered_order_items": "order_exchange",
    "return_delivered_order_items": "order_return",
    "get_product_details": "product_info",
    "list_all_product_types": "product_info",
    "calculate": "compute",
    "think": "reasoning",
    "transfer_to_human_agents": "escalate_to_human",
}

# Internal/agent-side tools that are not user-facing capabilities. These are
# excluded when computing the capabilities a task actually needs/provides.
NON_CAPABILITY_TOOLS: set[str] = {"think", "reasoning"}


def normalize_tool_name(name: str) -> str:
    """Lowercase, snake_case a raw tool name for stable matching."""
    key = name.strip().lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


def tool_to_capability(name: str) -> str:
    """Map a raw tool name onto the canonical capability vocabulary.

    Known tools use the curated mapping; unknown tools (e.g. MCP-Atlas) fall
    back to their normalized name so they still participate in matching.
    """
    normalized = normalize_tool_name(name)
    return TOOL_TO_CAPABILITY.get(normalized, normalized)


def tools_to_capabilities(tools: list[str], *, drop_internal: bool = True) -> list[str]:
    """Convert a list of raw tool names into a sorted set of capabilities."""
    capabilities: set[str] = set()
    for tool in tools:
        capability = tool_to_capability(tool)
        if drop_internal and capability in NON_CAPABILITY_TOOLS:
            continue
        capabilities.add(capability)
    return sorted(capabilities)
