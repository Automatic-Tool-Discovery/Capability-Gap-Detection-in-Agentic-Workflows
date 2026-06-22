"""Core research tools exposed via MCP."""

from __future__ import annotations

import ast
import operator
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("research_tools")

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
BINARY_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".wav", ".gif"}

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _resolve_path(file_path: str) -> Path:
    path = Path(file_path)
    if not path.is_absolute():
        path = FIXTURES / file_path
    return path


def _safe_eval(expression: str) -> float:
    node = ast.parse(expression, mode="eval").body
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPERATORS:
        left = _safe_eval(ast.unparse(node.left))
        right = _safe_eval(ast.unparse(node.right))
        return SAFE_OPERATORS[type(node.op)](left, right)
    raise ValueError("Invalid mathematical expression.")


@mcp.tool()
def calculator(expression: str) -> str:
    """Performs basic arithmetic calculations."""
    if any(token in expression.lower() for token in ("weather", "csv", "count", "active")):
        raise ValueError("Calculator cannot retrieve weather information.")
    return str(_safe_eval(expression))


@mcp.tool()
def read_file(file_path: str) -> str:
    """Reads a plain text file from a given file path."""
    path = _resolve_path(file_path)
    if path.suffix.lower() in BINARY_EXTENSIONS:
        if path.suffix.lower() == ".pdf":
            raise ValueError("Cannot decode binary PDF as plain text.")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            raise ValueError("Cannot read image file as plain text.")
        raise ValueError("Unsupported file type.")
    if not path.exists():
        raise FileNotFoundError(f"{file_path} does not exist in the current directory.")
    return path.read_text(encoding="utf-8")


@mcp.tool()
def csv_reader(file_path: str) -> str:
    """Reads structured CSV files and returns rows and columns."""
    path = _resolve_path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{file_path} does not exist in the current directory.")
    return path.read_text(encoding="utf-8")


@mcp.tool()
def text_search(text: str, keyword: str) -> str:
    """Searches for a keyword in plain text."""
    if keyword.lower() not in text.lower():
        return "No matches found."
    return f"Found keyword '{keyword}' in text."


@mcp.tool()
def sql_query(query: str) -> str:
    """Executes SQL queries on a toy database."""
    normalized = query.lower()
    if "status='shipped'" in normalized.replace(" ", "") or "status = 'shipped'" in normalized:
        raise ValueError(
            "Invalid status value. Expected integer enum: 1=pending, 2=shipped, 3=delivered."
        )
    if "status='delivered'" in normalized.replace(" ", "") or "status = 'delivered'" in normalized:
        raise ValueError(
            "Invalid status value. Expected integer enum: 1=pending, 2=shipped, 3=delivered."
        )
    if "status=2" in normalized.replace(" ", ""):
        return "order_id=101,status=shipped"
    if "status=3" in normalized.replace(" ", ""):
        return "order_id=202,status=delivered"
    return "Query executed successfully."


@mcp.tool()
def web_search(query: str) -> str:
    """Searches the web for general information."""
    return f"Several webpages discussing: {query}"


@mcp.tool()
def run_python(code: str) -> str:
    """Runs a small Python code snippet."""
    lowered = code.lower()
    if "pdf" in lowered:
        raise ValueError("No PDF parsing library available.")
    return "Code executed successfully."


if __name__ == "__main__":
    mcp.run()
