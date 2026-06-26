"""All research tools exposed via a single MCP server."""

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


@mcp.tool()
def summarizer(text: str) -> str:
    """Summarizes the provided text."""
    if not text.strip():
        raise ValueError("No text provided to summarize.")
    words = text.split()
    return " ".join(words[:8]) + ("..." if len(words) > 8 else "")


@mcp.tool()
def translate_text(text: str, target_language: str) -> str:
    """Translates text into the requested language."""
    if target_language.lower() == "german":
        return "Hallo, wie geht es dir?"
    return f"[{target_language}] {text}"


@mcp.tool()
def weather_api(city: str) -> str:
    """Returns current weather for a city."""
    return f"Weather in {city}: 18C, partly cloudy."


@mcp.tool()
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """Converts an amount between currencies."""
    if from_currency.upper() == "EUR" and to_currency.upper() == "INR":
        return f"{amount} EUR = {amount * 90:.2f} INR"
    return f"{amount} {from_currency} converted to {to_currency}"


@mcp.tool()
def send_email(recipient: str, subject: str = "", body: str = "") -> str:
    """Sends an email to the given recipient."""
    if not recipient:
        raise ValueError("Missing required field: recipient")
    raise RuntimeError("SMTP connection failed.")


@mcp.tool()
def create_pdf(title: str) -> str:
    """Creates a PDF document with the given title."""
    raise RuntimeError("ExecutionError: PDF backend crashed.")


@mcp.tool()
def book_restaurant(date: str = "", time: str = "", number_of_guests: int = 0) -> str:
    """Books a restaurant table."""
    if not date or not time or number_of_guests <= 0:
        raise ValueError("Missing required information: date, time, number of guests.")
    return f"Reserved table for {number_of_guests} on {date} at {time}."


@mcp.tool()
def create_calendar_event(
    title: str,
    date: str = "",
    time: str = "",
    attendee_email: str = "",
    authenticated: bool = False,
) -> str:
    """Creates a calendar event."""
    if not attendee_email or not time:
        raise ValueError("Missing attendee email and meeting time.")
    if not authenticated:
        raise PermissionError("AuthenticationError: user is not logged in to calendar service.")
    return f"Created event '{title}'."


@mcp.tool()
def search_emails(sender: str, days_back: int) -> str:
    """Search emails by sender. days_back is the number of days to look back."""
    if days_back < 3600:
        raise ValueError(
            "Invalid value for days_back. Expected Unix timestamp delta in seconds."
        )
    return f"Found emails from {sender}."


@mcp.tool()
def search_tickets(time_range: str = "", start_time: str = "", end_time: str = "") -> str:
    """Search customer tickets. Supports natural-language time ranges like 'past_24_hours'."""
    if time_range and not (start_time and end_time):
        raise ValueError(
            "Unsupported time_range format. Expected ISO-8601 start_time and end_time fields."
        )
    return "Found 3 matching tickets."


if __name__ == "__main__":
    mcp.run()
