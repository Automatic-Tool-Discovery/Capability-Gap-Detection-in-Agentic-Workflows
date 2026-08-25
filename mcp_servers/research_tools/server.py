"""Toy research-tool MCP server used to generate live capability-gap traces.

This server exposes a single collection of tools to ``src.live_agent`` through
MCP stdio. Some tools succeed, some deliberately fail for specific inputs, and
some represent external capabilities such as weather, currency conversion,
email, calendar, or ticket search. Live experiments run tasks twice: once with
the full toolset and once with required tools withheld, which creates
ground-truth F6 missing-capability examples for the rest of the project.
"""

from __future__ import annotations

import ast
import json
import operator
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("research_tools")

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
BINARY_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".wav", ".gif"}
CITY_COORDS = {
    "berlin": (52.52, 13.405),
    "dresden": (51.0504, 13.7373),
    "london": (51.5072, -0.1276),
    "new york": (40.7128, -74.0060),
    "san francisco": (37.7749, -122.4194),
    "tokyo": (35.6762, 139.6503),
}

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


def _get_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "capability-gap-research/0.1"})
    with urllib.request.urlopen(request, timeout=20) as response:
        body = response.read().decode("utf-8").strip()
        if not body:
            raise ValueError(f"Empty response from {url}")
        return json.loads(body)


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
    if from_currency.upper() == "USD" and to_currency.upper() == "EUR":
        return f"{amount} USD = {amount * 0.92:.2f} EUR"
    if from_currency.upper() == "EUR" and to_currency.upper() == "INR":
        return f"{amount} EUR = {amount * 90:.2f} INR"
    return f"{amount} {from_currency} converted to {to_currency}"


@mcp.tool()
def send_email(recipient: str, subject: str = "", body: str = "") -> str:
    """Sends an email to the given recipient."""
    if not recipient:
        raise ValueError("Missing required field: recipient")
    return f"Email sent to {recipient} with subject '{subject}'."


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
    return f"Created event '{title}'."


@mcp.tool()
def search_emails(sender: str, days_back: int) -> str:
    """Search emails by sender. days_back is the number of days to look back."""
    if days_back <= 0:
        raise ValueError(
            "Invalid value for days_back. Expected a positive number of days."
        )
    return f"Found one email from {sender}: Please schedule a research meeting tomorrow."


@mcp.tool()
def search_tickets(time_range: str = "", start_time: str = "", end_time: str = "") -> str:
    """Search customer tickets. Supports natural-language time ranges like 'past_24_hours'."""
    if time_range and not (start_time and end_time):
        raise ValueError(
            "Unsupported time_range format. Expected ISO-8601 start_time and end_time fields."
        )
    return "Found 3 matching tickets."


@mcp.tool()
def realtime_weather(city: str) -> str:
    """Returns current weather from the live Open-Meteo API for a supported city."""
    coords = CITY_COORDS.get(city.strip().lower())
    if coords is None:
        raise ValueError(f"Unsupported city '{city}'. Supported: {', '.join(sorted(CITY_COORDS))}.")
    latitude, longitude = coords
    query = urllib.parse.urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,wind_speed_10m",
            "timezone": "auto",
        }
    )
    data = _get_json(f"https://api.open-meteo.com/v1/forecast?{query}")
    current = data.get("current", {})
    return (
        f"Open-Meteo current weather for {city}: "
        f"temperature={current.get('temperature_2m')}C, "
        f"wind_speed={current.get('wind_speed_10m')}km/h, "
        f"time={current.get('time')}"
    )


@mcp.tool()
def realtime_exchange_rate(base_currency: str, quote_currency: str) -> str:
    """Returns the latest exchange rate from Frankfurter's public currency API."""
    base = base_currency.strip().upper()
    quote = quote_currency.strip().upper()
    data = _get_json(f"https://api.frankfurter.app/latest?from={base}&to={quote}")
    rate = (data.get("rates") or {}).get(quote)
    if rate is None:
        raise ValueError(f"No rate returned for {base}/{quote}.")
    return f"Frankfurter latest rate on {data.get('date')}: 1 {base} = {rate} {quote}"


@mcp.tool()
def realtime_earthquakes(min_magnitude: float = 4.5) -> str:
    """Returns recent earthquake data from the live USGS GeoJSON feed."""
    data = _get_json("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_day.geojson")
    features = data.get("features", [])
    matches = []
    for feature in features:
        props = feature.get("properties", {})
        magnitude = props.get("mag")
        if magnitude is not None and float(magnitude) >= min_magnitude:
            matches.append(f"M{magnitude} {props.get('place')} at {props.get('time')}")
    if not matches:
        return f"No significant earthquakes at or above M{min_magnitude} in the current USGS daily feed."
    return "Recent significant earthquakes: " + "; ".join(matches[:5])


@mcp.tool()
def realtime_iss_position() -> str:
    """Returns the current International Space Station position from Open Notify."""
    data = _get_json("http://api.open-notify.org/iss-now.json")
    position = data.get("iss_position", {})
    return (
        "Current ISS position: "
        f"latitude={position.get('latitude')}, longitude={position.get('longitude')}, "
        f"timestamp={data.get('timestamp')}"
    )


@mcp.tool()
def public_holidays(country_code: str, year: int) -> str:
    """Returns public holidays from the live Nager.Date public holidays API."""
    code = country_code.strip().upper()
    data = _get_json(f"https://date.nager.at/api/v3/PublicHolidays/{year}/{code}")
    holidays = [f"{item.get('date')} {item.get('localName')}" for item in data[:5]]
    return f"First public holidays for {code} in {year}: " + "; ".join(holidays)


@mcp.tool()
def open_library_search(title: str) -> str:
    """Searches live Open Library records for books matching a title."""
    query = urllib.parse.urlencode({"title": title, "limit": 3})
    data = _get_json(f"https://openlibrary.org/search.json?{query}")
    docs = data.get("docs", [])
    if not docs:
        return f"No Open Library records found for title '{title}'."
    results = []
    for doc in docs[:3]:
        authors = ", ".join(doc.get("author_name", [])[:2]) or "unknown author"
        results.append(f"{doc.get('title')} by {authors}, first_publish_year={doc.get('first_publish_year')}")
    return "Open Library matches: " + "; ".join(results)


if __name__ == "__main__":
    mcp.run()
