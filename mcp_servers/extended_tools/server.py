"""Additional tools for broader failure scenarios, exposed via MCP."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("extended_tools")


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
